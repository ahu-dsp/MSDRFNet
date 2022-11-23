import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from tqdm import trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from args import args
import pytorch_msssim
from net11 import FusionModule

def main():
	original_imgs_path= utils.list_images(args.dataset_ir)
	print(len(original_imgs_path))
	train_num =8000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)

	img_flag = False
	alpha_list = [10]
	w_all_list = [[3,0.5]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)

def train(original_imgs_path, img_flag,alpha, w1, w2):
	batch_size = args.batch_size
	deepsupervision = False  # true for deeply supervision
	nb_filter =[16, 48, 168, 512]
	f_type = 'res'
	# nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
	Is_testing = 0
	nest_model = FusionModule(Is_testing,nb_filter)
	nest_model.eval()
	nest_model.cuda()
	if args.resume is not None:
		print('Resuming, initializing using weight from {}.'.format(args.resume))
		nest_model.load_state_dict(torch.load(args.resume))

	optimizer = Adam(nest_model.parameters(), args.lr)
	# optimizer1 = Adam(fusion_model.parameters(), args.lr)
	ssim_loss = pytorch_msssim.msssim
	MSE_fun = torch.nn.MSELoss()
	L1_loss = torch.nn.L1Loss(reduction="mean")
	# L1_loss = torch.nn.L1Loss(reduction="mean")

	if args.cuda:
		nest_model.cuda()
	tbar = trange(args.epochs)
	print('Start training.....')

	Loss_pixel = []
	Loss_ssim = []
	Loss_all = []

	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		nest_model.train()
		count = 0
		# print(e)

		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode='L')
			# MRI数据集
			image_paths_vi = [x.replace('PET1','MRI1') for x in image_paths_ir]
			# print(image_paths_vi)
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode='L')
			count += 1
			# print(count)
			optimizer.zero_grad()
			# optimizer1.zero_grad()
			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)
			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()

			# encoder
			en_ir = nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)

			# fusion_model
			# fs_type = 'add'  # res (low_RFN), add, avg, max, spa, nuclear,attention
			# fusion_strategy = Fusion_strategy(fs_type)
			# f = fusion_strategy(en_ir, en_vi)

			f = nest_model.fusion(en_ir, en_vi)
			# decoder
			output = nest_model.decoder(f, Is_testing=0)
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			EPSILON = 1e-5
			# print(x_ir.shape)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)
			loss1_value = 0.
			loss2_value = 0.

			output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
			output = output * 255
			ssim_loss_temp1 = ssim_loss(output, x_ir, normalize=True)
			ssim_loss_temp2 = ssim_loss(output, x_vi, normalize=True)

			# 1-ssim_loss_temp1:功能信息  1-ssim_loss_temp2:细节信息
			gama = [0.1]
			ssim_loss1 = (1 - gama[0]) * (1 - ssim_loss_temp1)
			ssim_loss2 = (gama[0]) * (1 - ssim_loss_temp2)
			# ssim_loss2 = (1 - ssim_loss_temp2)
			ssim_loss_all = ssim_loss1 + ssim_loss2
			# ssim_loss_all =ssim_loss2
			loss1_value += alpha * ssim_loss_all

			# feature loss
			g2_ir_fea = en_ir
			g2_vi_fea = en_vi
			g2_fuse_fea = f
			w_ir = [w1, w1, w1, w1]
			w_vi = [w2, w2, w2, w2]
			nb_filter = [16, 48, 168, 512]

			F1 = L1_loss(g2_fuse_fea[0], w_ir[0] * g2_ir_fea[0] + w_vi[0] * g2_vi_fea[0]) / nb_filter[0]
			F2 = L1_loss (g2_fuse_fea[1], w_ir[1] * g2_ir_fea[1] + w_vi[1] * g2_vi_fea[1]) / nb_filter[1]
			F3 = L1_loss(g2_fuse_fea[2], w_ir[2] * g2_ir_fea[2] + w_vi[2] * g2_vi_fea[2]) / nb_filter[2]
			F4 = L1_loss(g2_fuse_fea[3], w_ir[3] * g2_ir_fea[3] + w_vi[3] * g2_vi_fea[3]) / nb_filter[3]
			# F=[F1,F2,F3,F4]
			eps = 1e-5
			p1 = torch.exp(F1) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
			p2 = torch.exp(F2) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
			p3 = torch.exp(F3) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)
			p4 = torch.exp(F4) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4) + eps)

			for ii in range(4):
				w_fea = [p1, p2, p3, p4]
				# w_fea = [1, 10, 100,1000]
				g2_ir_temp = g2_ir_fea[ii]
				g2_vi_temp = g2_vi_fea[ii]
				g2_fuse_temp = g2_fuse_fea[ii]
				# Mse=f[ii]
				# IR:RGB VIS:MRI
				fea_loss = w_fea[ii] * L1_loss(g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp)
				loss2_value += fea_loss

			# color_loss = hist_similar(output, x_ir) * 0.001
			# loss3_value += color_loss
			loss1_value /= len(output)
			loss2_value /= len(output)
			# loss3_value /= len(output)
			total_loss = loss1_value + loss2_value
			total_loss.backward()
			optimizer.step()
			# optimizer1.step()
			all_fea_loss += loss2_value.item()
			all_ssim_loss += loss1_value.item()

			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Epoch {}:\t[{}/{}]\t ssim loss: {:.6f}\tpixel loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), e + 1, count, batches,
								  all_fea_loss / args.log_interval,
								   all_ssim_loss / args.log_interval,
								  (all_fea_loss  + all_ssim_loss) / args.log_interval

				)
				tbar.set_description(mesg)
				print(e)
				# 保存所有数据
				Loss_pixel.append(all_fea_loss / args.log_interval)
				Loss_ssim.append(all_ssim_loss / args.log_interval)
				Loss_all.append(( all_ssim_loss +all_fea_loss) / args.log_interval)

				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.
			# 执行完
			if (batch + 1) % (80000 * args.log_interval) == 0:
				# save model
				nest_model.eval()
				nest_model.cuda()
				save_model_filename =  "Epoch_" + str(e) + "_iters_" + str(count) + "_" + ".model"
				save_model_path = os.path.join(args.save_fusion_model_onestage, save_model_filename)
				torch.save(nest_model.state_dict(), save_model_path)

				loss_data_pixel = Loss_pixel

				loss_filename_path = args.save_loss_dir_onestage + "loss_pixel_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" + ".mat"
				scio.savemat(loss_filename_path, {'loss_pixel': loss_data_pixel})
				# SSIM loss
				loss_data_ssim = Loss_ssim
				# for t in range(0, 800):
				# 	writer.add_scalar("loss_data_ssim",Loss_ssim, t)
				loss_filename_path = args.save_loss_dir_onestage + "loss_ssim_epoch_" + str(
					args.epochs) + "_iters_" + str(count) + "_" +".mat"
				scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				# all loss
				loss_data = Loss_all
				# for m in range(0, 800):
				# 	writer.add_scalar("loss_all", Loss_all, m)
				loss_filename_path = args.save_loss_dir_onestage +"loss_all_epoch_" + str(e) + "_iters_" + \
									 str(count) + "-" + ".mat"
				scio.savemat(loss_filename_path, {'loss_all': loss_data})

				nest_model.train()
				nest_model.cuda()
				tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)

	# pixel loss
	loss_data_pixel = Loss_pixel
	loss_filename_path = args.save_loss_dir  + '/' + "Final_loss_fea_epoch_" + str(
		args.epochs) + "_.mat"
	scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_pixel})


	loss_data_ssim = Loss_ssim
	loss_filename_path = args.save_loss_dir +  '/' + "Final_loss_ssim_epoch_" + str(
		args.epochs) + "_" +  ".mat"
	scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})
	# SSIM loss
	loss_data = Loss_all
	loss_filename_path = args.save_loss_dir  +'/' + "Final_loss_all_epoch_" + str(
		args.epochs) + "_" + ".mat"
	scio.savemat(loss_filename_path, {'final_loss_all': loss_data})

	# save model
	nest_model.eval()
	nest_model.cuda()
	save_model_filename1 ="onestage_model_epoch_" + str(args.epochs) + "_" + "batch_size_" +str(args.batch_size) +  ".model"
	save_model_path1 = os.path.join(args.save_fusion_model , save_model_filename1)
	torch.save(nest_model.state_dict(), save_model_path1)
	print("\nDone, trained all_model saved at", save_model_path1)
if __name__ == "__main__":
	main()
