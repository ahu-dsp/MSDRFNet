# Training a NestFuse network
# auto-encoder
import os

import utils
from net_f import Fusion_network
from load_model1 import load_model1
import sys
import time
from tqdm import trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import pytorch_msssim
# from fusion_strategy_chn import device

# from fusion_strategy1 import attention_fusion_weight
EPSILON = 1e-5
from args import  args
def main():
	original_imgs_path= utils.list_images(args.dataset_ir)
	# print(original_imgs_path)
	print(len(original_imgs_path))
	print(args.batch_size)
	train_num=22000

	# train_num=6400
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	img_flag = False
	alpha_list = [10]
	print(alpha_list)
	w_all_list = [[3,1]]
	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)

def train(original_imgs_path, img_flag,alpha, w1, w2):
	batch_size = args.batch_size
	# load network model

	nb_filter =[16, 48, 168, 512]
	f_type = 'res'
	img_flag=False
	with torch.no_grad():
		path = '/Share/home/Z21301084/test/RFN1/UN1/model/function/8-P-stride.model'
		# path = 'model/8-P-stride.model'
		nest_model=load_model1(path)
		# print(nest_model)
		fusion_model = Fusion_network(nb_filter, f_type)
		# print(fusion_model)
		fusion_model.cuda()
		fusion_model.eval()
	if args.resume_fusion_model is not None:
		print('Resuming, initializing layer net using weight from {}.'.format(args.resume_fusion_model))
		fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
	# optimizer = SGD(fusion_model.parameters(), args.lr)
	optimizer = Adam(fusion_model.parameters(), args.lr)
	print(args.lr)
	MSE_fun = torch.nn.MSELoss()
	L1_loss = torch.nn.L1Loss(reduction="mean")

	# ssim_loss = pytorch_msssim.ssim
	ssim_loss = pytorch_msssim.msssim

	if args.cuda:
		nest_model.cuda()
		fusion_model.cuda()

	tbar = trange(args.epochs)
	print('Start training.....')

	temp_path_model = os.path.join(args.save_fusion_model)
	temp_path_loss= os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	# temp_path_model_w = os.path.join(args.save_fusion_model, str(alpha))
	# temp_path_loss_w  = os.path.join(args.save_loss_dir, str(alpha))
	a='Mish'
	temp_path_model_w = os.path.join(args.save_fusion_model, str(a))
	temp_path_loss_w = os.path.join(args.save_loss_dir, str(a))

	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	Loss_feature = []
	Loss_ssim = []
	Loss_color=[]

	Loss_all = []
	count_loss = 0

	all_ssim_loss = 0.
	all_fea_loss = 0.
	all_color_loss=0.

	for e in tbar:
		print('Epoch %d.....' % e)
		# print(e)
		image_set_ir, batches = utils.load_dataset(original_imgs_path , batch_size)
		# image_set_vi, batches = utils.load_dataset(dataset_vi, batch_size)
		# print(batches)
		fusion_model.train()
		count = 0

		# batches:迭代总轮数
		for batch in range(batches):
			# YSPECT
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images_auto(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, mode='L')
			# image_paths_vi = [x.replace('SPECT_Y','MRI') for x in image_paths_ir]
			image_paths_vi = [x.replace('SPECT', 'MRI') for x in image_paths_ir]
			img_vi = utils.get_train_images_auto(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, mode='L')
			count += 1
			optimizer.zero_grad()

			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)

			if args.cuda:
				img_ir = img_ir.cuda()
				img_vi = img_vi.cuda()
			# get layer image
			# encoder
			en_ir = nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)
			# fusion_model
			f = fusion_model(en_ir, en_vi)

			# decoder
			output = nest_model.decoder(f,Is_testing=0)
			# print(output.shape)

			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			# print(x_ir.shape)
			# PET
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			loss1_value = 0.
			loss2_value = 0.
			loss3_value = 0.
			# for output in outputs:
			# 数据的归一化
			# for output in outputs:
			output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
			output = output * 255
			# PET
			# ssim_loss_all = ssim_loss(output1,output, normalize=True)
			ssim_loss_temp1 = ssim_loss(output, x_ir,normalize=True)
			ssim_loss_temp2 = ssim_loss(output, x_vi,normalize=True)
			# 1-ssim_loss_temp1:功能信息  1-ssim_loss_temp2:细节信息
			gama = [0.1]
			ssim_loss1 =(1-gama[0])*(1 - ssim_loss_temp1)
			ssim_loss2 = (gama[0])*(1 - ssim_loss_temp2)
			ssim_loss_all=ssim_loss1+ssim_loss2
			loss1_value += alpha * ssim_loss_all
			# feature loss
			g2_ir_fea = en_ir
			g2_vi_fea = en_vi
			g2_fuse_fea = f
			w_ir = [w1, w1, w1, w1]
			w_vi = [w2, w2, w2, w2]
			# nb_filter = [16, 48, 168, 512]

			F1 = L1_loss(g2_fuse_fea[0], w_ir[0] * g2_ir_fea[0] + w_vi[0] * g2_vi_fea[0])
			F2 = L1_loss(g2_fuse_fea[1], w_ir[1] * g2_ir_fea[1] + w_vi[1] * g2_vi_fea[1])
			F3 = L1_loss(g2_fuse_fea[2], w_ir[2] * g2_ir_fea[2] + w_vi[2] * g2_vi_fea[2])
			F4 = L1_loss(g2_fuse_fea[3], w_ir[3] * g2_ir_fea[3] + w_vi[3] * g2_vi_fea[3])

			eps=1e-5
			p1=torch.exp(F1)/(torch.exp(F1)+torch.exp(F2)+torch.exp(F3)+torch.exp(F4)+eps)
			p2 = torch.exp(F2) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4)+eps)
			p3 = torch.exp(F3) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4)+eps)
			p4 = torch.exp(F4) / (torch.exp(F1) + torch.exp(F2) + torch.exp(F3) + torch.exp(F4)+eps)


			for ii in range(4):
				w_fea = [p1, p2, p3, p4]
				# w_fea = [1, 10, 100, 1000]
				g2_ir_temp = g2_ir_fea[ii]
				g2_vi_temp = g2_vi_fea[ii]
				g2_fuse_temp = g2_fuse_fea[ii]

				# IR:RGB VIS:MRI
				# fs_type = 'channel'  #  avg, max, nuclear,attention.channel,
				# fusion_strategy = Fusion_strategy(fs_type)
				# V1 = fusion_strategy(g2_ir_temp, g2_vi_temp)
				# V1=attention_fusion_weight(g2_ir_temp,g2_vi_temp,p_type="l1_mean")
				fea_loss = w_fea[ii] * L1_loss(g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp)
				# fea_loss = w_fea[ii]*hist_similar(g2_fuse_temp, w_ir[ii] * g2_ir_temp + w_vi[ii] * g2_vi_temp)
				# fea_loss = w_fea[ii] * MSE_fun(g2_fuse_temp, V1)
				loss2_value += fea_loss

			loss1_value /= len(output)
			loss2_value /= len(output)
			# loss3_value /= len(output)
			total_loss = loss1_value + loss2_value
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item()
			all_ssim_loss += loss1_value.item()
			# all_color_loss += loss3_value.item()
			# 训练过程种颜色偏差严重Lcolor


			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t detail loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
									# all_color_loss / args.log_interval,
								  (all_fea_loss + all_ssim_loss) / args.log_interval
				)
				print(e)
				tbar.set_description(mesg)

				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				# Loss_color.append(all_color_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.

				# if (batch + 1) % (200000 * args.log_interval) == 0:
				#
				# 	# save model
				# 	fusion_model.eval()
				# 	fusion_model.cuda()
				# 	save_model_filename = "RGB_Epoch_" + str(e) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".model"
				# 	save_model_path = os.path.join(temp_path_model, save_model_filename)
				# 	torch.save(fusion_model.state_dict(), save_model_path)
				#
				# 	# save loss YSPECT1
				# 	# -----------------------------------------------------
				# 	# pixel loss
				# 	loss_data_ssim = Loss_ssim
				# 	loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	scio.savemat(loss_filename_path, {'loss_ssim': loss_data_ssim})
				#
				# 	# SSIM loss
				# 	loss_data_fea = Loss_feature
				# 	loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	scio.savemat(loss_filename_path, {'loss_fea': loss_data_fea})
				#
				# 	# grd loss
				# 	# loss_data_grd = Loss_grd
				# 	# loss_filename_path = temp_path_loss_w + "/loss_ssim_epoch_" + str(args.epochs) + "_iters_" + str(
				# 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	# scio.savemat(loss_filename_path, {'loss_grd': loss_data_grd})
				#
				# 	# color loss
				# 	# loss_data_hist = Loss_hist
				# 	# loss_filename_path = temp_path_loss_w + "/loss_fea_epoch_" + str(args.epochs) + "_iters_" + str(
				# 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	# scio.savemat(loss_filename_path, {'loss_hist': loss_data_hist})
				#
				# 	# color loss
				# 	# loss_data_mse = Loss_mse
				# 	# loss_filename_path = temp_path_loss_w + "/loss_mse_epoch_" + str(args.epochs) + "_iters_" + str(
				# 	# 	count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	# scio.savemat(loss_filename_path, {'loss_mse': loss_data_mse})
				#
				# 	# all loss
				# 	loss_data = Loss_all
				# 	loss_filename_path = temp_path_loss_w + "/loss_all_epoch_" + str(args.epochs) + "_iters_" + str(count) + "_alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + ".mat"
				# 	scio.savemat(loss_filename_path, {'loss_all': loss_data})
				#
				# 	# fontP = FontProperties()
				# 	# fontP.set_size('large')
				# 	# plt.plot(loss_data, 'b', label='$Loss_all')
				# 	# plt.plot(loss_data_ssim, 'c', label='$ssim$')
				# 	# plt.plot(loss_data_fea, 'c', label='$loss_fea$')
				# 	# plt.xlabel('epoch', fontsize=15)
				# 	# plt.ylabel('Loss values', fontsize=15)
				# 	# plt.legend(loc=2, prop=fontP)
				# 	# # plt.title('FunFuseAn $\lambda = 0.8, \gamma_{ssim} = 0.5, \gamma_{l2} = 0.5$', fontsize='15')
				# 	# plt.savefig('./results/1oss.png')
				#
				# 	fusion_model.train()
				# 	fusion_model.cuda()
				# 	tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
				# 	print(Loss_ssim)
				# 	# writer = SummaryWriter("logs")
				# 	# writer.add_scalar("ssim_loss", Loss_ssim, batch)
				# 	# writer.add_scalar("fea_loss", Loss_feature, batch)
				# 	# writer.add_scalar("all_loss", Loss_all, batch)

		# 五种 loss
		loss_data_ssim = Loss_ssim
		loss_filename_path = temp_path_loss_w + "/Final_loss_ssim_epoch_" + str(
			args.epochs) + "_lamda_" + str(alpha)  +"_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
		scio.savemat(loss_filename_path, {'final_loss_ssim': loss_data_ssim})

		loss_data_fea = Loss_feature
		loss_filename_path = temp_path_loss_w + "/Final_loss_fea_epoch_" + str(
			args.epochs) + "_lamda_" + str(alpha)  + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
		scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})
		# scio.savemat(loss_filename_path, {'final_loss_fea': loss_data_fea})

		# Total loss
		loss_data = Loss_all
		loss_filename_path = temp_path_loss_w + "/Final_loss_all_epoch_" + str(
			args.epochs) + "_lamda_" + str(alpha)  + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-gama0.8.mat"
		scio.savemat(loss_filename_path, {'final_loss_all': loss_data})

		# save model
		fusion_model.eval()
		fusion_model.cuda()
		save_model_filename = "alpha_" + str(alpha) + "_wir_" + str(w1) + "_wvi_" + str(w2) + "-SPECT-gama0.1-8-P.model"
		save_model_path = os.path.join(temp_path_model_w, save_model_filename)
		torch.save(fusion_model.state_dict(), save_model_path)
		print("\nDone, trained model saved at", save_model_path)

def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)

if __name__ == "__main__":
	main()





