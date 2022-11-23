import numpy as np
import torch
from net_f import Fusion_network
from net11 import FusionModule


# 加载AE-model
def load_model1(path):
	nb_filter = [16, 48, 168, 512]
	Is_testing=1
	nest_model =FusionModule(Is_testing)
	nest_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))
	nest_model.eval()
	nest_model.cuda()
	return nest_model

# 加载DRFN-model
def load_model2(path):
	fs_type = 'res'
	# nb_filter = [8, 24, 84, 256]
	nb_filter = [16, 48, 168, 512]
	RFN_model =Fusion_network(nb_filter, fs_type)
	# print(RFN_model)
	RFN_model.load_state_dict(torch.load(path))
	para = sum([np.prod(list(p.size())) for p in RFN_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(RFN_model._get_name(), para * type_size / 1000 / 1000))
	RFN_model.eval()
	RFN_model.cuda()
	return RFN_model