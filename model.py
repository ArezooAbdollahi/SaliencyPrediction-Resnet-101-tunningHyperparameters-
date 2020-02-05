import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# import pytorch_msssim 
class ModelBasedResnet(nn.Module):
	"""Atrous Spatial Pyramid Pooling"""

	def __init__(self,num_classes = 1):
		super(ModelBasedResnet, self).__init__()

		self.upsample_1 = nn.Sequential(
			nn.Conv2d(2048,1024,kernel_size = 1,stride = 1,padding = 0),
			nn.BatchNorm2d(1024),
			Interpolate(),
			)

		
		self.upsample_2 = nn.Sequential(
			nn.Conv2d(1024,512,kernel_size = 1,stride = 1,padding = 0),
			nn.BatchNorm2d(512),
			Interpolate(),
			)

		self.upsample_3 = nn.Sequential(
			nn.Conv2d(512,256,kernel_size = 1,stride = 1,padding = 0),
			nn.BatchNorm2d(256),
			Interpolate(),
			)

		self.upsample_4 = nn.Sequential(
			nn.Conv2d(256,1,kernel_size = 1,stride = 1,padding = 0),
			Interpolate(),
			)



		self.weight_init()
		resnet = models.resnet101(pretrained=True)
		self.num_classes = num_classes
		self.conv1 = resnet.conv1
		self.bn1 = resnet.bn1
		self.relu = resnet.relu
		self.maxpool = resnet.maxpool
		self.layer1 = resnet.layer1
		self.layer2 = resnet.layer2
		self.layer3 = resnet.layer3
		self.layer4 = resnet.layer4

		for n, m in self.layer4.named_modules():
			if 'conv2' in n:
				m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
			elif 'downsample.0' in n:
				m.stride = (1, 1)
			if 'conv1' in n or 'conv3' in n:
				m.dilation, m.stride = (2, 2), (1, 1)
		
		### if you want to try it with other loss functions, uncomment them here 
		
		# self.L1_loss = nn.L1Loss()
		#self.BCELoss= nn.BCELoss(size_average=True)
		# self.MSELoss = nn.MSELoss()
		# self.loss= nn.loss()

	def weight_init(self):

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.kaiming_normal_(m.weight.data)
		

	def freeze_bn(self):
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				m.eval()


	def forward(self, x):

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.layer2(x)
		
		x = self.layer3(x)
		x = self.layer4(x)

		x = F.relu(self.upsample_1(x))
		x = F.relu(self.upsample_2(x))
		x = F.relu(self.upsample_3(x))
		x = F.relu(self.upsample_4(x)) 

		return x



	###KL-divergance loss function 
	def loss(self, y_pred_, y_true_):

		y_true = torch.ge(y_true_, 0.5).float()
		y_pred = y_pred_
		shape_c_out = y_true.shape[-1]  # width
		shape_r_out = y_true.shape[-2]  # height
		ep = 1e-07
		max_y_pred = torch.repeat_interleave(torch.unsqueeze(
			torch.repeat_interleave(torch.unsqueeze(torch.max(torch.max(y_pred, dim=2)[0], dim=2)[0], dim=-1), shape_r_out,
									dim=-1), dim=-1), shape_c_out, dim=-1)
		

		sum_y_true = torch.repeat_interleave(
			torch.unsqueeze(torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum(y_true, dim=2), dim=2), dim=-1),
													shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)

		sum_y_pred = torch.repeat_interleave(torch.unsqueeze(
			torch.repeat_interleave(torch.unsqueeze(torch.sum(torch.sum((y_pred / max_y_pred), dim=2), dim=2), dim=-1),
									shape_r_out, dim=-1), dim=-1), shape_c_out, dim=-1)


		return torch.sum(torch.sum((y_true / (sum_y_true + ep)) * torch.log(
			((y_true / (sum_y_true + ep)) / (((y_pred / max_y_pred) / (sum_y_pred + ep)) + ep)) + ep), dim=-1), dim=-1)



class Interpolate(nn.Module):
	"""Atrous Spatial Pyramid Pooling"""
	def __init__(self,scale_factor = 2,):
		super(Interpolate, self).__init__()
		self.interp = F.interpolate
		self.scale_factor = scale_factor

	def forward(self,x):
		return self.interp(x,scale_factor = self.scale_factor,mode = "bilinear", align_corners = True)
