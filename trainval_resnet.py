from __future__ import absolute_import, division, print_function
import random
import numpy as np
import os.path 
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torch.backends import cudnn
import torchvision.utils
from dataset import Inpainting
import datetime
import time

import torch.nn.init as init
from tqdm import tqdm
from model import *

from utils_saliency.salgan_utils import load_image, postprocess_prediction
from utils_saliency.salgan_utils import normalize_map

from IPython import embed
from evaluation.metrics_functions import AUC_Judd, AUC_Borji, AUC_shuffled, CC, NSS, SIM, EMD




cudnn.benchmark = True
manual_seed=627937
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)

def PSNR(mse):
	psnr = 20 * (255.0 /torch.sqrt(mse)).log10()
	return psnr

def normalizeOutout(output):
	output = output - output.min()
	output = output / output.max()
	return output

			
def CalculateMetrics(fground_truth, mground_truth, predicted_map):


	predicted_map = normalize_map(predicted_map)
	predicted_map = postprocess_prediction(predicted_map, (predicted_map.shape[0], predicted_map.shape[1]))
	predicted_map = normalize_map(predicted_map)
	predicted_map *= 255

	fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
	predicted_map = cv2.resize(predicted_map, (0,0), fx=0.5, fy=0.5)
	mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)

	fground_truth = fground_truth.astype(np.float32)/255
	predicted_map = predicted_map.astype(np.float32)
	mground_truth = mground_truth.astype(np.float32)

	AUC_judd_answer = AUC_Judd(predicted_map, fground_truth)
	AUC_Borji_answer = AUC_Borji(predicted_map, fground_truth)
	nss_answer = NSS(predicted_map, fground_truth)
	cc_answer = CC(predicted_map, mground_truth)
	sim_answer = SIM(predicted_map, mground_truth)

	return AUC_judd_answer, AUC_Borji_answer, nss_answer, cc_answer, sim_answer


def main():

	writer = SummaryWriter(comment='Multi-scale-resnet-Train, OriginalImg(input)=Cat2000, GT:4 seperated Heatmap, 20 Categories, split:1800,200, resnet101. 150epoch, lr=1e-5, Adam, loss:KL-div')

	model=ModelBasedResnet().to('cuda:0') 
	model.train()
	mode='train' 
	

	val_dataset = Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='val' 
	)

	train_dataset = Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='train' 
	)

		train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=16,
		num_workers=32,
		shuffle=False,
	)

	val_loader = torch.utils.data.DataLoader(
		dataset=val_dataset,
		batch_size=16,
		num_workers=32,
		shuffle=False,
	)
	print(val_loader.__len__())
	print(train_loader.__len__())
	optimizer = torch.optim.Adam(model.parameters(), lr= 1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

	global_train_iter = 0
	global_val_iter = 0
	for epoch in tqdm(range(150),desc='epoch: '):
	
		folderpath=os.path.join('./outputImages', str(epoch))
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)

		###Pass data to training
		print("epoch training: " , epoch)
		model= train(model,train_loader,optimizer,epoch,writer,global_train_iter)

		###Saveing the model
		modelpath=os.path.join('./models',str("model-")+str(epoch) +".pt")
		torch.save(model.state_dict(), modelpath)

		global_train_iter +=1

		###pass data to validation step

		folderpath=os.path.join('./outputImagesVal', str(epoch))
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)
		print("epoch validation: " , epoch)
		Val(model,val_loader,epoch,writer,global_val_iter)

		scheduler.step()
		global_val_iter += 1


def Val(model,val_loader,epoch,writer,global_val_iter):
	sum_loss = 0.0
	model.eval()
	length=val_loader.__len__()
	
	folderpath=os.path.join('./outputImagesVal', str(epoch))
	for i, data in tqdm(enumerate(val_loader), desc = "validating: "):
		with torch.no_grad():

			image, nameImg, label_id_128, label_id_64, label_id_32, label_id_16 = data
			lenInputImgs = image.shape[0]

			image = Variable(image).to('cuda:0')
			output=model(image)

			label_id_128 = label_id_128.to('cuda:0')
			label_id_64 = label_id_64.to('cuda:0')
			label_id_32 = label_id_32.to('cuda:0')
			label_id_16 = label_id_16.to('cuda:0')
			loss_1 = model.loss(output, label_id_128)
			loss_2 = model.loss(output, label_id_64)
			loss_3 = model.loss(output, label_id_32)
			loss_4 = model.loss(output, label_id_16)

			loss = torch.sum(loss_1)+ torch.sum(loss_2)+ torch.sum(loss_3)+ torch.sum(loss_4)

			
			sum_loss += loss.item()

	avg_loss = sum_loss / 200
	writer.add_scalar('loss/val_loss',avg_loss,global_val_iter)
	print("val_loss: ", avg_loss)


def train(model,train_loader,optimizer,epoch,writer,global_train_iter):

	folderpath=os.path.join('./outputImages', str(epoch))
	model.train()

	length=train_loader.__len__()
	sum_psnr=0.0
	sum_loss = 0.0
	sum_l1_loss = 0.0


	for i, data in tqdm(enumerate(train_loader), desc= "training"):
	
		image, nameImg, label_id_128, label_id_64, label_id_32, label_id_16 = data

		lenInputImgs = image.shape[0]

		sizebatch=image.size()
		NumBatch= sizebatch[0]
		optimizer.zero_grad()
		model.zero_grad()
		image = Variable(image).to('cuda:0')
		output = model(image)

		label_id_128 = label_id_128.to('cuda:0')
		label_id_64 = label_id_64.to('cuda:0')
		label_id_32 = label_id_32.to('cuda:0')
		label_id_16 = label_id_16.to('cuda:0')
		loss_1 = model.loss(output, label_id_128)
		loss_2 = model.loss(output, label_id_64)
		loss_3 = model.loss(output, label_id_32)
		loss_4 = model.loss(output, label_id_16)

		loss = torch.sum(loss_1)+ torch.sum(loss_2)+ torch.sum(loss_3)+ torch.sum(loss_4)

		loss.backward()
		optimizer.step()
		sum_loss += loss.item()
	
	print("length", length)	
	avg_loss = sum_loss / 1800
	writer.add_scalar('loss/train_loss',avg_loss,global_train_iter)
	print("train_loss: ", avg_loss)


	return model

if __name__ == '__main__':
	main()
