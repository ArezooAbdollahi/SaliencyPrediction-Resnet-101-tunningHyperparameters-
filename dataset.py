from PIL import Image
import numpy as np
import torch
import torchvision
from torch.utils import data
import os
from torchvision import datasets
import torchvision.transforms as standard_transforms
import cv2
from random import *

class Inpainting(data.Dataset):
	def __init__(self, root = '/home/arezoo/5-DataSet', mode = 'train'):
		self.root = root
		self.mode = mode
		self.mean_std = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
		self.images_list = []
		self.labels_list = []
		self.files = []
		self.ignore_label = 255


		# standard_transforms.Normalize(*self.mean_std)

		self.input_transform = standard_transforms.Compose([
		standard_transforms.ToTensor(),
		# standard_transforms.Normalize(*self.mean_std)
		
	])
		self.target_transform = standard_transforms.Compose([
		standard_transforms.ToTensor(),
	])

		# Load all path to images
		if self.mode in ['train','val']:

			with open(os.path.join(root, self.mode + '-MultiScaleModifyCat.txt'), 'r') as f:

				file_list = f.readlines()
				file_list = [x.strip() for x in file_list]
				print("len(file_list)", len(file_list))
		self.files.append(file_list)
		print("len(self.files)",len(self.files))
		self.files=self.files[0]
		print("len(self.files0)",len(self.files))


	def __len__(self):
		return len(self.files)

	#transforming images 
	def __getitem__(self, index):

		image_id = self.files[index]
		image, nameImg, label_128, label_64, label_32, label_16  = self._load_data(image_id)

		image = self.input_transform(image)
		label_128 = self.target_transform(label_128)
		label_64 = self.target_transform(label_64)
		label_32 = self.target_transform(label_32)
		label_16 = self.target_transform(label_16)

		return image, nameImg, label_128, label_64, label_32, label_16

	# loading data from the dataset	
	def _load_data(self, image_id):

		input_id, label_id_128, label_id_64, label_id_32, label_id_16=image_id.split(',')
		
		nameDir= input_id.split('/')[0]
		indexImg= input_id.split('/')[1].split('.')[0]
		nameImg= nameDir+'-'+indexImg

		image = cv2.imread(os.path.join(self.root,'Saliencyresize-total',input_id))
		label_128 = cv2.imread(os.path.join(self.root,'HeatmapGT',label_id_128), 0)/255.0
		label_64 = cv2.imread(os.path.join(self.root,'HeatmapGT',label_id_64), 0)/255.0
		label_32 = cv2.imread(os.path.join(self.root,'HeatmapGT',label_id_32), 0)/255.0
		label_16 = cv2.imread(os.path.join(self.root,'HeatmapGT',label_id_16), 0)/255.0

		return image, nameImg, label_128, label_64, label_32, label_16
   
