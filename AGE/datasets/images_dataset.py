import numpy as np
import torch
import os 
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.target_root = target_root     
		self.source_paths = sorted(data_utils.make_dataset(source_root)) # folder of neutral expressions
		self.target_paths = sorted(data_utils.make_dataset(target_root))  # folder of other expressions 
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.average_codes = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
		self.average_codes1 = torch.load(opts.secondary_class_embedding_path, map_location=torch.device("cpu"))   # class embedding for emotions
		self.opts = opts
	def find_matching_target_file(self,from_path):
		cate = from_path.split('/')[-1].split('_')[0]      
		files = [ file for file in self.target_paths if os.path.relpath(file, self.target_root).split('_')[0] == cate ] # find images of the same identity in target folder
		idx = np.random.choice( len(files) , 1)[0] # randomly pick one emotion
		to_path = files[idx]

		cate1 = os.path.relpath(to_path, self.target_root).split('_')[1]     
		return to_path, cate1      
	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		cate = from_path.split('/')[-1].split('_')[0]
		cate1 = from_path.split('/')[-1].split('_')[1] # category for emotion
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')  #if self.opts.label_nc == 0 else from_im.convert('L') 
		to_path, cate1= self.find_matching_target_file(from_path)
		to_im = Image.open(to_path).convert('RGB') #if self.opts.label_nc == 0 else from_im.convert('L')     
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		return from_im, to_im, self.average_codes[cate],self.average_codes1[cate1]