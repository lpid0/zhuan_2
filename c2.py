import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image
import re

b9 = [(0,0,113,113), (115,0,228,113), (231,0,344,113), 
      (0,115,113,228), (115,115,228,228), (231,115,344,228), 
      (0,231,113,344), (115,231,228,344), (231,231,344,344)]
p1 = (0,344,45,384)

def compute_divisibility(n):
    count = 0
    while n >= 9:
        n -= 9
        count += 1
    remainder = n
    return count, remainder

class ImageDataset(Dataset):
    def __init__(self, filepath, train=True):
        self.train = train
        self.fnames = [filepath + '/' + filename for filename in os.listdir(filepath)] # 读取所有图片的文件路径
        self.compose = transforms.Compose([  # 图片的transform
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                        ])
    
    def __len__(self):
        len1 = len(self.fnames)
        return len1*9
    
    def __getitem__(self, index1):
        count, remainder = compute_divisibility(index1)
        '''if self.m == 8:
            self.i += 1
            self.m = 0
        if self.i >= len(self.fnames):
            self.i = 0'''

        match = re.search(r'_index(\d+)', self.fnames[count])
        index = match.group(1)
        index_list = [int(digit) for digit in index]

        '''match = re.search(r'_index(\d+)', self.fnames[self.i])
        index = match.group(1)
        index_list = [int(digit) for digit in str(int(index))]'''
        img = Image.open(self.fnames[count]).convert('RGB')
        img1 = img.crop(b9[remainder]).resize((32, 32))
        img2 = img.crop(p1).resize((32, 32))
        tag = 1 if remainder in index_list else 0
        return self.compose(img1), self.compose(img2), torch.tensor(tag).float()
