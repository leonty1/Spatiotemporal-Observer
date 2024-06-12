import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class PaddingRadar_(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((128,128,1))
        padding_data[13:-14,13:-14,:] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255.0
        return imgs,self.dirs[index]


class PaddingRadar(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((128,128,1))
        padding_data[13:-14,13:-14,:] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]        #[1, 101, 101]
            img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)      ## S*C*H*W   [15, 1, 101, 101]
        return imgs

class Radar(Dataset):
    def __init__(self,data_type,data_root='train'):
        self.data_type = data_type
        self.data_root = data_root # [train , valid , test]
        self.dirs = os.listdir("{}".format(os.path.join(self.data_root,self.data_type)))
    def __len__(self):
        return len(self.dirs)

    def padding_img(self,data):
        padding_data = np.zeros((1,128,128))
        padding_data[:,13:-14,13:-14] = data
        return padding_data

    def __getitem__(self, index):
        cur_fold = os.path.join(self.data_root,self.data_type,self.dirs[index])
        files = os.listdir(cur_fold)
        files.sort()
        imgs = []
        for i in range(len(files)):
            file = 'img_'+str(i+1)+'.png'
            img_path = os.path.join(cur_fold,file)
            img = cv2.imread(img_path,0)[:,:,np.newaxis]
            #img = cv2.resize(img, (101, 101))
            # img = self.padding_img(img)
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = imgs.astype(np.float32)/255
        imgs = imgs.reshape((15, 1, -1, -1))
        input = imgs[0:5,...]
        output = imgs[5:10,...]
        return input, output

def load_cikm_radar(data_root):
    data_root = data_root  #'.\CIKM2017''
    train_data = Radar(data_type='train',data_root=data_root)
    valid_data = Radar(data_type='validation',data_root=data_root)
    test_data = Radar(data_type='test',data_root=data_root)
    print(f'train set has {len(train_data)} sequences')
    print(f'valid set has {len(valid_data)} sequences')
    print(f'test set has {len(test_data)} sequences')
    return train_data, valid_data, test_data

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_set,val_set, test_set = load_cikm_radar(data_root)


    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_validation, dataloader_test, 0, 1