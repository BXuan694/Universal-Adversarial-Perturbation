import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from transform_file import cut

root='/home/wang/Dataset/Caltech256/'
#root='/media/this/02ff0572-4aa8-47c6-975d-16c3b8062013/Caltech256/'

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, pert=np.zeros(1), loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.pert = pert

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.fromarray(np.clip(cut(self.loader(fn))+self.pert,0,255).astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)


class ResNet50_ft(nn.Module):
    def __init__(self, model):
        super(ResNet50_ft, self).__init__()

        self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.Linear_layer = nn.Linear(2048, 257)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer(x)
        return x