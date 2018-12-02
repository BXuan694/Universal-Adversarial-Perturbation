import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from transform_file import transform

net = models.resnet34(pretrained=True)
net.eval()

im_orig = Image.open('./data/test_im2.jpg')

im = transform(im_orig)
r, loop_i, label_orig, label_pert, pert_image = deepfool(im, net)

labels = open(os.path.join('./data/synset_words.txt'), 'r').read().split('\n')

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

clip = lambda x: clip_tensor(x, 0, 255)


tf = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=map(lambda x: 1 / x, [0.229,0.224,0.225])),
                        transforms.Normalize(mean=map(lambda x: -x, [0.485,0.456,0.406]), std=[1, 1, 1]),
                        transforms.Lambda(clip),
                        transforms.ToPILImage(),
                        transforms.CenterCrop(224)])

plt.figure()
plt.imshow(tf(pert_image.cpu()[0]))
plt.title(str_label_pert)
plt.show()
