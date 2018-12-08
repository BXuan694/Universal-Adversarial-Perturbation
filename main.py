import matplotlib.pyplot as plt
import torchvision.models as models
import os
import numpy as np
import torch
from PIL import Image
import torch.backends.cudnn as cudnn
import sys
from transform_file import transform,cut
from targetmodel import ResNet50_ft, root as PATH_DATASETS
from generate import generate


print('>> Loading network...')
resnet = models.resnet50(pretrained=False)
net = ResNet50_ft(resnet)
net.load_state_dict(torch.load('./checkpoint/ckpt96.498054.t7')['net'])
net.eval()

print('>> Checking dataset...')
if not os.path.exists(PATH_DATASETS):
    print("Data set path wrong. please check!")
    sys.exit()

print('>> Checking devices...')
if torch.cuda.is_available():
    device = 'cuda'
    net.cuda()
    # speed up slightly
    cudnn.benchmark = True
else:
    device = 'cpu'

print('>> Loading perturbation...')
# generate perturbation v of 224*224*3
file_perturbation = 'data/universal.npy'
if os.path.isfile(file_perturbation) == 0:
    print('   >> No perturbation found, computing...')
    v = generate(PATH_DATASETS, 'dataset4u-trn.txt', 'dataset4u-val.txt', net, max_iter_uni=1000, delta=0.2, p=np.inf, num_classes=10, overshoot=0.2, max_iter_df=10)
    # Saving the universal perturbation
    np.save('./data/universal.npy', v)
else:
    print('   >> Found a pre-computed universal perturbation! Retrieving it from', file_perturbation)
    v = np.load(file_perturbation)[0]

testimg = "./data/test_im2.jpg"
print('>> Testing the universal perturbation on',testimg)
labels = open('./data/labels.txt', 'r').read().split('\n')
testimgToInput = Image.open(testimg).convert('RGB')
pertimgToInput = np.clip(cut(testimgToInput)+v,0,255)
pertimg = Image.fromarray(pertimgToInput.astype(np.uint8))

img_orig = transform(testimgToInput)
inputs_orig = img_orig[np.newaxis, :].to(device)
outputs_orig = net(inputs_orig)
_, predicted_orig = outputs_orig.max(1)
label_orig = labels[predicted_orig[0]]

img_pert=transform(pertimg)
inputs_pert=img_pert[np.newaxis, :].to(device)
outputs_pert=net(inputs_pert)
_, predicted_pert = outputs_pert.max(1)
label_pert=labels[predicted_pert[0]]

# Show original and perturbed image
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cut(testimgToInput), interpolation=None)
plt.title(label_orig)

plt.subplot(1, 2, 2)
plt.imshow(pertimg, interpolation=None)
plt.title(label_pert)

plt.savefig("./data/result.png")
plt.show()