import torch.nn as nn
import torchvision.models as models
import os
import numpy as np
import torch
from transform_file import transform
from torch.utils.data import DataLoader
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import sys
import math
from targetmodel import ResNet50_ft, MyDataset, root
'''
DATA_PATH="/media/this/02ff0572-4aa8-47c6-975d-16c3b8062013/Caltech256/256_ObjectCategories"
if not os.path.exists(DATA_PATH):
    print("No dataset folder found, please check.")
    sys.exit()

dirs = [x[0] for x in os.walk(DATA_PATH)][1:]             # list of classes
dirs = sorted(dirs)                                       # (same order as list.txt, full path)
#print(dirs)
num_classes=257
trn_size=70
val_size=7

it = 0
Matrix = [0 for x in range(num_classes)]                # all filenames under DATA_PATH
val_data=[0 for x in range(num_classes)]
trn_data=[0 for x in range(num_classes)]
for d in dirs:
    for _, _, filename in os.walk(d):
        Matrix[it] = filename  # filename is a list of pic files under the fold
        val_data[it] = filename[0:val_size]
        trn_data[it] = filename[val_size:val_size+trn_size]
    it = it + 1

#print(Matrix)
#print(len(trn_data[0]))
#print(len(val_data[0]))
'''

if not os.path.exists(root):
    print("No dataset found, please check!")
    sys.exit()

trn_batch=16
val_batch=64
train_data = MyDataset(txt=root+'dataset-trn.txt', transform=transform)
test_data = MyDataset(txt=root+'dataset-val.txt', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=trn_batch, pin_memory=True, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=val_batch, pin_memory=True)

resnet = models.resnet50(pretrained=False)
net = ResNet50_ft(resnet)
#best_acc=0
loadfile=torch.load('./checkpoint/ckpt96.498054.t7')
net.load_state_dict(loadfile['net'])
best_acc=loadfile['acc']
print("the accuracy of current model: ", best_acc)

if torch.cuda.is_available():
    device = 'cuda'
    net.cuda()
    # speed up slightly
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.005)


def train(epoch, trn_batch):
    print('\nEpoch: %d:' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader)
    for (batch_idx, (inputs, targets)) in enumerate(pbar):
        pbar.set_description("batch " + str(batch_idx) + '/' + str(math.ceil(28037 / trn_batch)))
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("loss: "+str(round(train_loss/trn_batch,4))+'\t'+str(correct)+'\t'+str(total)+'\t'+str(correct/total))


def test(epoch, val_batch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for batch_idx, (inputs, targets) in enumerate(pbar):
            pbar.set_description("batch " + str(batch_idx) + '/' + str(math.ceil(2570 / val_batch)))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("loss: "+str(round(test_loss/val_batch,4))+'\t'+str(correct)+'\t'+str(total)+'\t'+str(correct/total))

    acc = correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt'+str(round(acc,6))+'.t7')
        best_acc = acc

def demo(path):
    labels = open('./data/labels.txt', 'r').read().split('\n')
    print(labels)
    net.eval()
    img=Image.open(path).convert('RGB')
    img=transform(img)[np.newaxis,:].to(device)
    _,out=net(img).max(1)
    print(labels[out[0]])


test(0, val_batch)
demo('./data/test_im3.png')
'''
for epoch in range(10):
    train(epoch, trn_batch)
    test(epoch, val_batch)
    torch.cuda.empty_cache()
'''

