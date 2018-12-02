from deepfool import deepfool
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.autograd.gradcheck import zero_gradients
import torch.backends.cudnn as cudnn
import sys
from transform_file import transform, cut, convert
from targetmodel import ResNet50_ft, MyDataset

def generate(path, dataset, testset, net, delta=0.2, max_iter_uni=np.inf, p=np.inf, num_class=10, overshoot=0.2, max_iter_df=10):
    '''

    :param path:
    :param dataset:
    :param testset:
    :param net:
    :param delta:
    :param max_iter_uni:
    :param p:
    :param num_class:
    :param overshoot:
    :param max_iter_df:
    :return:
    '''
    net.eval()
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda()
        cudnn.benchmark = True
    else:
        device = 'cpu'

    dataset = os.path.join(path, dataset)
    testset = os.path.join(path, testset)
    if not os.path.isfile(dataset):
        print("Trainingdata of UAP does not exist, please check!")
        sys.exit()
    if not os.path.isfile(testset):
        print("Testingdata of UAP does not exist, please check!")
        sys.exit()

    img_trn = []
    with open(dataset, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img_trn.append((words[0], int(words[1])))
    img_tst = []
    with open(testset, 'r') as f:
        for line in f:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            img_tst.append((words[0], int(words[1])))
    num_img_trn = len(img_trn)
    num_img_tst = len(img_tst)
    order = np.arange(num_img_trn)
    np.random.shuffle(order)


    v = np.zeros((224, 224, 3), dtype=np.float32)
    fooling_rate = 0.0
    iter = 0
    # start an epoch
    labels = open('./data/label.txt', 'r').read().split('\n')

    while fooling_rate < 1-delta and iter < max_iter_uni:
        print("Starting pass number ", iter)
        # start an iter
        for k in order:
            cur_img = Image.open(img_trn[k][0]).convert('RGB')

            cur_img1 = transform(cur_img)[np.newaxis, :].to(device)
            r2 = int(net(cur_img1).max(1)[1])
            #print(labels[r2])
            #cur_img.show()
            torch.cuda.empty_cache()

            #tmp=cut(cur_img)
            #tmp1=convert(tmp)[np.newaxis, :].to(device)

            per_img =  Image.fromarray(cut(cur_img)+v.astype(np.uint8))
            per_img1 = convert(per_img)[np.newaxis, :].to(device)
            r1 = int(net(per_img1).max(1)[1])
            #print(labels[r1])
            #per_img.show()
            torch.cuda.empty_cache()

            if r1 == r2:
                print(">> k = ", k, ', pass #', iter, end='      ')
                dr, iter_k, label, k_i, pert_image = deepfool(per_img1[0], net)
                print('deepfool result: ', label, k_i)

                if iter_k < max_iter_df-1:
                    v[:, :, 0] += dr[0, 0, :, :]
                    v[:, :, 1] += dr[0, 1, :, :]
                    v[:, :, 2] += dr[0, 2, :, :]

        iter = iter + 1

        with torch.no_grad():
            # Compute fooling_rate
            est_labels_orig = torch.tensor(np.zeros(0, dtype=np.int64))
            est_labels_pert = torch.tensor(np.zeros(0, dtype=np.int64))

            batch = 32

            test_data_orig = MyDataset(txt=testset, transform=transform)
            test_loader_orig = DataLoader(dataset=test_data_orig, batch_size=batch)
            test_data_pert = MyDataset(txt=testset, pert=v, transform=transform)
            test_loader_pert = DataLoader(dataset=test_data_pert, batch_size=batch)

            for batch_idx, (inputs, _) in enumerate(test_loader_orig):
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_orig = torch.cat((est_labels_orig, predicted.cpu()))
            torch.cuda.empty_cache()

            for batch_idx, (inputs, _) in enumerate(test_loader_pert):
                inputs = inputs.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                est_labels_pert = torch.cat((est_labels_pert, predicted.cpu()))
            torch.cuda.empty_cache()

            fooling_rate = float(torch.sum(est_labels_orig != est_labels_pert))/float(num_img_tst)
            print("FOOLING RATE: ", fooling_rate)
            np.save('v'+str(round(fooling_rate, 4)), v)
        #fooling_rate = 1


    #return np.zeros(shape=(224, 224, 3), dtype=np.float32)
    return v

