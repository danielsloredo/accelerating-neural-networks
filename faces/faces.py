import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
import sys
import os
from scheduler import *
import scipy.io as sio

def my_loss(array1_t,array2_t):
    loss_function = torch.nn.MSELoss(reduction='none')
    return loss_function(array1_t,array2_t).sum(1).mean()

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(25*25, 2000),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2000, 1000),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1000, 500),
            torch.nn.Sigmoid(),
            torch.nn.Linear(500, 30)
        )

        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(30, 500),
            torch.nn.Sigmoid(),
            torch.nn.Linear(500, 1000),
            torch.nn.Sigmoid(),
            torch.nn.Linear(1000, 2000),
            torch.nn.Sigmoid(),
            torch.nn.Linear(2000, 25*25),
            #torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# takes in a module and applies the specified weight initialization
def weights_init_sparse(m):
    classname = m.__class__.__name__
    try:
        m.weight.data.fill_(0)
        s = m.weight.data.size()
        print(s[1])
        for i in range(s[0]):
            list_ind = random.sample(range(s[1]), 15)
            for j in list_ind:
                m.weight.data[i,j] = random.gauss(0, 1)
        print('{} nonzeros out of {}'.format(torch.sum(m.weight.data != 0),s[1]*s[0]))
        m.bias.data.fill_(0)
    except:
        print('{} has no weight'.format(classname))
    


if __name__ == '__main__':
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    images_ = sio.loadmat('newfaces_rot_single.mat')
    images_ = np.asarray(images_['newfaces_single'])
    images_ = np.transpose(images_)
    train_images = images_[:103500]
    batch_size = 200
    train_images = torch.from_numpy(train_images).float()
        

    train_labels = train_images
    faces_data = torch.utils.data.TensorDataset(train_images, train_labels)
    trainloader = torch.utils.data.DataLoader(dataset=faces_data, batch_size = batch_size, shuffle = True)
    
    trainloader2 = torch.utils.data.DataLoader(dataset = faces_data, batch_size = 103500)
    for img,_ in trainloader2:
        testTensor = img.to(device)
    
    lam_max = float(sys.argv[1])
    step = float(sys.argv[2])
    alg = ['sgd', 'pol', 'nes']
    num_epoch = 1450
    num_iter = int(np.ceil(len(faces_data)/batch_size))
    
    loss_all = np.zeros((3,5*num_epoch+1))
    final_loss = np.zeros(3)
    weight_list_final = []

    momentums = [min(lam_max,1-2**(-1-np.log2(np.floor(x/250)+1))) for x in range(num_iter*num_epoch)]
    momentums[-1000:]=[0.9 for x in range(1000)]
    net_init = Net()
    
    if not os.path.exists("net_init_faces.pt"):
        net_init.apply(weights_init_sparse)
        torch.save(net_init.state_dict(), "net_init_faces.pt")
    else:
        net_init.load_state_dict(torch.load("net_init_faces.pt"))

    weight_list_init = []
    for param in net_init.parameters():
        weight_list_init.append(param.data.cpu().numpy())
    
    loss_function = torch.nn.MSELoss()
    for ind_alg in alg:
        print('running max_mom {} step {} and algorithm {}'.format(lam_max,step,ind_alg))
        net = copy.deepcopy(net_init).to(device)
        if ind_alg == 'sgd':
            lrs = [step/(1-m) for m in momentums]
            optimizer = optim.SGD(net.parameters(), lr=lrs[0], momentum=0)
            scheduler_ad = ListScheduler(optimizer, lrs = lrs, momentums = [0 for x in range(num_iter*num_epoch)])
            idx_alg = 0
        if ind_alg == 'pol':
            optimizer = optim.SGD(net.parameters(), lr=step, momentum=momentums[0])
            scheduler_ad = ListScheduler(optimizer, lrs = [step for x in range(num_iter*num_epoch)],momentums = momentums)
            idx_alg = 1
        if ind_alg == 'nes':
            optimizer = optim.SGD(net.parameters(), lr=step, momentum=momentums[0], nesterov= True)
            scheduler_ad = ListScheduler(optimizer, lrs = [step for x in range(num_iter*num_epoch)],momentums = momentums)
            idx_alg = 2
        j = 0
        
        with torch.no_grad():
            net.eval()
            pred = net(testTensor)
            test_loss = my_loss(pred, testTensor)
            print('Initial loss: %.3f' % (test_loss))
            loss_all[idx_alg,0] = test_loss
            net.train()
        
        blowup = False
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            i = 1
            for (image, _) in trainloader:
                
                image = image.to(device)
                                
                reconstructed = net(image)

                loss = loss_function(reconstructed, image)

                optimizer.zero_grad()
                
                loss.backward()
                
                optimizer.step()
          
                if torch.isnan(loss):
                    blowup = True
                    print("Diverge!")
                    break
                
                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    print('lr: %.6f, mom: %.6f' % (optimizer.param_groups[0]['lr'],optimizer.param_groups[0]['momentum']))
                    j = j + 1
                    loss_all[idx_alg,j] = running_loss / 100
                    running_loss = 0.0
                
                scheduler_ad.step()
                i = i + 1
            if blowup:
                break
        weight_list = []
        for param in net.parameters():
            weight_list.append(param.data.cpu().numpy())
        weight_list_final.append(weight_list)
        with torch.no_grad():
            net.eval()
            pred = net(testTensor)
            final_loss[idx_alg] = my_loss(pred, testTensor).cpu()
            print('Final loss: %.3f' % (final_loss[idx_alg]))
            net.train()

    np.savez("{}_{}_ae_faces".format(lam_max,step)+'_final.npz', loss_all = loss_all,
             weight_list_init = weight_list_init, weight_list_final = weight_list_final, final_loss = final_loss)