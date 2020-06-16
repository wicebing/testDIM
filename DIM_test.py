import os
import time
import unicodedata
import random
import string
import re
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

CIFAR10 = torchvision.datasets.CIFAR10(
    './CIFAR10', train=True, download=True)

CIFAR10_test= torchvision.datasets.CIFAR10(
    './CIFAR10', train=False, download=True)

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, ds, transform=None, target_transform=None, loader=default_loader):
        self.imgs = ds
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        img, label = self.imgs[index][0], self.imgs[index][1]
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

batch_size = 4
lr = 1e-5
checkpoint_file = './checkpoint' 
parallel= False
device = 'cuda'

transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])

train_data=MyDataset(ds=CIFAR10, transform=transform)
data_loader_train = DataLoader(train_data, batch_size=batch_size,shuffle=True)

test_data=MyDataset(ds=CIFAR10_test, transform=transform)
data_loader_test = DataLoader(test_data, batch_size=batch_size,shuffle=False)


def save_checkpoint(checkpoint_file,checkpoint_path, model, parallel, optimizer=None):
    if parallel:
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in model.state_dict().items():
            name = k[7:] # remove module.
            state_dict[name] = v
    else:
        state_dict = model.state_dict()

    state = {'state_dict': state_dict,}
             #'optimizer' : optimizer.state_dict()}
    torch.save(state, os.path.join(checkpoint_file,checkpoint_path))

    print('model saved to %s / %s' % (checkpoint_file,checkpoint_path))
    
def load_checkpoint(checkpoint_file,checkpoint_path, model):
    state = torch.load(os.path.join(checkpoint_file,checkpoint_path),
                       map_location='cuda:0'
#                       map_location={'cuda:0':'cuda:1'}
                       )
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s / %s' % (checkpoint_file,checkpoint_path))
    name_='W_'+checkpoint_path
    torch.save(model,os.path.join(checkpoint_file,name_))
    print('model saved to %s / %s' % (checkpoint_file,name_))
    return model

def plot_2d(feature,sex_tlabel, ep): 
    import matplotlib.pyplot as plt
    
    dim_len = feature.shape[1]
    pic_len = int(dim_len/2)
    
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(pic_len):
        pn = '77{:.0f}'.format(p+1)
        xi = p%7
        yi = int(p/7)
        ax = plt.subplot2grid((7,7),(yi,xi))
        #ax = fig.add_subplot(int(pn))
        for tril in range(10):
            cls = sex_tlabel==tril        
            ax.scatter(feature[cls,2*p],feature[cls,2*p+1],
                       marker = '.', alpha=0.3)    
        plt.title('EDisease (cifar10) '+str(p))     
                    
    plt.savefig('./pic/'+'EDisease_2d_cifar10_'+str(ep)+'.png')
    plt.show()
    plt.close()


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encode0 = nn.Sequential(nn.Conv2d(3, 1, 1),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.Conv2d(1, 1, 1),
                                     nn.GELU(),
                                     nn.Dropout(0.5))
        self.encode1 = nn.Sequential(nn.Conv2d(1, 16, 3),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(16),
                                     nn.Conv2d(16, 32, 3),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(32),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 64, 3),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(64, 64, 3),
                                     nn.GELU(),
                                     nn.BatchNorm2d(64),
                                     nn.MaxPool2d(2),                                     
                                     nn.Conv2d(64, 64, 2),
                                     nn.Tanh()
                                     )

    def forward(self, img):
        bs = len(img)
        img = self.encode0(img)
        output = self.encode1(img)
        return output.view(bs,-1), img.view(bs,-1)
    
class GnLD(nn.Module):
    def __init__(self):
        super(GnLD, self).__init__()
        self.GLD = nn.Sequential(nn.Linear(1088, 64),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.LayerNorm(64),
                                 nn.Linear(64, 64),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.LayerNorm(64),
                                 nn.Linear(64, 32),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.LayerNorm(32),                                 
                                 nn.Linear(32, 2),
                                 )
        
    def forward(self, img, emb):
        bs = img.shape[0]
        img = img.view(bs,-1)
        
        EM = torch.cat([img,emb],dim=1)    
        output = self.GLD(EM)
                     
        return output

class PriorD(nn.Module):
    def __init__(self):
        super(PriorD, self).__init__()
        self.dense = nn.Sequential(nn.Linear(64,4*64),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(4*64),
                                   nn.Linear(4*64,2*64),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(2*64),
                                   nn.Linear(2*64,1),
                                   nn.Sigmoid()
                                   )  
        
    def forward(self, EDisease):
        output = self.dense(EDisease)
        
        return output
    
def target_real_fake(batch_size, device, soft):
    t = torch.ones(batch_size,1,device=device) 
    return soft*t, 1 - soft*t, t, 1-t
    
class DIM(nn.Module):
    def __init__(self,device='cpu',alpha=1, beta=1, gamma=10):
        super(DIM, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.GnLD = GnLD()
        self.PriorD = PriorD()
               
    def forward(self, 
                EDisease,
                M,
                soft=0.7,
                mode=None,
                ptloss=False):
        
        bs = EDisease.shape[0]
        shuffle = 1#random.randint(1,bs)
        EDiseaseFake = torch.cat([EDisease[shuffle:],EDisease[:shuffle]],dim=0)
 
        fake_domain, true_domain, fake_em, true_em = target_real_fake(batch_size=bs, device=self.device, soft=soft)
        
        criterion_DANN = nn.MSELoss().to(self.device)
        criterion_em = nn.CrossEntropyLoss().to(self.device)
        #using Transformer to similar Global + Local diversity
        
        #GLD0 = -1*F.softplus(-1*self.GnLD(EDisease, M, SEP_emb_emb, token_type_ids=None)).mean()
        #GLD1 = -1*F.softplus(-1*self.GnLD(EDiseaseFake, M, SEP_emb_emb, token_type_ids=None)).mean()
        GLD0 = self.GnLD(EDisease, M)
        GLD1 = self.GnLD(EDiseaseFake, M)
        
        GLD0_loss = criterion_em(GLD0,true_em.view(-1).long())
        GLD1_loss = criterion_em(GLD1,fake_em.view(-1).long())
                
        GLD_loss = self.alpha*(GLD0_loss+GLD1_loss)
               
        #using DANN to further train encoder        
        fake_domain+=(1.1*(1-soft)*torch.rand_like(fake_domain,device=self.device))
        true_domain-=(1.1*(1-soft)*torch.rand_like(true_domain,device=self.device))        
             
        # Proir setting
        '''
        owing to y= x ln x, convex function, a+b+c=1; a,b,c>0, <=1; when a=b=c=1/3, get the min xln x
        set prior=[-1,1] uniform
        '''
        prior = torch.rand_like(EDisease.view(bs,-1),device=self.device)
        #prior = (prior - prior.mean())/(prior.std()+1e-6)
        prior = 2*prior-1
        
        if mode=='D':
            #only train the D , not G
            for param in self.PriorD.parameters():
                param.requires_grad = True
            #d_EDisease = EDisease.view(bs,-1).detach()
            pred_domain_T = self.PriorD(EDisease.view(bs,-1).detach())
            loss_domain_T = criterion_DANN(pred_domain_T,true_domain)
            pred_domain_F = self.PriorD(prior) 
            loss_domain_F = criterion_DANN(pred_domain_F,fake_domain)          
            prior_loss = self.gamma*(loss_domain_T+loss_domain_F)
        else:
            #only train the G , not D
            for param in self.PriorD.parameters():
                param.requires_grad = False
                
            pred_domain_T = self.PriorD(EDisease.view(bs,-1))
            loss_domain_T = criterion_DANN(pred_domain_T,fake_domain)
            prior_loss = self.gamma*(loss_domain_T)
            
        if ptloss:
            with torch.no_grad():
                print('GT:{:.4f}, GF:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                              GLD1_loss.item(),
                                                                              prior_loss.item()
                                                                              ))
                print(EDisease[0,0,:24])
                print(EDisease[1,0,:24])
                print('GLD0',GLD0[:2])#,true_em,true_domain)
                print('GLD1',GLD1[:2])#,fake_em,fake_domain)
                #print('pred_domain_T',pred_domain_T)
        return GLD_loss+prior_loss


E = encoder()
D = DIM()

try: 
    E = load_checkpoint(checkpoint_file,'EDisease.pth',E)
    print(' ** Complete Load EDisease Model ** ')
except:
    print('*** No Pretrain_EDisease_Model ***')

try:     
    D = load_checkpoint(checkpoint_file,'DIM.pth',D)
except:
    print('*** No Pretrain_DIM_Model ***')
    pass


def test_AIemb(DS_model,
               dloader,
               ep):
    DS_model.eval()
    
    f_target_ = []
    all_label_ = []
    
    with torch.no_grad():         
        for batch_idx, data in enumerate(dloader):            
            sample,label = data
            imgE, _ = E(sample)   
            
            f_target_.append(imgE.cpu())
            all_label_.append(label.cpu())
            
    f_target = torch.cat(f_target_,dim=0)    
    all_label = torch.cat(all_label_,dim=0)            

    plot_2d(f_target,all_label,ep): 


def train_AIemb(DS_model,
                dim_model,
                dloader,
                lr=1e-5,
                epoch=100,
                log_interval=10,
                parallel=parallel):
    global checkpoint_file, data_loader_test
    DS_model.to(device)
    dim_model.to(device)

    model_optimizer = optim.Adam(DS_model.parameters(), lr=lr)
    model_optimizer_dim = optim.Adam(dim_model.parameters(), lr=lr)
    if parallel:
        DS_model = torch.nn.DataParallel(DS_model)
        dim_model = torch.nn.DataParallel(dim_model)
    else:
        if device == 'cuda':
            torch.cuda.set_device(0)

    iteration = 0
    total_loss = []
    
    debug = []
    debugnan = []
    LargeLossDebug =[]
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        DS_model.train()
        dim_model.train()
        
        for batch_idx, data in enumerate(dloader):
            
            sample,label = data
            sample,label = sample.to(device),label.to(device)
            
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
            ptloss = False
            
            imgE, imgM = E(sample)   
            
            bs = len(s)
              
            loss = dim_model(imgE,
                             imgM,
                             mode=mode,
                             ptloss=ptloss
                            )

            loss.backward()
            model_optimizer.step()
            model_optimizer_dim.step()

            with torch.no_grad():
                epoch_loss += loss.item()*bs
                epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.item()))
                ptloss = True
            else:
                ptloss = False
                
            if iteration % 500 == 0:
                try:  
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='EDisease.pth',
                                    model=DS_model,
                                    parallel=parallel)
                    save_checkpoint(checkpoint_file=checkpoint_file,
                                    checkpoint_path='DIM.pth',
                                    model=dim_model,
                                    parallel=parallel)
                except: 
                    print('** error save checkpoint **')
                    pass
                
            iteration +=1
        if ep % 1 ==0:
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='EDisease.pth',
                            model=DS_model,
                            parallel=parallel)
            save_checkpoint(checkpoint_file=checkpoint_file,
                            checkpoint_path='DIM.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
                 
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune.csv', sep = ',')
        
        test_AIemb(DS_model,
                   data_loader_test,
                   ep)
    print(total_loss)

