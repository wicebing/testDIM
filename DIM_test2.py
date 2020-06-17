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

try:
    task = sys.argv[1]
    print('*****task= ',task)
except:
    task = 'test'

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

batch_size = 8
lr = 1e-5

parallel= False
device = 'cuda'

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.RandomRotation(8),
                                transforms.RandomVerticalFlip(0.5),
                                transforms.ToTensor()])

transform_test = transforms.Compose([transforms.Grayscale(),
                                     transforms.ToTensor()])

train_data=MyDataset(ds=CIFAR10, transform=transform)
data_loader_train = DataLoader(train_data, batch_size=batch_size,shuffle=True)

test_data=MyDataset(ds=CIFAR10_test, transform=transform_test)
data_loader_test = DataLoader(test_data, batch_size=batch_size,shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  

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

def plot_2d(feature,sex_tlabel, ep,picpath='./pic/'): 
    print('** draw 2d **')
    import matplotlib.pyplot as plt
    
    dim_len = feature.shape[1]
    pic_len = int(dim_len/2)
    
    fig = plt.figure(figsize=(48,48),dpi=100)
    
    for p in range(pic_len):
        pn = '66{:.0f}'.format(p+1)
        xi = p%6
        yi = int(p/6)
        ax = plt.subplot2grid((6,6),(yi,xi))
        #ax = fig.add_subplot(int(pn))
        for tril in range(10):
            cls = sex_tlabel==tril        
            ax.scatter(feature[cls,2*p],feature[cls,2*p+1],
                       marker = '.', alpha=0.3,label=str(tril))    
        plt.title('EDisease (cifar10) '+str(p))     
        plt.legend()
    plt.savefig(picpath+'EDisease_2d_cifar10_'+str(ep)+'.png')
    # plt.show()
    plt.close()

def plot_tsen(feature,sex_tlabel, ep,picpath='./pic_tsne/'): 
    print('** draw 2d **')
    import matplotlib.pyplot as plt
    from sklearn import manifold

    #draw tsne
    print(' ** tsne ** ')
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    T_tsne = tsne.fit_transform(feature.numpy())
    
    dim_len = feature.shape[1]
    pic_len = int(dim_len/2)
    
    fig = plt.figure(figsize=(10,10),dpi=100)
    
    ax = fig.add_subplot(111)
    #ax = fig.add_subplot(int(pn))
    for tril in range(10):
        cls = sex_tlabel==tril        
        ax.scatter(T_tsne[cls,0],T_tsne[cls,1],
                   marker = '.', alpha=0.3,label=str(tril))    
    plt.title('EDisease (cifar10) tsne '+str(ep))     
    plt.legend()
    plt.savefig(picpath+'EDisease_2d_cifar10_tsne_'+str(ep)+'.png')
    # plt.show()
    plt.close()


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.encode0 = nn.Sequential(nn.Conv2d(1, 1, 1),
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
        return output.view(bs,-1), img#.view(bs,-1)

class decoder(nn.Module):
    def __init__(self,device='cpu'):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ConvTranspose2d(64,64,kernel_size=3, stride=1, padding=0),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(64),
                                     nn.ConvTranspose2d(64,64,kernel_size=3, stride=2, padding=0),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(64),
                                     nn.ConvTranspose2d(64,32,kernel_size=3, stride=2, padding=0),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(32),
                                     nn.ConvTranspose2d(32,16,kernel_size=3, stride=2, padding=0),
                                     nn.GELU(),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(16),                                    
                                     nn.ConvTranspose2d(16,1,kernel_size=4, stride=1, padding=1),
                                     nn.Sigmoid()
                                     )
        self.device = device

    def forward(self, emb, ori):
        bs = len(emb)
        emb = emb.view(bs,64,1,1)
        img = self.decoder(emb)
        
        criterion_DANN = nn.MSELoss().to(self.device)
        
        loss = criterion_DANN(img,ori)
        
        return loss, img
    
class GnLD(nn.Module):
    def __init__(self):
        super(GnLD, self).__init__()
        self.GLD = nn.Sequential(nn.Conv2d(65, 128, 3),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.BatchNorm2d(128),
                                 nn.Conv2d(128, 64, 3),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.BatchNorm2d(64),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(64, 32, 3),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.BatchNorm2d(32),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32, 16, 3),
                                 nn.GELU(),
                                 nn.Dropout(0.5),
                                 nn.BatchNorm2d(16),
                                 nn.Conv2d(16, 2, 4),                                 
                                 )

        
    def forward(self, emb, img):
        bs = img.shape[0]
        
        emb = emb.unsqueeze(2).unsqueeze(2)
        emb = emb.expand([*emb.shape[:2],32,32])
        
        EM = torch.cat([img,emb],dim=1)    
        output = self.GLD(EM)
                            
        return output.view(bs,-1)

class GloD(nn.Module):
    def __init__(self):
        super(GloD, self).__init__()
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
        self.Global =nn.Sequential(nn.Linear(64*2,4*64),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(4*64),
                                   nn.Linear(4*64,2*64),
                                   nn.LeakyReLU(),
                                   nn.Dropout(0.5),
                                   nn.BatchNorm1d(2*64),
                                   nn.Linear(2*64,2),                              
                                   )

        
    def forward(self, emb, img):
        bs = img.shape[0]        
        ME = self.encode1(img)
                
        EM = torch.cat([emb,ME.view(bs,-1)],dim=1)    
        output = self.Global(EM)
                            
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
    def __init__(self,device='cuda',alpha=1, beta=1, gamma=1):
        super(DIM, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.GnLD = GnLD()
        self.PriorD = PriorD()
        self.GLobal = GloD()
               
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
        
        #global loss
        Global0 = self.GLobal(EDisease, M)
        Global1 = self.GLobal(EDiseaseFake, M)
        
        Global0_loss = criterion_em(Global0,true_em.view(-1).long())
        Global1_loss = criterion_em(Global1,fake_em.view(-1).long())
                
        Global_loss = self.beta*(Global0_loss+Global1_loss)

               
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
                print('GT:{:.4f}, GF:{:.4f}, GloT:{:.4f}, GloF:{:.4f}, Prior{:.4f}'.format(GLD0_loss.item(),
                                                                              GLD1_loss.item(),
                                                                              Global0_loss.item(),
                                                                              Global1_loss.item(),
                                                                              prior_loss.item()
                                                                              ))
                print(EDisease[0])
                print(EDisease[1])
                print('GLD0',GLD0)#,true_em,true_domain)
                print('GLD1',GLD1)#,fake_em,fake_domain)
                print('Global0',Global0)#,true_em,true_domain)
                print('Global1',Global1)#,fake_em,fake_domain)
                #print('pred_domain_T',pred_domain_T)
        return GLD_loss+prior_loss+Global_loss

def test_AIemb(DS_model,
               dloader,
               ep,
               picpath,
               tsnepath):
    global device
    DS_model.eval()
    
    f_target_ = []
    all_label_ = []
    
    with torch.no_grad():         
        for batch_idx, data in enumerate(dloader):            
            sample,label = data
            sample,label = sample.to(device),label.to(device)
            imgE, _ = E(sample)   
            
            f_target_.append(imgE.cpu())
            all_label_.append(label.cpu())
            
    f_target = torch.cat(f_target_,dim=0)    
    all_label = torch.cat(all_label_,dim=0)            

    plot_2d(f_target,all_label,ep,picpath)
    if ep%5==0:
        plot_tsen(f_target,all_label,ep,tsnepath)
    return f_target,all_label


def train_AIemb(DS_model,
                dim_model,
                dloader,
                lr=lr,
                epoch=10000,
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
    
    # f_target,all_label = test_AIemb(DS_model,
    #            data_loader_test,
    #            'init')
    
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        DS_model.train()
        dim_model.train()
        ptloss = False
        
        for batch_idx, data in enumerate(dloader):
            
            sample,label = data
            sample,label = sample.to(device),label.to(device)
            
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
            
            
            imgE, imgM = E(sample)   
            
            bs = len(sample)
              
            loss = dim_model(imgE,
                             imgM,
                             mode=mode,
                             ptloss=ptloss
                            )

            loss.sum().backward()
            model_optimizer.step()
            model_optimizer_dim.step()

            with torch.no_grad():
                epoch_loss += loss.sum().item()*bs
                epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.sum().item()))
            if iteration % 1000 == 1:
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
        
        f_target,all_label = test_AIemb(DS_model,
                   data_loader_test,
                   ep+375,
                   picpath='./pic/',
                   tsnepath='./tsne_pic/')
    print(total_loss)


def train_AIAED(DS_model,
                dim_model,
                dloader,
                lr=lr,
                epoch=10000,
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
    
    # f_target,all_label = test_AIemb(DS_model,
    #            data_loader_test,
    #            'init',
    #            picpath='./pic_AE/')
    
    for ep in range(epoch):   
        t0 = time.time()
        epoch_loss = 0
        epoch_cases =0
        DS_model.train()
        dim_model.train()
        ptloss = False
        
        for batch_idx, data in enumerate(dloader):
            
            sample,label = data
            sample,label = sample.to(device),label.to(device)
            
            model_optimizer.zero_grad()
            model_optimizer_dim.zero_grad()
            loss = 0

            mode = 'D' if batch_idx%2==0 else 'G'
            
            
            imgE, imgM = E(sample)   
            
            bs = len(sample)
              
            loss,img = dim_model(imgE,
                                 sample,
                                 )

            loss.sum().backward()
            model_optimizer.step()
            model_optimizer_dim.step()

            with torch.no_grad():
                epoch_loss += loss.sum().item()*bs
                epoch_cases += bs

            if iteration % log_interval == 0:
                print('Ep:{} [{} ({:.0f}%)/ ep_time:{:.0f}min] L:{:.4f}'.format(
                        ep, batch_idx * batch_size,
                        100. * batch_idx / len(dloader),
                        (time.time()-t0)*len(dloader)/(60*(batch_idx+1)),
                        loss.sum().item()))
            if iteration % 1000 == 1:
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
                                    checkpoint_path='decoder.pth',
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
                            checkpoint_path='decoder.pth',
                            model=dim_model,
                            parallel=parallel)

            print('======= epoch:%i ========'%ep)
                 
        print('++ Ep Time: {:.1f} Secs ++'.format(time.time()-t0)) 
        total_loss.append(float(epoch_loss/epoch_cases))
        pd_total_loss = pd.DataFrame(total_loss)
        pd_total_loss.to_csv('./loss_record/total_loss_finetune_ae.csv', sep = ',')
        
        f_target,all_label = test_AIemb(DS_model,
                   data_loader_test,
                   ep+170,
                   picpath='./pic_AE/',
                   tsnepath='./tsne_pic_AE/')
    print(total_loss)
    
if task == 'dim':    
    device = 'cuda:1'
    checkpoint_file = './checkpoint' 

    E = encoder()
    D = DIM(device=device,gamma=1)
    
    dec = decoder()
    
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
    
    try:     
        dec = load_checkpoint(checkpoint_file,'decoder.pth',dec)
    except:
        print('*** No Pretrain_decoder_Model ***')
        pass

    train_AIemb(E,
                D, 
                data_loader_train,
                lr=1e-5, 
                epoch=10000,
                log_interval=10,
                parallel=parallel)
    
elif task == 'ae': 
    device = 'cuda'
    checkpoint_file = './checkpoint_AE' 
    
    E = encoder()
    D = DIM(device=device,gamma=1)
    
    dec = decoder()
    
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
    
    try:     
        dec = load_checkpoint(checkpoint_file,'decoder.pth',dec)
    except:
        print('*** No Pretrain_decoder_Model ***')
        pass

    train_AIAED(E,
                dec, 
                data_loader_train,
                lr=1e-5, 
                epoch=10000,
                log_interval=10,
                parallel=parallel)

