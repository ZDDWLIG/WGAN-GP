import os

import torch as th
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

# def plot_fake_imgs(generator):


# def plot_loss(gener_loss_list,discr_loss_list):
#     pass
#Generator model: [b,noise_size]->[b,C,H,W]

class Generator(nn.Module):
    def __init__(self,noise_size):
        super().__init__()
        # self.mlp=nn.Sequential(nn.Linear(noise_size,512),nn.ReLU(),nn.Linear(512,1024),nn.ReLU(),
        #                        nn.Linear(1024,3*32*32),nn.Tanh())
        self.mlp = nn.Sequential(nn.Linear(noise_size, 512), nn.ReLU(), nn.Linear(512, 1024), nn.ReLU(),
                                 nn.Linear(1024, 1 * 28 * 28), nn.Tanh())
    def forward(self,x):
        x=self.mlp(x)
        # x=x.view(-1,3,32,32)
        x = x.view(-1, 1, 28, 28)
        return x # output batch_size images

#Discriminator [b,C,H,W]->[b]
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.mlp=nn.Sequential(nn.Linear(3*32*32,1024),nn.ReLU(),nn.Linear(1024,256),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())
        self.mlp=nn.Sequential(nn.Linear(1*28*28,1024),nn.ReLU(),nn.Linear(1024,256),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,x):
        # x=x.view(-1,3*32*32)
        x = x.view(-1, 1 * 28 * 28)
        x=self.mlp(x)
        return x #output batch_size scores

class GAN():
    def __init__(self,noise_size,batch_size,lr,epoch_num,device,eval_gap=40):
        super().__init__()
        self.batch_size=batch_size
        self.noise_size=noise_size
        self.lr=lr
        self.epoch_num=epoch_num
        self.device=device
        self.eval_gap=eval_gap
        self.model_path=os.getcwd()+'/results/checkpoints/'
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.generator=Generator(self.noise_size).to(self.device)
        self.discriminator=Discriminator().to(self.device)
        self.loss=nn.BCELoss()
        self.generator_opt=optim.Adam(self.generator.parameters(),self.lr)
        self.discriminator_opt=optim.Adam(self.discriminator.parameters(),self.lr)

    def train(self,train_dataloader):
        gener_loss_list=[]
        discr_loss_list=[]
        for ep in range(self.epoch_num):
            gener_avg_loss=0.
            discr_avg_loss=0.
            for batch_id,(real_img,_) in enumerate(train_dataloader):
                real_img=real_img.to(self.device)
                noise=th.randn(real_img.size(0),self.noise_size).to(self.device)
                fake_img=self.generator(noise)

                #updata discriminator network, discriminator loss=real image loss +fake image loss
                real_img_scores=self.discriminator(real_img)
                discr_real_loss=self.loss(real_img_scores,th.ones_like(real_img_scores))
                fake_img_scores=self.discriminator(fake_img.detach())
                discr_fake_loss=self.loss(fake_img_scores,th.zeros_like(fake_img_scores))
                self.discriminator_opt.zero_grad()
                discr_fake_loss.backward()
                discr_real_loss.backward()
                self.discriminator_opt.step()
                discr_avg_loss+=(discr_fake_loss.item()+discr_real_loss.item())/real_img.size(0)
                #updata generator network, generator loss derives from discriminator
                fake_img_scores=self.discriminator(fake_img)
                gener_loss=self.loss(fake_img_scores,th.ones_like(fake_img_scores))
                self.generator_opt.zero_grad()
                gener_loss.backward()
                self.generator_opt.step()
                gener_avg_loss+=gener_loss.item()/real_img.size(0)
            gener_loss_list.append(gener_avg_loss)
            discr_loss_list.append(discr_avg_loss)
            print('epoch:{} generator_loss={} discriminator_loss={}'.format(ep,gener_avg_loss,discr_avg_loss))
            if ep % self.eval_gap == 0:
                self.plot_img(self.generator)
                self.plot_loss(gener_loss_list,discr_loss_list)
                self.save(ep)
    def save(self,ep):

        th.save(self.generator.state_dict(),self.model_path+f'generator_{ep}.pth')
        th.save(self.discriminator.state_dict(),self.model_path+f'discriminator_{ep}.pth')
    def load(self,generator_path,discriminator_path):
        self.generator.load_state_dict(th.load(generator_path))
        self.discriminator.load_state_dict(th.load(discriminator_path))

    def plot_img(self,generator):
        noise = th.randn(16, self.noise_size).to(self.device)
        # fake_img=np.transpose(np.squeeze(generator(noise).detach().cpu().numpy()),axes=[0,2,3,1])#C H W-> W H C
        fake_img=np.squeeze(generator(noise).detach().cpu().numpy())
        fig=plt.figure(figsize=(4,4))
        for i in range(16):
            plt.subplot(4,4,i+1)
            plt.imshow((fake_img[i]+1)/2)
            plt.axis('off')
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    def plot_loss(self,gener_loss_list,discr_loss_list):
        fig,axs=plt.subplots(1,2,figsize=(8,6))
        axs[0].plot([x for x in range(len(gener_loss_list))],gener_loss_list,label='generator loss')
        axs[1].plot([x for x in range(len(discr_loss_list))],discr_loss_list,label='discriminator loss')
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def test(model,testloader,test_epoch_num,batch_size,device,noise_size=300):
    gener_loss_list=[]
    discr_loss_list=[]
    for ep in range(test_epoch_num):
        gener_loss=0.
        discr_loss=0.
        for batch_id,(real_img,_) in enumerate(testloader):
            noise=th.randn(batch_size,noise_size).to(device)
            real_img=real_img.to(device)
            fake_img=model.generator(noise)
            fake_img_scores=model.discriminator(fake_img)
            real_img_scores=model.discriminator(real_img)
            discr_loss+=(model.loss(fake_img_scores,th.zeros_like(fake_img_scores)).item()+
                         model.loss(real_img_scores,th.ones_like(real_img_scores)).item())
            gener_loss+=model.loss(fake_img_scores,th.ones_like(fake_img_scores)).item()
        gener_loss_list.append(gener_loss)
        discr_loss_list.append(discr_loss)
        model.plot_img(model.generator)
    model.plot_loss(gener_loss_list,discr_loss_list)









