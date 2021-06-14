import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model import *
from data import *
import loss


import os
import numpy as np
import math
import itertools
import time


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.Tensor

    # parameters
    epochs = 1
    sample_interval = 100
    last_epoch = 9

    # image transformations
    data_process_steps = [
        transforms.Grayscale(),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ]

    # dataloader
    train_dir="./data"
    test_dir="./data"

    train_data = DataLoader(
        img_data(
            train_dir,
            transforms_=data_process_steps,
        ),
        batch_size=1,
        #shuffle=True,
        num_workers=4,
    )

    # Build model
    gen = GenUnet().to(device)
    dis = Dis().to(device)
    vgg = VGG19_fea().to(device)

    # Loss Function
    adv_loss = torch.nn.BCELoss()
    pix_loss = torch.nn.MSELoss()
    fea_loss = torch.nn.MSELoss()
    ssim_loss = loss.SSIM()

    # optimizer
    opt_g = torch.optim.Adam(
        gen.parameters(),
        lr=0.0002,
        betas=(0.01, 0.999),
    )
    opt_d = torch.optim.Adam(
        dis.parameters(),
        lr=0.0002,
        betas=(0.01, 0.999),
    )
    def PSNR(img1, img2):
        MSE = torch.nn.MSELoss()
        psnr = 10*math.log10((255**2)/MSE(img1,img2).item())
        return psnr

    def sample_images(batches_done):
        imgs = next(iter(train_data))
        gen.eval()

        noise = Variable(imgs["noise"].type(Tensor))
        dis_noise = gen(noise)
        ground = Variable(imgs["groundtruth"].type(Tensor))

        ssim = loss.SSIM()
        ssim_loss = ssim(dis_noise,ground).item()
        psnr_loss = PSNR(dis_noise,ground)

        img_noise = make_grid(noise, nrow=5, normalize=True)
        img_dis = make_grid(dis_noise, nrow=5, normalize=True)
        img_ground = make_grid(ground, nrow=5, normalize=True)

        image_grid = torch.cat((img_noise, img_dis, img_ground), 1)
        save_image(image_grid, "train_image/%s,SSIM=%.3f,PSNR=%.3f.png" % (batches_done, ssim_loss, psnr_loss), normalize=False)


    gen.load_state_dict(torch.load("saved_model/Gen_%d.pth" % (last_epoch-1)))
    dis.load_state_dict(torch.load("saved_model/Dis_%d.pth" % (last_epoch-1)))
    opt_g.load_state_dict(torch.load("saved_model/opt_g_%d.pth" % (last_epoch-1)))
    opt_d.load_state_dict(torch.load("saved_model/opt_d_%d.pth" % (last_epoch-1)))

    gg=[]
    for epoch in range(last_epoch,epochs+last_epoch):

        g=[]
        for i, batch in enumerate(train_data):
            groundtruth = Variable(batch["groundtruth"].type(Tensor))
            noise = Variable(batch["noise"].type(Tensor))

            # Adversarial ground truths
            vaild = Variable(Tensor(np.ones((groundtruth.size(0), *dis.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((noise.size(0), *dis.output_shape))), requires_grad=False)

            #
            # Train Generator
            #

            opt_g.zero_grad()

            dis_noise = gen(noise)
            g_loss = adv_loss(dis(dis_noise), vaild)

            ssim = ssim_loss(dis_noise, groundtruth)
            ssim_ = abs(ssim-1)
            pix = pix_loss(dis_noise, groundtruth)
            img_loss = ssim_ + pix

            dis_fea = vgg(dis_noise)
            ground_fea = vgg(groundtruth)
            fea = fea_loss(dis_fea, ground_fea)

            total_loss = g_loss + fea + 0.5*pix + 1.5*ssim_
            g.append(total_loss.item())
            total_loss.backward()

            opt_g.step()

            #
            # Train Discriminator
            #

            opt_d.zero_grad()

            real_loss = adv_loss(dis(groundtruth), vaild)
            fake_loss = adv_loss(dis(dis_noise.detach()), fake)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            opt_d.step()

            #
            # Log progress
            #
            batches_done = (epoch) * len(train_data) + i
            print("Epoch: {}/{}, Batch: {}/{}, D loss: {:.4f}, G loss: {:.4f}, img loss: {:.4f}, feature loss: {:.4f}, ssim: {:.4f}, pix: {:.4f}, total G: {:.4f}".format(epoch,last_epoch+epochs-1,i,len(train_data),d_loss.item(),g_loss.item(),img_loss.item(),fea.item(),ssim.item(),pix.item(),total_loss.item()))

            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)

        gg.append(sum(g) / len(g))
        print(gg)

        torch.save(gen.state_dict(),"saved_model/Gen_%d.pth" % (epoch))
        torch.save(dis.state_dict(),"saved_model/Dis_%d.pth" % (epoch))
        torch.save(opt_g.state_dict(),"saved_model/opt_g_%d.pth" % (epoch))
        torch.save(opt_d.state_dict(), "saved_model/opt_d_%d.pth" % (epoch))

    with open("loss.txt", "w") as f:
        for i in gg:
            f.write("%f " % i)


if __name__ == "__main__":
    train()