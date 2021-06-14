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

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if device == "cuda" else torch.Tensor
    gen = GenUnet().to(device)
    gen.load_state_dict(torch.load("saved_model/Gen_8.pth"))

    data_process_steps = [
            transforms.Grayscale(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]

    test_dir="./data"
    test_data = DataLoader(
            img_data(
                test_dir,
                transforms_=data_process_steps,
                mode="test",
            ),
            batch_size=5,
            #shuffle=True,
            num_workers=4,
        )

    def PSNR(img1, img2):
        MSE = torch.nn.MSELoss()
        psnr = 10*math.log10((255**2)/MSE(img1,img2).item())
        return psnr

    import loss
    SSIM = loss.SSIM()

    for i, batch in enumerate(test_data):
        gen.eval()

        noise = Variable(batch["noise"].type(Tensor))
        dis_noise = gen(noise)
        ground = Variable(batch["groundtruth"].type(Tensor))

        psnr = PSNR(dis_noise,ground)
        ssim = SSIM(dis_noise,ground).item()

        img_noise = make_grid(noise, nrow=5, normalize=True)
        img_dis = make_grid(dis_noise, nrow=5, normalize=True)
        img_ground = make_grid(ground, nrow=5, normalize=True)

        image_grid = torch.cat((img_noise, img_dis, img_ground), 1)
        save_image(image_grid, "test_image/%s,PSNE=%.4f,SSIM=%.4f.png" %(i,psnr,ssim), normalize=False)

if __name__ == "__main__":
    test()