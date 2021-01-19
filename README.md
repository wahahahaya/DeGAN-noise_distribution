# DeGAN_noise_distribuyion

## Reference
Qiongshuai Lyu, Min Guo, Zhao Pei,
DeGAN: Mixed noise removal via generative adversarial networks,
Applied Soft Computing,
Volume 95,
2020,
106478,
ISSN 1568-4946,

## Data
- three types of noise
  - White Gaussian noise(AWGN)
  - Salt-and-Paper noise(SPIN)
  - random-valued noise(RVIN)
  
- Training Data
  - AWGW+SPIN 
  - > <img src="http://chart.googleapis.com/chart?cht=tx&chl= \sigma=20\;s=20%,40%\\ \sigma=50\;s=20%,40%\\ " style="border:none;">
  - AWGW+RVIN
  - > <img src="http://chart.googleapis.com/chart?cht=tx&chl= \sigma=10\;r=10%,30%\\ \sigma=40\;r=10%,30%\\ " style="border:none;">
  - AWGW+SPIN+RVIN
  - > <img src="http://chart.googleapis.com/chart?cht=tx&chl= \sigma=40\;s=5%\;r=5%\\ \sigma=30\;s=10%\;r=10%\\ \sigma=20\;s=15%\;r=15%\\ \sigma=10\;s=20%\;r=20%\\ " style="border:none;">
  
- Dataset in this paper
  - Joseph Chet Redmon, Pascal VOC Dataset Mirror (VOC2007).
  > https://pjreddie.com/projects/pascal-voc-dataset-mirror/
  - K. Zhang, Datasets, 2016.
  > https://drive.google.com/drive/u/0/folders/0B-_yeZDtQSnobXIzeHV5SjY5NzA
  
- Our noise dataset
  - https://drive.google.com/file/d/1xfFTD_gd_u-LqqYI3qkblhZGrSBnKCVV/view
  - https://drive.google.com/file/d/1OUqzojlkCKBVhheZjd3KslY-d9ZEY4fg/view

## Architecture
![image](https://ars.els-cdn.com/content/image/1-s2.0-S1568494620304178-gr2.jpg)

## Requirement
- numpy==1.19.2
- Pillow==8.1.0
- torch==1.7.1+cu110
- torchaudio==0.7.2
- torchvision==0.8.2+cu110
- typing-extensions==3.7.4.3


```shell
pip install -r requirements.txt 
```
## Our result
PSNR = 69.36

SSIM = 0.746

> noise image
> denoise image
> ground truth
![image](https://github.com/wahahahaya/DeGAN-noise_distribuyion/blob/main/result/48,PSNE=69.3609,SSIM=0.7463.png?raw=true)

## Proeblem
> noise distribution only for case-by-case
