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

## Architecture
![image](https://ars.els-cdn.com/content/image/1-s2.0-S1568494620304178-gr2.jpg)

## Proeblem
> noise distribution only for case-by-case
