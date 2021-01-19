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
  <img src="http://chart.googleapis.com/chart?cht=tx&chl= \sigma=10\;r=10%,30%\\ \sigma=40\;r=10%,30% " style="border:none;">
  - AWGW+RVIN
  - AWGW+SPIN+RVIN
  
- Dataset in this paper
  - Joseph Chet Redmon, Pascal VOC Dataset Mirror (VOC2007).
  > https://pjreddie.com/projects/pascal-voc-dataset-mirror/
  - K. Zhang, Datasets, 2016.
  > https://drive.google.com/drive/u/0/folders/0B-_yeZDtQSnobXIzeHV5SjY5NzA
<img src="http://chart.googleapis.com/chart?cht=tx&chl= \sigma" style="border:none;">

## Architecture
![image](https://ars.els-cdn.com/content/image/1-s2.0-S1568494620304178-gr2.jpg)

## Proeblem
> noise distribution only for case-by-case
