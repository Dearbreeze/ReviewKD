# ReviewKD
## ReviewKD: Review Boosted Knowledge Distillation for Crowd Counting with Performance  Beyond Teacher

## Prerequisites
We recommend Anaconda as the environment

* Linux Platform
* NVIDIA GPU + CUDA CuDNN
* Torch == 1.8.0
* torchvision == 0.9.0
* Python3.8.0
* numpy1.19.2
* opencv-python
* visdom

## Training
1. Modify --train_json and --test_json in ReviewKD_train.py
2. Preper a pre-trained of Teacher network and modify --teacher_ckpt to your loacl path
3. Begining training:
 ```
$ python ReviewKD_train.py
 ```

## Test
### The well-trained teacher networks Weight
#### UCF-QNRF
##### CSRNet
Baidu Yun [Link](https://pan.baidu.com/s/1WttH01bw2sqG5YFXK2un9Q) (extract code: fdji)
##### BL
Baidu Yun [Link](https://pan.baidu.com/s/1DNcV0FGNd0YU0e-HJlzc2w) (extract code: 0uqw)

#### ShanghaiTech A
##### CSRNet
Baidu Yun [Link](https://pan.baidu.com/s/1C9Xfe35X0uIFPYjY9ItykQ) (extract code: 9lrk)
##### BL
Baidu Yun [Link](https://pan.baidu.com/s/1r-U4zNcqfH9uYSbBnBpsrA) (extract code: pmos)


#### ShanghaiTech B
##### CSRNet
Baidu Yun [Link](https://pan.baidu.com/s/1Znjeh4AybCE2FlrQodmkPA) (extract code: tip1)
##### BL
Baidu Yun [Link](https://pan.baidu.com/s/1a2l1xyXJ3ZmjUIaadL1gfw) (extract code: h04q)

The pre-trained model will be released soon! 

## Datasets

ShanghaiTech_PartA: [Link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fa3h2T1ZKQlZ1eFBzdTc1WWZZaHY5VUJLUkZOUDdXZ0xkeFhGTVNlSEdoWGpRP3J0aW1lPVM1bHlyaGVQMlVn&id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FC3Data%2Fshanghaitech%5Fpart%5FA%2Ezip&parent=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FC3Data)

ShanghaiTech_PartB: [Link](https://mailnwpueducn-my.sharepoint.com/personal/gjy3035_mail_nwpu_edu_cn/_layouts/15/onedrive.aspx?originalPath=aHR0cHM6Ly9tYWlsbndwdWVkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2dqeTMwMzVfbWFpbF9ud3B1X2VkdV9jbi9Fa3h2T1ZKQlZ1eFBzdTc1WWZZaHY5VUJLUkZOUDdXZ0xkeFhGTVNlSEdoWGpRP3J0aW1lPVM1bHlyaGVQMlVn&id=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FC3Data%2Fshanghaitech%5Fpart%5FB%2Ezip&parent=%2Fpersonal%2Fgjy3035%5Fmail%5Fnwpu%5Fedu%5Fcn%2FDocuments%2F%E8%AE%BA%E6%96%87%E5%BC%80%E6%BA%90%E6%95%B0%E6%8D%AE%2FC3Data)

UCF-QNRF: [Link](https://www.crcv.ucf.edu/data/ucf-qnrf/)
## Acknowledgement 

