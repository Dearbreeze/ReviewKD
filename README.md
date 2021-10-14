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
The pre-trained model will be released soon! 

## Acknowledgement 

