"""
ReviewKD
"""
import sys
import os

import warnings

from models.model_teacher_vgg import CSRNet as CSRNet_teacher
from models.student_model_mobileNet import MobileNetV2 as CSRNet_student
import torchnet as tnt
from utils import save_checkpoint, cal_para
from models.distillation import cosine_similarity, scale_process, cal_dense_fsp
import Visualizer
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import glob
import numpy as np
import argparse
import json
import dataset
import time

parser = argparse.ArgumentParser(description='CSRNet-ReviewKD train code')
parser.add_argument('--train_json', metavar='TRAIN',default='/media/lyx/Ain/data/Shanghai/part_A_final',
                    help='path to train json')
# parser.add_argument('val_json', metavar='VAL',
#                     help='path to val json')
parser.add_argument('--test_json', metavar='TEST',default='/media/lyx/Ain/data/Shanghai/part_A_final',
                    help='path to test json')
parser.add_argument('--lr', default=1e-4, type=float,
                    help='learning rate')
# parser.add_argument('--teacher', '-t', default=None, type=str,
#                     help='teacher net version')
parser.add_argument('--teacher_ckpt', '-tc', default='./models/partA_teacher.pth.tar', type=str,
                    help='teacher checkpoint')
# parser.add_argument('--student', '-s', default=None, type=str,
#                     help='student net version')
parser.add_argument('--student_ckpt', '-sc', default=None, type=str,
                    help='student checkpoint')
parser.add_argument('--lamb_fsp', '-laf', type=float, default=0.5,
                    help='weight of dense fsp loss')
parser.add_argument('--lamb_cos', '-lac', type=float, default=0.5,
                    help='weight of cos loss')
parser.add_argument('--gpu', metavar='GPU', type=str, default='0',
                    help='GPU id to use')
parser.add_argument('--out', metavar='OUTPUT', type=str,default='./model',
                    help='path to output')
parser.add_argument('--name', default='ReviewKD_partA_mobilenet', type=str, help='name to visdom')


global args
args = parser.parse_args()


def main():
    global args, mae_best_prec1, mse_best_prec1

    mae_best_prec1 = 1e6
    mse_best_prec1 = 1e6

    args.batch_size = 1  # args.batch
    args.momentum = 0.95
    args.decay = 5 * 1e-4
    args.start_epoch = 0
    args.epochs = 1000
    args.workers = 2
    args.seed = time.time()
    args.print_freq = 400
    vis = Visualizer.Visualizer(args.name)
    train_list = glob.glob(os.path.join(args.train_json,'train_data','images' ,'*.jpg'))
    test_list = glob.glob(os.path.join(args.test_json, 'test_data','images' ,'*.jpg'))
    # print(len(train_list))
    # print(len(test_list))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    teacher = CSRNet_teacher()
    student = CSRNet_student()
    cal_para(student)  # include 1x1 conv transform parameters

    teacher.regist_hook()  # use hook to get teacher's features
    teacher = teacher.cuda()
    student = student.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.Adam(student.parameters(), args.lr, weight_decay=args.decay)
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.5) # TODO 学习率衰减

    if os.path.isdir(args.out) is False:
        os.makedirs(args.out.decode('utf-8'))

    if args.teacher_ckpt:
        if os.path.isfile(args.teacher_ckpt):
            print("=> loading checkpoint '{}'".format(args.teacher_ckpt))
            checkpoint = torch.load(args.teacher_ckpt)
            teacher.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.teacher_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.teacher_ckpt))

    if args.student_ckpt:
        if os.path.isfile(args.student_ckpt):
            print("=> loading checkpoint '{}'".format(args.student_ckpt))
            checkpoint = torch.load(args.student_ckpt)
            args.start_epoch = checkpoint['epoch']
            if 'best_prec1' in checkpoint.keys():
                mae_best_prec1 = checkpoint['best_prec1']
            else:
                mae_best_prec1 = checkpoint['mae_best_prec1']
            if 'mse_best_prec1' in checkpoint.keys():
                mse_best_prec1 = checkpoint['mse_best_prec1']
            student.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.student_ckpt, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.student_ckpt))


    for epoch in range(args.start_epoch, args.epochs):

        train(train_list, teacher, student, criterion, optimizer, epoch ,vis)
        mae_prec1, mse_prec1 = test(test_list, student, vis)

        mae_is_best = mae_prec1 < mae_best_prec1
        mae_best_prec1 = min(mae_prec1, mae_best_prec1)
        mse_is_best = mse_prec1 < mse_best_prec1
        mse_best_prec1 = min(mse_prec1, mse_best_prec1)
        print('Best val * MAE {mae:.3f} * MSE {mse:.3f}'
              .format(mae=mae_best_prec1, mse=mse_best_prec1))
        file = open('re_txt/'+args.name+'.txt', 'a+')
        file.write(str([epoch, 'best:', mae_best_prec1]))
        file.write('\n')
        file.close()
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.student_ckpt,
            'state_dict': student.state_dict(),
            'mae_best_prec1': mae_best_prec1,
            'mse_best_prec1': mse_best_prec1,
            'optimizer': optimizer.state_dict(),
        }, mae_is_best, mse_is_best, args.out ,args.name)
        # scheduler.step() # TODO 学习率衰减
        vis.save([args.name])

def train(train_list, teacher, student, criterion, optimizer, epoch ,vis ):
    losses_h = AverageMeter()
    # losses_s = AverageMeter()
    losses_fsp = AverageMeter()
    losses_cos = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss_meter.reset()
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]),
                            train=True,
                            seen=student.seen,
                            ),
        num_workers=args.workers,
        shuffle=True,
        batch_size=args.batch_size)
    print('epoch %d, lr %.10f %s' % (epoch, args.lr, args.out))
    teacher.eval()
    student.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        target = target.type(torch.FloatTensor).cuda()
        target = Variable(target)

        with torch.no_grad():
            teacher_output = teacher(img)
            # teacher.features.append(teacher_output)
            # for i in teacher.features:
            #     print(i.size())
            # print('===========================')
            teacher_fsp_features = [scale_process(teacher.features)]

            teacher_fsp = cal_dense_fsp(teacher_fsp_features)

        student_features = student(img)
        # for i in student_features:
        #     print(i.size())
        map1 = student_features[-3]
        map2 = student_features[-2]
        map3 = student_features[-1]
        student_output = map3
        student_features = student_features[:-3]
        student_fsp_features = [scale_process(student_features)]
        student_fsp = cal_dense_fsp(student_fsp_features)

        if i % 30 == 0:
            vis.img('images', (img.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
            vis.heatmap(target.data[0].squeeze(0), opts=dict(colormap='Jet'), win='heat_map_true')
            vis.heatmap(teacher_output.data[0].squeeze(0), opts=dict(colormap='Jet'), win='teacher_heat_map')
            vis.heatmap(student_output.data[0].squeeze(0), opts=dict(colormap='Jet'), win='student_heat_map')

        loss_h1 = criterion(map1, target)
        loss_h2 = criterion(map2, target)
        loss_h3 = criterion(map3, target)
        loss_h = loss_h1 + loss_h2 + loss_h3
        # loss_s = criterion(student_output, teacher_output)

        loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()
        if args.lamb_fsp:
            loss_f = []
            assert len(teacher_fsp) == len(student_fsp)
            for t in range(len(teacher_fsp)):
                loss_f.append(criterion(teacher_fsp[t], student_fsp[t]))
            loss_fsp = sum(loss_f) * args.lamb_fsp

        loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
        if args.lamb_cos:
            loss_c = []
            for t in range(len(student_features)):
                loss_c.append(cosine_similarity(student_features[t], teacher.features[t]))
            loss_cos = sum(loss_c) * args.lamb_cos

        loss = loss_h + loss_fsp + loss_cos
        # loss_meter.add(float(loss.data))
        # # am_meter.add(float(loss_att.data))
        # vis.plot('total_loss', loss_meter.value()[0])

        losses_h.update(loss_h.item(), img.size(0))
        # losses_s.update(loss_s.item(), img.size(0))
        losses_fsp.update(loss_fsp.item(), img.size(0))
        losses_cos.update(loss_cos.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == (args.print_freq - 1):
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}  '
                  'Data {data_time.avg:.3f}  '
                  'Loss_h {loss_h.avg:.4f}  '
                  'Loss_fsp {loss_fsp.avg:.4f}  '
                  'Loss_cos {loss_kl.avg:.4f}  '
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss_h=losses_h, #loss_s=losses_s,
                loss_fsp=losses_fsp, loss_kl=losses_cos))


# def val(val_list, model):
#     print('begin val')
#     val_loader = torch.utils.data.DataLoader(
#         dataset.listDataset(val_list,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                             std=[0.229, 0.224, 0.225]),
#                             ]), train=False),
#         num_workers=args.workers,
#         shuffle=False,
#         batch_size=args.batch_size)
#
#     model.eval()
#
#     mae = 0
#     mse = 0
#
#     for i, (img, target) in enumerate(val_loader):
#         img = img.cuda()
#         img = Variable(img)
#
#         with torch.no_grad():
#             output = model(img)
#
#         mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
#         mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)
#
#     N = len(val_loader)
#     mae = mae / N
#     mse = torch.sqrt(mse / N)
#     print('Val * MAE {mae:.3f} * MSE {mse:.3f}'
#           .format(mae=mae, mse=mse))
#
#     return mae, mse


def test(test_list, model, vis):
    print('testing current model...')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(test_list,
                            transform=transforms.Compose([
                                transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225]),
                            ]), train=False),
        num_workers=args.workers,
        shuffle=False,
        batch_size=args.batch_size)

    model.eval()

    mae = 0
    mse = 0
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img)

            with torch.no_grad():
                output = model(img)

            mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())
            mse += (output.data.sum() - target.sum().type(torch.FloatTensor).cuda()).pow(2)

        N = len(test_loader)
        mae = mae / N
        mse = torch.sqrt(mse / N)
        print('Test * MAE {mae:.3f} * MSE {mse:.3f} '
              .format(mae=mae, mse=mse))
        vis.plot('MAE', float(mae))
        vis.plot('MSE', float(mse))
        return mae, mse



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
