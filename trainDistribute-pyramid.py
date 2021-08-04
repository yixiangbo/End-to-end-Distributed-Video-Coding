"""
1215用于图片大小为32*32且无边信息时的训练
1229更新BPP的计算公式 与迭代次数有关
参考链接https://github.com/tensorflow/models/tree/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/compression/image_encoder
0307 训练数据集viemo 大小64*64 测试数据集foreman 原大小
0317 训练数据集viemo 大小64*64 测试数据集foreman 原大小 前后两帧作为边信息
0318 在之前的基础上加上：将边信息加入到不同的解码层中
"""
import time
import os
import sys
import math
import argparse

import numpy as np

import torch
from metric import *
import torch.optim as optim
import torch.optim.lr_scheduler as LS
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
from torchvision import transforms
import configparser


def get_args(filename,section):
    args = {}
    config = configparser.RawConfigParser()
    config.read(filename)
    options = config.options(section)
    print(len(options))
    for t in range(len(options)):
        if config.get(section,options[t] ).isdigit():
            args[options[t]] = int(config.get(section,options[t] ))
        else:
            try:
                float(config.get(section,options[t] ))
                args[options[t]] = float(config.get(section,options[t] ))
            except:
                args[options[t]] = config.get(section, options[t])
    return args

## load 32x32 patches from images
import datasetDistribute0318

## load networks on GPU
import networkDistribute_pyramid as network


def load_img(img_path,batch_size1):
    train_set = datasetDistribute0318.ImageFolder(is_train=True, root=img_path)
    train_loader = data.DataLoader(
        dataset=train_set, batch_size=batch_size1, shuffle=True, num_workers=1)
    print('total images: {}; total batches: {}'.format(
        len(train_set), len(train_loader)))
    return train_loader


def resume(epoch=None):
    if epoch is None:
        s = 'iter'
        epoch = 0
    else:
        s = 'epoch'

    encoder.load_state_dict(
        torch.load('checkpoint_{}/encoder_{}_{:08d}.pth'.format(args['bits'], s, epoch)))
    binarizer.load_state_dict(
        torch.load('checkpoint_{}/binarizer_{}_{:08d}.pth'.format(args['bits'], s, epoch)))
    decoder.load_state_dict(
        torch.load('checkpoint_{}/decoder_{}_{:08d}.pth'.format(args['bits'], s, epoch)))

def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    #image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    #out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out


def save(index, epoch=True):
    if not os.path.exists('level_{}/flag_{}/checkpoint_{}'.format(args['level'], args['flag'], args['iterations'])):
        # os.mkdir('flag_{}/checkpoint_{}'.format(args.flag, args.iterations))
        os.makedirs('level_{}/flag_{}/checkpoint_{}'.format(args['level'], args['flag'], args['iterations']))  # 创建多级目录

    # if not os.path.exists('checkpoint_{}'.format(args.iterations)):
    #     os.mkdir('checkpoint_{}'.format(args.iterations))

    if epoch:
        s = 'epoch'
    else:
        s = 'iter'

    torch.save(encoder.state_dict(), 'level_{}/flag_{}/checkpoint_{}/encoder_{}_{:05d}.pth'.format(
        args['level'], args['flag'], args['iterations'], s, index))

    torch.save(binarizer.state_dict(),
               'level_{}/flag_{}/checkpoint_{}/binarizer_{}_{:05d}.pth'.format(
                   args['level'], args['flag'], args['iterations'], s, index))

    torch.save(decoder.state_dict(), 'level_{}/flag_{}/checkpoint_{}/decoder_{}_{:05d}.pth'.format(
        args['level'], args['flag'], args['iterations'], s, index))

def flag_judge(flag,imgPre,imgMid,imgNext):
    if flag == 1:  # 前
        dataSide = imgPre
    elif flag == 2:  # 中
        dataSide = imgMid
    elif flag == 3:  # 后
        dataSide = imgNext
    elif flag == 4:  # 噪声
        dataSide = torch.FloatTensor(gasuss_noise(imgMid.numpy()))
    #elif flag == 5:  # 前一帧和后一帧
    #    dataSide = x = torch.cat([imgPre, imgNext], dim=1)
    elif flag == 5:
        dataSide = (imgPre + imgNext) / 2
    return dataSide


def psnr01(img1, img2):
    #mse = np.mean( (img1/255. - img2/255.) ** 2 )
    mse = np.mean((img1/1.0 - img2/1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def init_lstm(data):
    #global encoder_h_1,encoder_h_2,encoder_h_3,decoder_h_1,decoder_h_2,decoder_h_3,decoder_h_4
    encoder_h_1 = (Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()),
                   Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()))
    encoder_h_2 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()),
                   Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()))
    encoder_h_3 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()),
                   Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()))

    decoder_h_1 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()),
                   Variable(torch.zeros(data.size(0), 512, data.size(2) // 16, data.size(3) // 16).cuda()))
    decoder_h_2 = (Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()),
                   Variable(torch.zeros(data.size(0), 512, data.size(2) // 8, data.size(3) // 8).cuda()))
    decoder_h_3 = (Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()),
                   Variable(torch.zeros(data.size(0), 256, data.size(2) // 4, data.size(3) // 4).cuda()))
    decoder_h_4 = (Variable(torch.zeros(data.size(0), 128, data.size(2) // 2, data.size(3) // 2).cuda()),
                   Variable(torch.zeros(data.size(0), 128, data.size(2) // 2, data.size(3) // 2).cuda()))
    return encoder_h_1,encoder_h_2,encoder_h_3,decoder_h_1,decoder_h_2,decoder_h_3,decoder_h_4
def calc_psnr_mmsim(image,data,index,PSNR,MS_SSIM,BPP):
    PSNR[index] = psnr01(data.cpu().numpy(), image.cpu().numpy())
    MS_SSIM[index] = msssim(data.cpu().numpy() * 255, image.cpu().numpy() * 255)
    if index == 0:
        # BPP = (codes.size(1) * codes.size(2) * codes.size(3) * 8) / (output.size(1) * output.size(2) * output.size(3))
        # BPP = args.bits / 256  # 1214 再次更新BPP计算公式
        """
        1215 计算bpp
        BPP = (H/16 * W/16 * args.bits * args.iterations)/(H * W) = (args.bits * args.iterations)/256
        1229 BPP计算公式正确 但是应该修改的是迭代次数
        """
        BPP = (args['bits'] * args['iterations']) / 256
        print("BPP : {:4f}".format(BPP))
def draw_plot(PSNR,MS_SSIM,BPP,index):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()  # 让2个子图的x轴一样，同时创建副坐标轴。
    lns1 = ax1.plot(PSNR, color='red', label='PSNR')
    lns2 = ax2.plot(MS_SSIM, color='green', label='MS_SSIM')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('PSNR')
    ax2.set_ylabel('MS_SSIM')
    plt.title("BPP = {:.4f}".format(BPP))
    filename = "BPP = {:.4f}".format(BPP)

    lns = lns1 + lns2
    labels = ['PSNR', 'MS_SSIM']
    plt.legend(lns, labels, loc=7)
    plt.xlim(0, index)
    plt.savefig('./PSNR_Image/' + filename + '.png')
    plt.show()
def net_train(level,flag,train_loader,iterations,BPP,lr,max_epochs,save_epochs):
    solver = optim.Adam(
        [
            {
                'params': encoder.parameters()
            },
            {
                'params': binarizer.parameters()
            },
            {
                'params': decoder.parameters()
            },
        ],
        lr=lr)
    scheduler = LS.MultiStepLR(solver, milestones=[3, 10, 20, 50, 100], gamma=0.5)

    last_epoch = 0
    #if checkpoint:
    #    resume(checkpoint)
     #   last_epoch = checkpoint
    #    scheduler.last_epoch = last_epoch - 1
    for epoch in range(last_epoch + 1, max_epochs + 1):

        scheduler.step()

        for batch, (imgAll, filename, filenamePre, filenameNext) in enumerate(train_loader):
            batch_t0 = time.time()

            imgAll = torch.cat((imgAll[0], imgAll[1]), dim=0)

            imgPre = imgAll[:, 0:3, :, :]
            imgMid = imgAll[:, 3:6, :, :]
            imgNext = imgAll[:, 6:9, :, :]
            #compressed frame
            data = imgMid
            #side frame judge
            dataSide = flag_judge(flag,imgPre,imgMid,imgNext)

            patches = Variable(data.cuda())
            dataSide = Variable(dataSide.cuda())

            ## init lstm state
            #init_lstm(data)
            
                           
            encoder_h_1,encoder_h_2,encoder_h_3,decoder_h_1,decoder_h_2,decoder_h_3,decoder_h_4 = init_lstm(data)

            solver.zero_grad()

            losses = []

            res = patches - 0.5
            dataSide = dataSide - 0.5

            bp_t0 = time.time()

            image = torch.zeros(data.size()) + 0.5

            # sum_energy_encoded = 0
            for iteration in range(iterations):
                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    res, encoder_h_1, encoder_h_2, encoder_h_3)

                # sum_energy_encoded += np.sum(np.abs(encoded.data.cpu().numpy()))

                codes = binarizer(encoded)

                output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4, dataSide)

                res = res - output
                losses.append(res.abs().mean())
                image = image + output.data.cpu()  # additive reconstruction

                index = (epoch - 1) * len(train_loader) + batch
                calc_psnr_mmsim(image,data,index, PSNR, MS_SSIM,BPP)

            bp_t1 = time.time()

            loss = sum(losses) / iterations
            loss.backward()


            solver.step()

            batch_t1 = time.time()
            
            print(  # 华为云服务器版本升级修改
                '[TRAIN] Epoch[{}]({}/{}); Loss: {:.6f}; Backpropagation: {:.4f} sec; Batch: {:.4f} sec'.
                format(epoch, batch + 1,
                len(train_loader), loss.data, bp_t1 - bp_t0, batch_t1 -
                batch_t0))
            
            print("\t\tindex :{}; PSNR[{}] :{:.6f}; MS_SSIM[{}] :{:.6f}"
                  .format(index, index, PSNR[index], index, MS_SSIM[index]))

            # save checkpoint every 500 training steps
            if epoch %5==0:
                save(epoch)
def train_DataLevel_log(level,flag,BPP,PSNR,MS_SSIM,index):
    with open("trainDataLevel.txt", "a+") as f:
        f.writelines(
            "Train:\tlevel: {}; flag: {}; BPP: {:.4f}; PSNR: {:.4f}; MS_SSIM: {:.4f};\n".format(args.level, args.flag,
                                                                                                BPP, PSNR[index],
                                                                                                MS_SSIM[index]))
args = get_args("config.ini",'train')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #  指定GPU

encoder = network.EncoderCell().cuda()
binarizer = network.Binarizer().cuda()
decoder = network.DecoderCell().cuda()

train_loader = load_img(args['train'],args['batch_size'])
PSNR = np.zeros(args['max_epochs']*len(train_loader))
MS_SSIM = np.zeros(args['max_epochs']*len(train_loader))
BPP = 0
net_train(args['level'],args['flag'],train_loader,args['iterations'],BPP,args['lr'],args['max_epochs'],5)
train_DataLevel_log(args['level'],args['flag'],BPP,PSNR,MS_SSIM,index)



