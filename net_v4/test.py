#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import glob
import collections
import torch
import torch.nn.functional as F
from torchvision import transforms
import datetime
import os
import argparse
import cv2

from logsetting import get_log
from loss.metrics import Metrics
from net_v4 import ZZHnet

device = 'cuda'
path = './dataset'

def model_init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model = ZZHnet.zzh_net(num_class=2)
    model.load_state_dict(torch.load('/home/zzh/ZZHNet/result/best_checkpoint_' +name+ '/best_statedict_epoch76_f_score0.8899.pth'), strict=True)
    model = model.to(device)
    model.eval()
    return model,device

def test(num_classes, net, files, device,img_size):
    trf = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ])
    metrics = Metrics(range(num_classes))
    image_path1 = glob.glob(files + '/A' + '/*.png')
    image_path2 = glob.glob(files + '/B' + '/*.png')
    masks_path = glob.glob(files + '/label' + '/*.png')
    with torch.set_grad_enabled(False):
        for i in range(len(masks_path)):
            images1 = Image.open(image_path1[i])
            images2 = Image.open(image_path2[i])
            masks = Image.open(masks_path[i])
            images1 = trf(images1).unsqueeze(0).to(device)
            images2 = trf(images2).unsqueeze(0).to(device)
            masks = trf(masks)
            masks = (masks > 0).squeeze(1).type(torch.LongTensor).to(device)

            # images1 = images1.unsqueeze(0)
            # images2 = images2.unsqueeze(0)
            # image_input = torch.cat([images1, images2], dim=0)
            out = net(images1, images2)
            print('load:{:d}/{:d}'.format(i, len(masks_path)))

            # save
            _, preds = torch.max(out, 1)
            preds = torch.reshape(preds, (img_size, img_size))
            preds[preds == 0] = 0
            preds[preds == 1] = 255
            preds = preds.cpu().numpy()
            basename = os.path.basename(masks_path[i])
            cv2.imwrite('/home/zzh/ZZHNet/result/test_' +name+img+ '/' + 'pre_' + basename, preds)

            for mask, output in zip(masks, out):
                metrics.add(mask, output)

    return {
        "precision": metrics.get_precision(),
        "recall": metrics.get_recall(),
        "f_score": metrics.get_f_score(),
        "oa": metrics.get_oa(),
        "kappa": metrics.kappa(),
        "iou": metrics.get_miou()
    }


if __name__ == '__main__':
    num_classes = 2
    img_size = 1024
    name = "v4"
    img = '_1024'
    ##加载模型
    model, device = model_init()

    logger = get_log("/home/zzh/ZZHNet/result/logs/" + str(datetime.date.today()) + 'test_log' +name+img+ '.txt')
    test_datapath = '/home/zzh/remote_data/LEVIR-CD/test'
    test_hist = test(num_classes, model, test_datapath, device, img_size)
    logger.info(('precision={}'.format(test_hist["precision"]),
                 'recall={}'.format(test_hist["recall"]),
                 'f_score={}'.format(test_hist["f_score"]),
                 'oa={}'.format(test_hist["oa"]),
                 'kappa={}'.format(test_hist["kappa"]),
                 'iou={}'.format(test_hist["iou"])))
    history = collections.defaultdict(list)
    for k, v in test_hist.items():
        history["test " + k].append(v)




