"# -- coding: UTF-8 --"
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import torch.nn.functional as F
from data.dataset import Dataset
from loss.lovasz_loss import lovasz_softmax
from loss.metrics import Metrics
from net_v1 import ZZHnet
from tqdm import tqdm

def model_init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    print('Device:', device)
    model = ZZHnet.zzh_net()
    #model.load_state_dict(torch.load('/home/zzh/ZZHNet/best_checkpoint_' +name+ '/best_statedict_epoch49_f_score0.8515.pth'), strict=True)
    model = model.to(device)
    return model,device

def data_init():
    batch_size = 12
    img_size = 256
    train_pickle_file = '/home/zzh/remote_data/LEVIR-CD256/train'  # Path to training set
    val_pickle_file = '/home/zzh/remote_data/LEVIR-CD256/val'  # Path to validation set
    data_transforms = {
        'train': transforms.Compose([
            # 线性插值改变图片
            transforms.Resize(img_size),
            # 从图像的中心裁剪
            transforms.CenterCrop(img_size),
            # 以0.2的概率随机转换为灰度图
            transforms.RandomGrayscale(p=0.2),
            # 随机扰动，brightness和contrast分别控制亮度和对比度的变化范围，saturation和hue控制饱和度和色调的变化范围
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0, hue=0),
            # 将图像转换为PyTorch的Tensor格式，并归一化到[0, 1]区间
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
        ]),
    }
    train_dataset = Dataset(train_pickle_file, data_transforms['train'])
    val_dataset = Dataset(val_pickle_file, data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return {'train': train_loader, 'val': val_loader}


def train(model, device, data_loader,num_epochs):
    val_acc = []
    train_loss = []
    base_lr = 1e-3
    best_F1 = 0
    best_Pre = 0
    best_Rec = 0
    best_OA = 0
    best_loss = 0
    best_epoch =[0,0,0,0]
    #先来个预训练32残差跑边缘增强，跑通，看看得分,速度等等
    #再用武大的边缘增强单独跑，以及跟残差结合跑，对比结果变化，模型大小，速度变化等等
    #再结合异源
    #再考虑mamba
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    sc_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, verbose=True)
    #mode='max'表示调度器期望的指标是最大化，即当指标不再增加时，学习率会减少
    #patience=15表示调度器会等待15个epoch（或验证周期）来确认指标是否真的停止改善
    #verbose=True表示调度器会在每次调整学习率时打印一条消息

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 60)
#---------------------------------------------------------------------------------------------train
        model.train()
        # 在 PyTorch中，梯度是累积的，这意味着每次调用 .backward() 方法时，梯度都会累加到现有的梯度上。如果不将梯度清零，那么每次反向传播时，梯度都会累加，导致梯度计算错误。
        torch.cuda.empty_cache()
        running_loss_seg = 0.0
        #进度条
        process_bar = tqdm(data_loader['train'],desc="Training")
        iter = 1;
        for data in process_bar:
            A = data['A'].cuda().float()
            B = data['B'].cuda()
            Label = (data['label'] > 0).squeeze(1).type(torch.LongTensor).to(device)
            out = model(A,B) #输出的是两通道概率图，criterion中潜入了softmax

            optimizer.zero_grad()
            ce_loss_1 = F.cross_entropy(out, Label, ignore_index=255)
            lovasz_loss = lovasz_softmax(F.softmax(out, dim=1), Label, ignore=255)
            loss = ce_loss_1 + 0.75 * lovasz_loss

            writer.add_scalar("Loss/train", loss, iter)
            iter+=1
            loss.backward()
            optimizer.step()
            # statistics
            running_loss_seg += loss.item() * A.size(1)
            # 更新进度条
            process_bar.set_postfix(loss=loss.item())
        epoch_loss_seg = running_loss_seg / len(data_loader['train'].dataset)
        print('train | overall_Loss_seg: {:.6f}'.format(epoch_loss_seg))  # edge
        train_loss.append(epoch_loss_seg)
        print('lr:{}'.format(optimizer.param_groups[0]['lr']))
#---------------------------------------------------------------------------------------------val
        model.eval()
        torch.cuda.empty_cache()
        val_loss_seg = 0.0
        process_bar = tqdm(data_loader['val'],desc="Val")
        iter = 1;
        metrics = Metrics(range(num_class))
        for data in process_bar:
            #是否计算梯度，默认是True，推理和测试都不需要
            with torch.no_grad():
                A = data['A'].cuda().float()
                B = data['B'].cuda()
                Label = (data['label'] > 0).squeeze(1).type(torch.LongTensor).to(device)
                # 黑白图像只有1通道，这样操作label就只剩了 N H W,并且每个点上的值，先转换成了bool，又转换成了0或1
                #Label = (data['label'] > 0).squeeze(1).type(torch.LongTensor).to(device)
                out = model(A, B)  # 输出的是两通道概率图，criterion中潜入了softmax

                optimizer.zero_grad()
                ce_loss_1 = F.cross_entropy(out, Label, ignore_index=255)
                lovasz_loss = lovasz_softmax(F.softmax(out, dim=1), Label, ignore=255)
                loss = ce_loss_1 + 0.75 * lovasz_loss

                for mask, output in zip(Label, out):
                    metrics.add(mask, output)
                writer.add_scalar("Loss/Val", loss, iter)
                iter+=1
                # statistics
                val_loss_seg += loss.item() * A.size(1)
                process_bar.set_postfix(loss=loss.item())
        epoch_loss_seg = val_loss_seg / len(data_loader['val'].dataset)
        print('val | Loss_seg: {:.6f}'.format(epoch_loss_seg))  # edge
        precision = metrics.get_precision()
        recall = metrics.get_recall()
        f_score = metrics.get_f_score()
        oa = metrics.get_oa()
        print(
            'precision:{:.4f}, recall:{:.4f}, f_score:{:.4f}, oa:{:.4f}'
            .format(precision, recall, f_score, oa))
        sc_plt.step(f_score)  # 自适应学习率，调整学习率评价指标

        #记录一下得分的峰值
        if f_score > best_F1:
            best_F1 = f_score
            best_checkpoint = '/home/zzh/ZZHNet/result/best_checkpoint_v1/best_statedict_epoch{}_f_score{:.4f}.pth'.format(epoch, f_score)
            torch.save(model.state_dict(), best_checkpoint)
        if precision > best_Pre:
            best_Pre = precision
            best_epoch[0] = epoch+1
        if recall > best_Rec:
            best_Rec = recall
            best_epoch[1] = epoch + 1
        if oa > best_OA:
            best_OA = oa
            best_epoch[2] = epoch + 1
        if (epoch_loss_seg < best_loss) or best_loss==0:
            best_loss = epoch_loss_seg
            best_epoch[3] = epoch + 1
        val_acc.append(f_score)
        print('Best f_score: {:4f}'.format(best_F1))
        print('-' * 60)

    return val_acc, train_loss


# def val(model, device, val_loader):

if __name__ == '__main__':
    num_epochs = 100
    num_class = 2
    writer = SummaryWriter()
    model, device = model_init()
    data = data_init()
    val_acc, train_loss =train(model, device, data, num_epochs)
    writer.close()
    x = np.arange(0, num_epochs, 1)
    plt.figure()
    plt.plot(x, val_acc, 'r', label='val_f1')
    plt.savefig('/home/zzh/ZZHNet/result/train_v1/ZZHNet_Mamba_val_f_score.png')
    plt.figure()
    plt.plot(x, train_loss, 'g', label='train_loss')
    plt.savefig('/home/zzh/ZZHNet/result/train_v1/ZZHNet_Mamba_train_loss.png')


