import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from collections import OrderedDict
import argparse
import random
import numpy as np
import os
import params
import socket
import time
import utility
import nets


def train(train_loader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module, optimizer, epoch, f_id):
    # switch to train mode
    model.train()
    loss_avg = utility.AverageMeter()
    time_avg = utility.AverageMeter()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item())
        time_avg.update(time.time() - end)
        end = time.time()

        if i % params.display_interval == 0:
            print('[{0}/{1}][{2}/{3}] loss: {4} loss_avg: {5} total: {6:.1f}min eta: {7:.1f}min'.format(
                epoch,
                args.epoch,
                i,
                len(train_loader),
                loss.item(),
                loss_avg.avg,
                time_avg.sum / 60.0,
                time_avg.avg * (len(train_loader) * (args.epoch - epoch) - i - 1) / 60.0))
            if args.record == 'true':
                f_id.write('{0} {1} {2}\n'.format(epoch, i, loss.item()))
            loss_avg.reset()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--model_path', help='pretrained model path', default=None, type=str)
    args.add_argument('--record', help='record loss', default='false', type=str)
    args.add_argument('--epoch', help='iteration times', default=50, type=int)
    args.add_argument('--batch_size', help='batch size', default=16, type=int)
    args.add_argument('--cuda_device', help='index of cuda device', default='3', type=str)
    args.add_argument('--type', help='train or test', default='train', type=str)
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print('record loss: {}'.format(args.record))
    print('iteration times: {}'.format(args.epoch))
    print('batch size: {}'.format(args.batch_size))
    manualSeed = 10
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running in device: ', end='')
    print(device)
    if not os.path.exists(params.model_dir):
        os.mkdir(params.model_dir)
    f_id = None
    if args.record == 'true':
        loss_filename = params.model_dir + 'training_loss.txt'
        f_id = open(loss_filename, 'w')

    # model = models.resnet50(pretrained=True)  # type:nn.Module
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, params.num_classes)
    model = nets.resnext50_elastic(num_classes=params.num_classes)  # type:nn.Module
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), params.lr, momentum=params.momentum,
                                weight_decay=params.weight_decay)

    # Data loading code
    hostname = socket.gethostname()
    if hostname == 'DESKTOP-JBG1JGC':
        data_dir_root = 'C:/Users/xiangpu/Downloads/src_v2_20191120/HUAWEI/'
    else:
        data_dir_root = '/root/myDataSet/HUAWEI/'
    traindir = data_dir_root + 'train/'
    valdir = data_dir_root + 'val/'
    if not os.path.exists(data_dir_root):
        raise RuntimeError('DataSet not exists')
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    idx_to_class = OrderedDict()
    for key, value in train_dataset.class_to_idx.items():
        idx_to_class[value] = key

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=params.workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=params.workers, pin_memory=True)

    if args.type == 'train':
        if args.model_path is not None:
            model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
            print('loading model from {}'.format(args.model_path))
        model = model.to(device)
        criterion = criterion.to(device)
        for epoch in range(args.epoch):
            utility.adjust_learning_rate(optimizer, epoch, params.lr)
            train(train_loader, model, criterion, optimizer, epoch, f_id)
            torch.save(model.state_dict(), params.model_dir + 'training_net')
        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        torch.save(model.state_dict(), params.model_dir + 'net_' + t)
        os.remove(params.model_dir + 'training_net')
        os.rename(params.model_dir + 'training_loss.txt', params.model_dir + 'loss_' + t + '.txt')
        cp = {}
        cp['state_dict'] = model.state_dict()
        cp['idx_to_class'] = idx_to_class
        torch.save(cp, params.model_dir + 'model_' + t)
        f_id.close()
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))  # type:nn.Module
        model.eval()
        img_sum, img_right_sum = [], []
        labels = os.listdir(valdir)
        model = model.to(device)
        for label_index, label in enumerate(labels):
            files = os.listdir(valdir + label)
            img_sum.append(len(files))
            img_right_sum.append(0)
            for file in files:
                img = cv2.imread(valdir + label + '/' + file)  # type:np.ndarray
                img = cv2.resize(img, (224, 224))
                img = img.astype(np.float) / 255.0
                img = img.transpose(2, 0, 1)
                img = img[::-1].copy()
                img_tensor = torch.from_numpy(img).float().unsqueeze(0)
                img_tensor.sub_(0.5).div_(0.5)
                img_tensor = img_tensor.to(device)
                res = model(img_tensor).squeeze()
                _, index = res.max(0)
                if idx_to_class[index.item()] == label:
                    img_right_sum[label_index] += 1
            print('label {0} accuracy:{1:.3f}'.format(label, img_right_sum[label_index] / img_sum[label_index]))
        print('total accuracy:{0:.3f}'.format(sum(img_right_sum) / sum(img_sum)))

        # for img_idx, (img, target) in enumerate(train_loader):
        #     img = img[[0]]
        #     target = target[[0]]
        #     res = model(img)
        #     loss = criterion(res, target)
        #     _, index = res.max(1)
        #     print(index.item())
