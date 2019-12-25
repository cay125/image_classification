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
from efficientnet_pytorch import EfficientNet
import grad_cam
import torch.optim.lr_scheduler as lr_scheduler
import label_smooth


def val(_val_loader: torch.utils.data.DataLoader, model: nn.Module) -> float:
    with torch.no_grad():
        total, correct = 0, 0
        model.eval()
        for i, (images, target) in enumerate(_val_loader):
            images, target = images.to(device), target.to(device)
            output = model(images)
            _, index = output.max(1)
            index = index.squeeze()
            total += images.shape[0]
            correct += torch.sum(index == target).item()
    acc = correct / total
    print('validate accuracy: {:.3f}'.format(acc))
    return acc


def train(train_loader: torch.utils.data.DataLoader, model: nn.Module, criterion: nn.Module, optimizer, epoch, f_id,
          begin):
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
        if args.hard_sample_mining == 'true':
            topk = int(0.7 * loss.shape[0])
            loss, _ = torch.topk(loss, topk)
            loss = torch.mean(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_avg.update(loss.item())
        time_avg.update(time.time() - end)
        begin.update(time.time())
        end = time.time()

        if i % params.display_interval == 0:
            print('[{0}/{1}][{2}/{3}] loss: {4:.4f} loss_avg: {5:.4f} total: {6} eta: {7:.1f}min'.format(
                epoch,
                args.epoch,
                i,
                len(train_loader),
                loss.item(),
                loss_avg.avg,
                begin,
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
    args.add_argument('--multiGPU', help='whther to use gpus to train model', default='false', type=str)
    args.add_argument('--type', help='train or test', default='train', type=str)
    args.add_argument('--hard_sample_mining', help='enable hard sample mining', default='false', type=str)
    args.add_argument('--elastic', help='enable elastic arch', default='false', type=str)
    args.add_argument('--cbam', help='enable cbam arch', default='false', type=str)
    args.add_argument('--show_grad', help='visualize cnn heatmap', default='false', type=str)
    args.add_argument('--val', help='validate while training', default='false', type=str)
    args.add_argument('--label_smooth', help='whether use label smooth', default='false', type=str)
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    print('record loss: {}'.format(args.record))
    print('iteration times: {}'.format(args.epoch))
    print('batch size: {}'.format(args.batch_size))
    print('useMultiGPUs? {}'.format(args.multiGPU))
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
    # model = nets.resnext101_elastic(num_classes=params.num_classes)  # type:nn.Module
    model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=params.num_classes,
                                         elastic=True if args.elastic == 'true' else False,
                                         cbam=True if args.cbam == 'true' else False)
    print('model type: {}'.format(type(model)))
    if args.hard_sample_mining == 'true':
        print('use hard sample mining strategy')
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    if args.label_smooth == 'true':
        print('use label smooth method')
        criterion = label_smooth.LabelSmoothSoftmaxCE()
    # optimizer = torch.optim.SGD(model.parameters(), params.lr if args.model_path is None else 0.001,
    #                             momentum=params.momentum, weight_decay=params.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, betas=(0.9, 0.999), eps=1e-9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

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
    input_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
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
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=params.workers, pin_memory=True)

    if args.type == 'train':
        # for m in model.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(m.weight)
        if args.model_path is not None:
            res = model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
            if res.missing_keys is not None and len(res.missing_keys) != 0:
                for missing_key in res.missing_keys:
                    if ('elastic' not in missing_key) and ('cbam' not in missing_key):
                        raise RuntimeError('load state dict error')
                print('missing keys only from elastic and cbam arch')
            print('loading model from {}'.format(args.model_path))
        if args.multiGPU == 'true' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print('model being parallelize')
        best_acc = 0
        best_model = None
        model = model.to(device)
        criterion = criterion.to(device)
        begin = utility.total_time(time.time())
        for epoch in range(args.epoch):
            # if args.model_path is None:
            #     utility.adjust_learning_rate(optimizer, epoch, params.lr)
            train(train_loader, model, criterion, optimizer, epoch, f_id, begin)
            if args.val == 'true':
                acc = val(val_loader, model)
                scheduler.step(acc * 100)
                if best_acc < acc:
                    best_acc = acc
                    best_model = model
            if args.multiGPU == 'true' and torch.cuda.device_count() > 1:
                torch.save(model.module.state_dict(), params.model_dir + 'training_net')
            else:
                torch.save(model.state_dict(), params.model_dir + 'training_net')
        t = time.strftime('%Y_%m_%d_%H_%M', time.localtime())
        cp = {}
        cp['idx_to_class'] = idx_to_class
        if args.multiGPU == 'true' and torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), params.model_dir + 'net_' + t)
            cp['state_dict'] = model.module.state_dict()
            cp['model'] = model.module
            if best_model is not None:
                best_cp = {}
                best_cp['idx_to_class'] = idx_to_class
                best_cp['state_dict'] = best_model.module.state_dict()
                best_cp['model'] = model.module
                torch.save(best_cp, params.model_dir + 'best_model_' + t)
        else:
            torch.save(model.state_dict(), params.model_dir + 'net_' + t)
            cp['state_dict'] = model.state_dict()
            cp['model'] = model
            if best_model is not None:
                best_cp = {}
                best_cp['idx_to_class'] = idx_to_class
                best_cp['state_dict'] = best_model.state_dict()
                best_cp['model'] = model
                torch.save(best_cp, params.model_dir + 'best_model_' + t)
        os.remove(params.model_dir + 'training_net')
        os.rename(params.model_dir + 'training_loss.txt', params.model_dir + 'loss_' + t + '.txt')
        torch.save(cp, params.model_dir + 'model_' + t)
        f_id.write('{0} {1} {2}\n'.format(best_acc, best_acc, best_acc))
        f_id.close()
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu')['state_dict'])  # type:nn.Module
        model.eval()
        img_sum, img_right_sum = [], []
        labels = os.listdir(valdir)
        model = model.to(device)
        if args.show_grad == 'true':
            model._conv_head.register_forward_hook(grad_cam.farward_hook)
            model._conv_head.register_backward_hook(grad_cam.backward_hook)
        for label_index, label in enumerate(labels):
            files = os.listdir(valdir + label)
            img_sum.append(len(files))
            img_right_sum.append(0)
            for file in files:
                img_ori = cv2.imread(valdir + label + '/' + file)  # type:np.ndarray
                img = cv2.resize(img_ori, (input_size, input_size))
                img = img.astype(np.float) / 255.0
                img = img.transpose(2, 0, 1)
                img = img[::-1].copy()
                img_tensor = torch.from_numpy(img).float()
                # img_tensor.sub_(0.5).div_(0.5)
                img_tensor = normalize(img_tensor).unsqueeze(0)
                img_tensor = img_tensor.to(device)
                res = model(img_tensor).squeeze()
                _, index = res.max(0)
                if idx_to_class[index.item()] == label:
                    img_right_sum[label_index] += 1

                if args.show_grad == 'true' and idx_to_class[index.item()] != label:
                    print('true label: {0}  predict label: {1}'.format(label, idx_to_class[index.item()]))
                    model.zero_grad()
                    class_loss = grad_cam.comp_class_vec(res.unsqueeze(0), None, device)
                    class_loss.backward()
                    grads = grad_cam.grad_block[-1].cpu().data.numpy().squeeze()
                    fmap = grad_cam.fmap_block[-1].cpu().data.numpy().squeeze()
                    cam, img_show = grad_cam.gen_cam(fmap, grads, img_ori)
                    cv2.imshow("cam", cam)
                    cv2.imshow("img", img_show)
                    cv2.waitKey(0)
            print('label {0} accuracy:{1:.3f}'.format(label, img_right_sum[label_index] / img_sum[label_index]))
        print('total accuracy:{0:.3f}'.format(sum(img_right_sum) / sum(img_sum)))

        # for img_idx, (img, target) in enumerate(train_loader):
        #     img = img[[0]]
        #     target = target[[0]]
        #     res = model(img)
        #     loss = criterion(res, target)
        #     _, index = res.max(1)
        #     print(index.item())
