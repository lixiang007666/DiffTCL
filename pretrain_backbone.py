# import torch
# import torch.nn.functional as F
# from torchvision.models import resnet18, resnet50
# from torchvision.datasets import ImageFolder
# import torch.utils.data as data
# from torchvision.transforms import (Compose, RandomCrop, Grayscale,
#                                     RandomHorizontalFlip, Resize, ToTensor)
# from src.mix_transformer import *


# def accuracy(Y_hat, Y, averaged=True):
#     Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
#     preds = torch.argmax(Y_hat, dim=1).type(Y.dtype)
#     # print(Y_hat.shape, preds.shape, Y.shape)
#     compare = (preds == Y).type(torch.float32)
#     return torch.mean(compare) if averaged else compare


# if __name__=="__main__":
#     # model = resnet50(weights=None)
#     input_channel = 1
#     # model.conv1 = torch.nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     model = MixVisionTransformer(in_chans=input_channel,num_classes=3)
#     model = model.cuda()
#     transforms = Compose(
#         [
#             # Grayscale(),
#             Grayscale(num_output_channels=1),
#             Resize(256),
#             RandomCrop(224),
#             RandomHorizontalFlip(),
#             ToTensor(),
#         ]
#     )
#     img_dataset = ImageFolder("refuge2",transform=transforms)
#     img_num = len(img_dataset)
#     train_num = int(0.9 * img_num)

#     max_epoch = 150
#     batch_size = 64

#     train_dataset,val_dataset = data.random_split(img_dataset,[train_num, img_num-train_num],
#                       generator=torch.Generator().manual_seed(42))
#     print(len(train_dataset),len(val_dataset))
#     train_loader = data.DataLoader(train_dataset,num_workers=12,batch_size=batch_size)
#     val_loader = data.DataLoader(val_dataset,num_workers=12,batch_size=batch_size)
#     optim = torch.optim.AdamW(model.parameters(), lr=0.0001)

#     train_loss = []
#     val_loss = []
#     val_acc = []
#     best_acc = 0
#     for epoch in range(max_epoch):
#         batch_train_loss = []
#         batch_val_loss = []
#         batch_val_acc = []
#         model.train()
#         for X, y in train_loader:
#             # print(X.shape)
#             X = X.float().cuda()
#             y = y.cuda()
#             y_hat = model(X)
#             # print(type(y_hat))
#             # print(y_hat)
#             # print(y)
#             # print(y_hat.shape)
#             loss = F.cross_entropy(y_hat, y)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()
#             batch_train_loss.append(loss)
#             # print(model.state_dict())
#         model.eval()
#         y_true, y_pred = torch.tensor([]).cuda(), torch.tensor([]).cuda()
#         with torch.no_grad():
#             for X, y in val_loader:
#                 X = X.float().cuda()
#                 y = y.cuda()
#                 y_hat = model(X)
#                 loss = F.cross_entropy(y_hat, y)
#                 acc = accuracy(y_hat, y)

#                 y_true = torch.concat([y_true, y], 0)
#                 preds = torch.argmax(y_hat, dim=1).type(y.dtype).cuda()
#                 y_pred = torch.concat([y_pred, preds], 0)

#                 batch_val_acc.append(acc)
#                 batch_val_loss.append(loss)
#         val = [torch.mean(torch.tensor(batch_train_loss)), torch.mean(torch.tensor(batch_val_loss)),
#                torch.mean(torch.tensor(batch_val_acc))]
#         print(f'epoch:{epoch} train loss:{val[0]}, 'f'val loss:{val[1]}, 'f'val acc:{val[2]}')
#         train_loss.append(val[0])
#         val_loss.append(val[1])
#         val_acc.append(val[2])
#         epoch_acc = val_acc[-1]
#         if epoch_acc > best_acc:
#             print("---------save best model----------")
#             torch.save(model.state_dict(), "pretrain_trans_gamma.pth")
#             best_acc = epoch_acc

import argparse
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
import torch.utils.data as data
from torchvision.transforms import (Compose, RandomCrop, Grayscale,
                                    RandomHorizontalFlip, Resize, ToTensor)
from src.mix_transformer import *


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(Y_hat, Y, averaged=True):
    Y_hat = torch.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = torch.argmax(Y_hat, dim=1).type(Y.dtype)
    compare = (preds == Y).type(torch.float32)
    return torch.mean(compare) if averaged else compare


def main(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')

    # model = resnet50(pretrained=False)
    model = MixVisionTransformer(in_chans=3,num_classes=3)
    model = model.cuda()
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    transforms = Compose(
        [
            Grayscale(num_output_channels=3),
            # Grayscale(),
            Resize(256),
            RandomCrop(224),
            RandomHorizontalFlip(),
            ToTensor(),
        ]
    )
    img_dataset = ImageFolder("refuge2", transform=transforms)
    img_num = len(img_dataset)
    train_num = int(0.9 * img_num)

    train_dataset, val_dataset = data.random_split(img_dataset, [train_num, img_num - train_num],
                                                   generator=torch.Generator().manual_seed(42))
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_loader = data.DataLoader(train_dataset, num_workers=12, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = data.DataLoader(val_dataset, num_workers=12, batch_size=args.batch_size, sampler=val_sampler)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.8, 0.99))

    best_acc = 0
    val_acc = []

    for epoch in range(args.epochs):
        adjust_learning_rate(optim, epoch, args)

        train_sampler.set_epoch(epoch)
        batch_train_loss = []
        batch_val_loss = []
        batch_val_acc = []
        model.train()
        for X, y in train_loader:
            X = X.float().cuda()
            y = y.cuda()
            y_hat = model(X)
            loss = F.cross_entropy(y_hat, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_train_loss.append(loss.item())

        model.eval()
        y_true, y_pred = torch.tensor([]).cuda(), torch.tensor([]).cuda()
        with torch.no_grad():
            for X, y in val_loader:
                X = X.float().cuda()
                y = y.cuda()
                y_hat = model(X)
                loss = F.cross_entropy(y_hat, y)
                acc = accuracy(y_hat, y)
                y_true = torch.cat([y_true, y], 0)
                preds = torch.argmax(y_hat, dim=1).type(y.dtype).cuda()
                y_pred = torch.cat([y_pred, preds], 0)
                batch_val_acc.append(acc.item())
                batch_val_loss.append(loss.item())

        val = [sum(batch_train_loss) / len(batch_train_loss),
               sum(batch_val_loss) / len(batch_val_loss),
               sum(batch_val_acc) / len(batch_val_acc)]
        print(f'epoch:{epoch} train loss:{val[0]}, 'f'val loss:{val[1]}, 'f'val acc:{val[2]}')
        val_acc.append(val[2])

        epoch_acc = val_acc[-1]
        if epoch_acc > best_acc and dist.get_rank() == 0:
            print("---------save best model----------")
            torch.save(model.module.state_dict(), "pretrain_trans_gamma.pth")
            best_acc = epoch_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, dest='local_rank')
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--min_lr", type=float, default=0.00001)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)