import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms

import torch.optim as optim
import torch.backends.cudnn as cudnn

import datetime

from models import *

import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 反转经过了transforms_test的图片，并显示
def imshow_transform_img(img):
    img = img / 2.0 + 0.5  # 反转transforms.Normalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 反转transforms.ToTensor
    plt.show()


if __name__ == '__main__':
    # 准备数据集
    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),  # 相当于padding完之后，再crop
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # 定义rgb3个维度的均值和方差
    ])

    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 50000张训练图片
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=36, shuffle=True, num_workers=0)

    # 10000张验证图片
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_test)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=10000, shuffle=False, num_workers=0)

    val_images, val_labels = iter(val_loader).next()

    # num_to_show = 4
    # show_images_labels = val_labels[:num_to_show]
    # show_images = val_images[:num_to_show]
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # print(' '.join('%5s' % classes[show_images_labels[j]] for j in range(num_to_show)))
    # imshow_transform_img(torchvision.utils.make_grid(show_images))

    net = LeNet()
    save_path = './save_models/LeNet.pth'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        val_images, val_labels = val_images.to(device), val_labels.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) # 变化的学习率

    print('Start Training: ', datetime.datetime.now())
    for epoch in range(5):
        net.train()
        running_loss = 0.0
        print_step = 500
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            net.zero_grad()  # 清空梯度
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()  # 反向传播loss
            optimizer.step()  # 更新参数

        # 打印训练信息
        running_loss += loss.item()
        with torch.no_grad():
            val_outputs = net(val_images)
            predict_labels = torch.max(val_outputs, dim=1)[1]
            accuracy = torch.eq(val_labels, predict_labels).sum().item() / val_labels.size(0)
            print('[%d, %5d] train_loss: %.3f, accuracy: %.3f time: %s' % (
                epoch + 1, step + 1, running_loss / print_step, accuracy, datetime.datetime.now()))
            running_loss = 0.0

        scheduler.step()

    print('Finish Training: ', datetime.datetime.now())

    torch.save(net.state_dict(), save_path)



