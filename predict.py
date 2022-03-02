

import torch
import torchvision.transforms as transforms

from PIL import Image

from models import *

if __name__ == '__main__':
    pre_trans = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), # 能不能 在这里把alpha通道去掉？
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) # 这里的transform要和训练时一致

    transform2 = transforms.Compose(
        [transforms.Resize((32, 32))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('./save_models/LeNet.pth'))

    im = Image.open('plane.png')
    im = transforms.ToTensor()(im)
    im = im[:3]
    t2 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    im = t2(im)
    im = torch.unsqueeze(im, dim=0) # 扩展成(batch_size, channel, height, width)

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])
