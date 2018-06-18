'''
    MNIST training with SegNet+FC (end to end training)
    for SegNet+CNN end to end training, you can just replace FC model with CNN model, so I will not provide a repeated version

    Author: Chenxi Wang
    Date: June 2018
'''

from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_util
from torchvision import datasets, transforms
from PIL import Image

## get mnist dataset filename
#  original datasets
TRAIN_FILE = 'mnist_train_data'
TEST_FILE = 'mnist_test_data'
#  denoised datasets
TRAIN_FILE_CC = 'mnist_train_cc1.0_data'
TEST_FILE_CC = 'mnist_test_cc1.0_data'

BEST_ACC = 0.0
EPOCH_CNT = 0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Training Code')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()


class myMNIST(torch.utils.data.Dataset):
    ''' pytorch dataset class, used for load and get data'''
    def __init__(self, datapath, cls_labelpath, seg_labelpath):
        data = np.fromfile(datapath,dtype=np.uint8).reshape(-1,1,45,45)
        cls_label = np.fromfile(cls_labelpath,dtype=np.uint8)
        seg_label = np.fromfile(seg_labelpath,dtype=np.uint8).reshape(-1,45,45)
        self.data = data.astype(np.float32) / 256.0
        self.cls_label = cls_label.astype(np.int64)
        self.seg_label = (seg_label>0).astype(np.int64)

    def __getitem__(self, index):
        return self.data[index,...], self.cls_label[index], self.seg_label[index,...]

    def __len__(self):
        return self.data.shape[0]


class CNN(nn.Module):
    def __init__(self):
        '''CNN baseline implementation'''
        super(CNN, self).__init__()
        self.conv1 = torch_util.conv2d(1, 16, kernel_size=4)
        self.conv2 = torch_util.conv2d(16, 64, kernel_size=3)
        self.conv3 = torch_util.conv2d(64, 256, kernel_size=3)
        self.conv4 = torch_util.conv2d(256, 1024, kernel_size=4)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = torch_util.fully_connected(1024*3*3, 256)
        self.fc2 = torch_util.fully_connected(256, 64)
        self.fc3 = torch_util.fully_connected(64, 10)

    def forward(self, x):
        x = self.conv1(x) #[16,42,42]
        x = F.max_pool2d(self.conv2(x), 2) #[64,20,20]
        x = F.max_pool2d(self.conv3(x), 2) #[256,9,9]
        x = F.max_pool2d(self.conv3_drop(self.conv4(x)), 2) #[1024,3,3]
        x = x.view(-1, 1024*3*3)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class FC(nn.Module):
    '''FC baseline implementation'''
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(45*45, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 45*45)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class SegNet(nn.Module):
    '''SegNet implementation'''
    def __init__(self):
        super(SegNet, self).__init__()
        self.conv1 = torch_util.conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = torch_util.conv2d(16, 64, kernel_size=3, padding=1)
        self.conv3 = torch_util.conv2d(64, 256, kernel_size=3, padding=1)
        self.conv3_drop = nn.Dropout2d()
        self.conv4 = torch_util.conv2d(256, 64, kernel_size=1)
        self.conv4_drop = nn.Dropout2d()
        self.conv5 = torch_util.conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3_drop(self.conv3(x))
        x = self.conv4_drop(self.conv4(x))
        x = self.conv5(x)
        return F.log_softmax(x, dim=1)


class Seg_FC(nn.Module):
    '''SegNet+FC end to end training model implementation'''
    def __init__(self):
        super(Seg_FC, self).__init__()
        self.segnet = SegNet()
        self.nn = FC()

    def forward(self, x):
        # get segmentation mask from SegNet
        seg_score = self.segnet(x)
        seg_mask = seg_score.max(1, keepdim=True)[1].float()
        # mask image input
        x = x * seg_mask
        # get class score from FC model
        cls_score = self.nn(x)
        return seg_score, cls_score


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    ''' train the model in one epoch'''
    model.train()
    for batch_idx, (data, cls_target, seg_target) in enumerate(train_loader):
        # get data
        data, cls_target, seg_target = data.to(device), cls_target.to(device), seg_target.to(device)
        optimizer.zero_grad()
        # get prediction and loss (seg_loss + cls_loss)
        seg_score, cls_score = model(data)
        seg_loss = F.nll_loss(seg_score, seg_target)
        cls_loss = F.nll_loss(cls_score, cls_target)
        loss = seg_loss + cls_loss
        # update weights
        loss.backward()
        optimizer.step()
        # log training loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_one_epoch(args, model, device, test_loader):
    global BEST_ACC
    ''' test the model in one epoch'''
    img_cnt = 0
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, cls_target, seg_target in test_loader:
            # get data
            data, cls_target, seg_target = data.to(device), cls_target.to(device), seg_target.to(device)
            # get prediction and loss (seg loss + cls loss)
            seg_score, cls_score = model(data)
            test_loss += (F.nll_loss(cls_score, cls_target, size_average=False).item() + F.nll_loss(seg_score, seg_target, size_average=False).item()) # sum up batch loss
            seg_pred = seg_score.max(1, keepdim=True)[1] # get the index of the max log-probability
            cls_pred = cls_score.max(1, keepdim=True)[1]
            '''img_cnt = save_pred(seg_pred.view(-1,45,45).cpu().numpy(), img_cnt)'''
            correct += cls_pred.eq(cls_target.view_as(cls_pred)).sum().item()
    # get mean loss and acc, log results
    test_loss /= (45*45*len(test_loader.dataset))
    test_acc = 100. * correct / len(test_loader.dataset)
    BEST_ACC = max(test_acc, BEST_ACC)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))


def save_pred(pred, img_cnt):
    ''' save sgementation images, used for visualization'''
    pred = (pred*255).astype(np.uint8)
    for idx in range(pred.shape[0]):
        im = Image.fromarray(pred[idx,...])
        im.save("./seg_pred/%06d_%03d.png" % (img_cnt+idx,EPOCH_CNT))
    img_cnt += pred.shape[0]
    return img_cnt


def main():
    global EPOCH_CNT
    # get device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # get model and optimizer
    model = Seg_FC().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # get data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_train/'+TRAIN_FILE,
                cls_labelpath='../../mnist/mnist_train/mnist_train_label',
                seg_labelpath='../../mnist/mnist_train/'+TRAIN_FILE_CC),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_test/'+TEST_FILE,
                cls_labelpath='../../mnist/mnist_test/mnist_test_label',
                seg_labelpath='../../mnist/mnist_test/'+TEST_FILE_CC),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    # train model
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_one_epoch(args, model, device, test_loader)
        EPOCH_CNT += 1
    print('Best accuracy: %.2f%%' % BEST_ACC)

if __name__ == '__main__':
    main()
