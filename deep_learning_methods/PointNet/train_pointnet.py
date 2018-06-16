'''
    MNIST training with PointNet (no input T-Net)

    Author: Chenxi Wang
    Date: June 2018
'''
from __future__ import print_function
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_util
from torchvision import datasets, transforms

## get mnist dataset filename
#  original datasets
TRAIN_FILE = 'mnist_train_data'
TEST_FILE = 'mnist_test_data'
#  denoised datasets
TRAIN_FILE_CC = 'mnist_train_cc1.0_data'
TEST_FILE_CC = 'mnist_test_cc1.0_data'
#  digit centering datasets
TRAIN_FILE_CC_CENTERED = 'mnist_train_cc1.0_crop45_data'
TEST_FILE_CC_CENTERED = 'mnist_test_cc1.0_crop45_data'

BEST_ACC = 0.0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Training Code')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
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
    def __init__(self, datapath, labelpath):
        data = np.fromfile(datapath,dtype=np.uint8).reshape(-1,45,45)
        label = np.fromfile(labelpath,dtype=np.uint8)
        self.data = data.astype(np.float32) / 256.0
        self.label = label.astype(np.int64)
        self.cache = {}

    def __getitem__(self, index):
        if index not in self.cache:
            tmp_data = self.data[index,...]
            gxy = np.zeros((45*45,3),dtype=np.float32)
            # data format convertion
            # from image matrix to point sets (g,x,y)
            for i in range(45):
                for j in range(45):
                    gxy[i*45+j] = [i/44.,j/44.,tmp_data[j,i]]
            self.cache[index] = (gxy, self.label[index])
        return self.cache[index]

    def __len__(self):
        return self.data.shape[0]


class feature_transform_net(nn.Module):
    '''
        Feature T-Net implementation
        Sub model to PointNet
    '''
    def __init__(self, in_channels, momentum, num_point, batch_size):
        super(feature_transform_net, self).__init__()
        self.batch_size = batch_size
        self.conv1 = torch_util.conv2d(in_channels, 64, (1,1), momentum)
        self.conv2 = torch_util.conv2d(64, 128, (1,1), momentum)
        self.conv3 = torch_util.conv2d(128, 1024, (1,1), momentum)
        self.maxpool = nn.MaxPool2d((num_point, 1))
        self.fc1 = torch_util.fully_connected(1024, 512)
        self.fc2 = torch_util.fully_connected(512, 256)
        self.fc3 = nn.Linear(256, 64*64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = torch.squeeze(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        bias = Variable(torch.eye(64).view(64*64).repeat(self.batch_size,1).cuda())
        x = torch.add(x, bias)
        x = x.view(self.batch_size, 64, 64)
        return x


class PointNet(nn.Module):
    '''PointNet implementation (no input T-Net)'''
    def __init__(self, num_point, batch_size, momentum=0.1):
        super(PointNet, self).__init__()
        self.conv1 = torch_util.conv2d(in_channels=1, out_channels=64, kernel_size=(1,3), momentum=momentum)
        self.conv2 = torch_util.conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), momentum=momentum)
        self.conv3 = torch_util.conv2d(in_channels=64, out_channels=64, kernel_size=(1,1), momentum=momentum)
        self.conv4 = torch_util.conv2d(in_channels=64, out_channels=128, kernel_size=(1,1), momentum=momentum)
        self.conv5 = torch_util.conv2d(in_channels=128, out_channels=1024, kernel_size=(1,1), momentum=momentum)
        self.fc1 = torch_util.fully_connected(1024, 512)
        self.fc2 = torch_util.fully_connected(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.feature_transform = feature_transform_net(in_channels=64, momentum=momentum, num_point=num_point, batch_size=batch_size)
        self.maxpool = nn.MaxPool2d((num_point, 1))
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.7)
        self.drop2 = nn.Dropout(p=0.7)

    def forward(self, point_cloud):
        net = point_cloud.unsqueeze(1)
        net = self.conv1(net)
        net = self.conv2(net)
        feat_trans = self.feature_transform(net)
        net = torch.bmm(torch.squeeze(net).transpose(1,2), feat_trans)
        net = net.transpose(1,2).unsqueeze(-1)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)
        net = torch.squeeze(self.maxpool(net))
        net = self.fc1(net)
        net = self.drop1(net)
        net = self.fc2(net)
        net = self.drop2(net)
        net = self.fc3(net)
        return F.log_softmax(net, dim=1)


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    ''' train the model in one epoch'''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # get data
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # get prediction and loss
        output = model(data)
        loss = F.nll_loss(output, target)
        # update weights
        loss.backward()
        optimizer.step()
        # log training loss
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_one_epoch(args, model, device, test_loader):
    ''' test the model in one epoch'''
    global BEST_ACC
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # get data
            data, target = data.to(device), target.to(device)
            # get prediction and loss
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    # get mean loss and acc, log results
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    BEST_ACC = max(test_acc, BEST_ACC)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))


def main():
    # get device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # get model and optimizer
    model = PointNet(128,64).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # get data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_train/'+TRAIN_FILE,
                labelpath='../../mnist/mnist_train/mnist_train_label'),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_test/'+TEST_FILE,
                labelpath='../../mnist/mnist_test/mnist_test_label'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    # train model
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_one_epoch(args, model, device, test_loader)
    print('Best accuracy: %.2f%%' % BEST_ACC)

if __name__ == '__main__':
    main()