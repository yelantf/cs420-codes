'''
    MNIST training with FC baseline models

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
from torchvision import datasets, transforms
import torch_util

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
    def __init__(self, datapath, labelpath):
        data = np.fromfile(datapath,dtype=np.uint8).reshape(-1,1,45,45)
        label = np.fromfile(labelpath,dtype=np.uint8)
        self.data = data.astype(np.float32) / 256.0
        self.label = label.astype(np.int64)

    def __getitem__(self, index):
        return self.data[index,...], self.label[index]

    def __len__(self):
        return self.data.shape[0]


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
    BEST_ACC = max(BEST_ACC, test_acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))


def main():
    # get device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    # get model and optimizer
    model = FC().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # get data loader
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_train/'+TRAIN_FILE_CC,
                labelpath='../../mnist/mnist_train/mnist_train_label'),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='../../mnist/mnist_test/'+TEST_FILE_CC,
                labelpath='../../mnist/mnist_test/mnist_test_label'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    # train model
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_one_epoch(args, model, device, test_loader)
    print("Best accuracy: %.2f%%" % BEST_ACC)

if __name__ == '__main__':
    main()