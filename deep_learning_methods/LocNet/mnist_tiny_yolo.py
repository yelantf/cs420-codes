from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_util
from torchvision import datasets, transforms

TRAIN_FILE = 'mnist_train_data'
TEST_FILE = 'mnist_test_data'
TRAIN_FILE_CC = 'mnist_train_cc1.0_data'
TEST_FILE_CC = 'mnist_test_cc1.0_data'
TRAIN_FILE_CC_CENTERED = 'mnist_train_cc1.0_crop45_data'
TEST_FILE_CC_CENTERED = 'mnist_test_cc1.0_crop45_data'

BEST_ACC = 0.0
MIN_LOSS = 100.0
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

    def __init__(self, datapath, cls_labelpath, loc_labelpath):
        data = np.fromfile(datapath,dtype=np.uint8).reshape(-1,1,45,45)
        cls_label = np.fromfile(cls_labelpath,dtype=np.uint8)
        loc_label = np.fromfile(loc_labelpath,dtype=np.float32).reshape(-1,4)
        print(np.mean(loc_label[:,2]), np.mean(loc_label[:,3]))
        self.data = data.astype(np.float32) / 256.0
        self.cls_label = cls_label.astype(np.int64)
        self.loc_label = loc_label
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            data = np.zeros((3,45,45),dtype=np.float32)
            data[0,:,:] = self.data[index,...]
            for i in range(45):
                for j in range(45):
                    data[1,j,i] = i / 45.0
                    data[2,j,i] = j / 45.0
            self.cache[index] = (data, self.cls_label[index], self.loc_label[index])
            return data, self.cls_label[index], self.loc_label[index]

    def __len__(self):
        return self.data.shape[0]


class TinyYolo(nn.Module):
    def __init__(self):
        super(TinyYolo, self).__init__()
        self.conv1 = torch_util.conv2d(3, 16, kernel_size=4)
        self.conv2 = torch_util.conv2d(16, 64, kernel_size=3)
        self.conv3 = torch_util.conv2d(64, 256, kernel_size=3)
        self.conv4 = torch_util.conv2d(256, 1024, kernel_size=4)
        self.conv4_drop = nn.Dropout2d()
        self.fc1 = torch_util.fully_connected(1024*3*3, 256)
        self.fc2 = torch_util.fully_connected(256, 64)
        self.fc3 = torch_util.fully_connected(64, 4)

    def forward(self, x):
        x = self.conv1(x) #[16,42,42]
        x = F.max_pool2d(self.conv2(x), 2) #[64,20,20]
        x = F.max_pool2d(self.conv3(x), 2) #[256,9,9]
        x = F.max_pool2d(self.conv4_drop(self.conv4(x)), 2) #[1024,3,3]
        x = x.view(-1, 1024*3*3)
        x = self.fc1(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return x


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, cls_target, loc_target) in enumerate(train_loader):
        data, cls_target, loc_target = data.to(device), cls_target.to(device), loc_target.to(device)
        optimizer.zero_grad()
        loc_pred = model(data)
        loc_loss = F.mse_loss(loc_pred, loc_target/45.0)
        loss = loc_loss
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_one_epoch(args, model, device, test_loader):
    global BEST_ACC, MIN_LOSS
    model.eval()
    test_loss = 0
    img_cnt = 0
    correct = 0
    with torch.no_grad():
        for data, cls_target, loc_target in test_loader:
            data, cls_target, loc_target = data.to(device), cls_target.to(device), loc_target.to(device)
            loc_pred = model(data)
            img_cnt = save_pred(img_cnt, loc_pred, loc_target)
            test_loss += F.mse_loss(loc_pred*45, loc_target, size_average=False).item() # sum up batch loss
            # loc_pred = loc_pred.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += loc_pred.eq(loc.view_as(loc_pred)).sum().item()

    test_loss /= (4*len(test_loader.dataset))
    # test_acc = 100. * correct / len(test_loader.dataset)
    # BEST_ACC = max(test_acc, BEST_ACC)
    MIN_LOSS = min(MIN_LOSS,test_loss)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     test_acc))
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def save_pred(img_cnt, loc_pred, loc_label):
    pred = (loc_pred*45)
    pred_label = np.concatenate([pred,loc_label],axis=-1)
    for idx in range(pred.shape[0]):
        pred_label[idx].astype(np.float32).tofile("./loc_pred/%06d_%03d.bin" % (img_cnt+idx,EPOCH_CNT))
    img_cnt += pred.shape[0]
    return img_cnt


def main():
    global EPOCH_CNT
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = TinyYolo().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='./mnist/mnist_train/'+TRAIN_FILE_CC,
                cls_labelpath='./mnist/mnist_train/mnist_train_label',
                loc_labelpath='./mnist/mnist_train/mnist_train_cc1.0_loc'),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='./mnist/mnist_test/'+TEST_FILE_CC,
                cls_labelpath='./mnist/mnist_test/mnist_test_label',
                loc_labelpath='./mnist/mnist_test/mnist_test_cc1.0_loc'),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_one_epoch(args, model, device, test_loader)
        EPOCH_CNT += 1
    print('Best accuracy: %.2f%%' % BEST_ACC)

if __name__ == '__main__':
    main()