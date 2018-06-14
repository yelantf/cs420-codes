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

TRAIN_FILE = 'mnist_train_data'
TEST_FILE = 'mnist_test_data'
TRAIN_FILE_CC = 'mnist_train_cc1.0_data'
TEST_FILE_CC = 'mnist_test_cc1.0_data'
TRAIN_FILE_CC_CENTERED = 'mnist_train_cc1.0_crop45_data'
TEST_FILE_CC_CENTERED = 'mnist_test_cc1.0_crop45_data'

BEST_ACC = 0.0
EPOCH_CNT = 0

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Training Code')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
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
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=6)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
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


class SegNet(nn.Module):
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


def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, cls_target, seg_target) in enumerate(train_loader):
        data, cls_target, seg_target = data.to(device), cls_target.to(device), seg_target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(output.shape)
        # print(seg_target.shape)
        loss = F.nll_loss(output, seg_target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test_one_epoch(args, model, device, test_loader):
    global BEST_ACC
    img_cnt = 0
    model.eval()
    test_loss = 0
    correct = 0
    class_correct = [0 for _ in range(2)]
    class_seen = [0 for _ in range(2)]
    with torch.no_grad():
        for data, cls_target, seg_target in test_loader:
            data, cls_target, seg_target = data.to(device), cls_target.to(device), seg_target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, seg_target, size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if EPOCH_CNT%3==0:
                img_cnt = save_pred(pred.view(-1,45,45).cpu().numpy(), img_cnt)
            correct += pred.eq(seg_target.view_as(pred)).sum().item()
            tmp_target = seg_target.view_as(pred)
            for i in range(2):
                class_correct[i] += pred.eq(tmp_target)[(tmp_target==i)].sum().item()
                class_seen[i] += (tmp_target==i).sum().item()

    test_loss /= (45*45*len(test_loader.dataset))
    test_acc = 100. * correct / (45*45*len(test_loader.dataset))
    BEST_ACC = max(test_acc, BEST_ACC)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, 45*45*len(test_loader.dataset),
        test_acc))
    print('Negative class precision: {}/{} ({:.2f}%)'.format(
        class_correct[0], class_seen[0], 100. * class_correct[0] / (1e-6 + class_seen[0])))
    print('Positive class precision: {}/{} ({:.2f}%)\n'.format(
        class_correct[1], class_seen[1], 100. * class_correct[1] / (1e-6 + class_seen[1])))


def save_pred(pred, img_cnt):
    pred = (pred*255).astype(np.uint8)
    for idx in range(pred.shape[0]):
        im = Image.fromarray(pred[idx,...])
        im.save("./seg_pred/%06d_%03d.png" % (img_cnt+idx,EPOCH_CNT))
    img_cnt += pred.shape[0]
    return img_cnt


def main():
    global EPOCH_CNT
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    model = SegNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='./mnist/mnist_train/'+TRAIN_FILE,
                cls_labelpath='./mnist/mnist_train/mnist_train_label',
                seg_labelpath='./mnist/mnist_train/'+TRAIN_FILE_CC),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        myMNIST(datapath='./mnist/mnist_test/'+TEST_FILE,
                cls_labelpath='./mnist/mnist_test/mnist_test_label',
                seg_labelpath='./mnist/mnist_test/'+TEST_FILE_CC),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        test_one_epoch(args, model, device, test_loader)
        EPOCH_CNT += 1
    print('Best accuracy: %.2f%%' % BEST_ACC)

if __name__ == '__main__':
    main()