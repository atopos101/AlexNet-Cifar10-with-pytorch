# import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,64,3,1,padding=1)
        self.conv2=nn.Conv2d(64,128,3,1,padding=1)
        self.conv3=nn.Conv2d(128,256,3,1,padding=1)
        self.conv4=nn.Conv2d(256,256,3,1,padding=1)
        self.conv5=nn.Conv2d(256,128,3,1,padding=1)

        self.fc1=nn.Linear(128*2*2,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,10)

        self.dropout1=nn.Dropout(0.5)
        self.dropout2=nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2,2)  # nn.AvgPool2d
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2,2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model,device,train_loader,optimizer,epoch,train_losses):
    model.train()
    losses=0
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        losses+=loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if args.dry_run:
                break
    train_losses.append(losses / len(train_loader))


def test(model,device,test_loader,test_accs):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.argmax(dim=1,keepdim=True)
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    test_accs.append(100. * correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# def visualize_trainloss(train_losses):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training Loss Curve')
#     plt.legend()
#     plt.show()
#
#
# def visualize_testacc(test_accs):
#     plt.figure(figsize=(10, 5))
#     plt.plot(test_accs, label='Test Accuracy', marker='o')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
#     plt.title('Test Accuracy Curve')
#     plt.legend()
#     plt.show()
#
#
# def visualize_samples(dataset, num_samples=10):`
#     """
#     可视化数据集中的样本图片及其标签
#
#     :param dataset: 数据集对象
#     :param num_samples: 要可视化的样本数量
#     """
#     fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
#     for i in range(num_samples):
#         img, label = dataset[i]
#         img = img.squeeze()  # 移除单通道维度
#         axes[i].imshow(img, cmap='gray')
#         axes[i].set_title(f'Label: {label}')
#         axes[i].axis('off')
#     plt.tight_layout()
#     plt.show()


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    # 数据预处理
    transfrom=transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集
    train_set=datasets.CIFAR10('./data',train=True,download=True,transform=transfrom)
    test_set=datasets.CIFAR10('./data',train=False,download=True,transform=transfrom)

    train_loader=DataLoader(train_set,batch_size=128,shuffle=True)
    test_loader=DataLoader(test_set,batch_size=512,shuffle=True)

    model=Net().to(device)
    optimizer=optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

    # visualize_samples(test_set)

    # 曲线
    train_losses = []
    test_accs = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch,train_losses)
        test(model, device, test_loader,test_accs)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    print(model)

    # visualize_trainloss(train_losses)
    # visualize_testacc(test_accs)


if __name__=='__main__':
    main()