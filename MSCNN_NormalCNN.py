import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.datasets as dsets
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import time
import os.path
import numpy as np
import pickle

num_epochs = 500
batch_size = 500
learning_rate = 0.001
print_every = 1
best_accuracy = torch.FloatTensor([0])
start_epoch = 0
num_input_channel = 1

resume_weights = "sample_data/checkpointMSCNN1.pth.tar"

cuda = torch.cuda.is_available()

torch.manual_seed(1)

if cuda:
    torch.cuda.manual_seed(1)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    torchvision.transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

print("Loading the dataset")
train_set = torchvision.datasets.ImageFolder(root="ISIDigit/Train", transform=transform)
# train_set = dsets.MNIST(root='/input', train=True, download=True, transform=transform)
indices = list(range(len(train_set)))
val_split = 4000

val_idx = np.random.choice(indices, size=val_split, replace=False)
train_idx = list(set(indices) - set(val_idx))

val_sampler = SubsetRandomSampler(val_idx)
val_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=val_sampler, shuffle=False)
val_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=val_sampler, shuffle=False)

train_sampler = SubsetRandomSampler(train_idx)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_sampler,
                                           shuffle=False)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, sampler=train_sampler, shuffle=False)

test_set = torchvision.datasets.ImageFolder(root="ISIDigit/Test", transform=transform)
# test_set = dsets.MNIST(root='/input', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
test_loader2 = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
print("Dataset is loaded")

print("Saving the dataset...")
pickle.dump(train_loader, open("sample_data/train_loader.txt", 'wb'))
pickle.dump(val_loader, open("sample_data/val_loader.txt", 'wb'))
pickle.dump(test_loader, open("sample_data/test_loader.txt", 'wb'))

pickle.dump(train_loader2, open("sample_data/train_loader2.txt", 'wb'))
pickle.dump(val_loader2, open("sample_data/val_loader2.txt", 'wb'))
pickle.dump(test_loader2, open("sample_data/test_loader2.txt", 'wb'))

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))
print("Dataset is saved")


def train(model, optimizer, train_loader, loss_fun):
    average_time = 0
    total = 0
    acc = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        batch_time = time.time()
        images = Variable(images)
        labels = Variable(labels)

        if cuda:
            images, labels = images.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fun(outputs, labels)

        if cuda:
            loss.cpu()

        loss.backward()
        optimizer.step()

        batch_time = time.time() - batch_time
        average_time += batch_time

        total += labels.size(0)
        prediction = outputs.data.max(1)[1]
        correct = prediction.eq(labels.data).sum()
        acc += correct

        if (i + 1) % print_every == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f, Accuracy: %.4f, Batch time: %f'
                  % (epoch + 1,
                     num_epochs,
                     i + 1,
                     len(train_loader),
                     loss.data[0],
                     acc / total,
                     average_time / print_every))


def eval(model, test_loader):
    model.eval()

    acc = 0
    total = 0
    for i, (data, labels) in enumerate(test_loader):
        data, labels = Variable(data), Variable(labels)
        if cuda:
            data, labels = data.cuda(), labels.cuda()

        data = data.squeeze(0)
        labels = labels.squeeze(0)

        outputs = model(data)
        if cuda:
            outputs.cpu()

        total += labels.size(0)
        prediction = outputs.data.max(1)[1]
        correct = prediction.eq(labels.data).sum()
        acc += correct
    return acc / total


def save_checkpoint(state, is_best, filename="sample_data/checkpointMSCNN.pth.tar"):
    if is_best:
        print("=> Saving a new best")
        torch.save(state, filename)
    else:
        print("=> Validation Accuracy did not improve")


class Skip_Model(nn.Module):
    def __init__(self):
        super(Skip_Model, self).__init__()

        self.layer11 = nn.Sequential(
            nn.Conv2d(num_input_channel, 32, kernel_size=3, stride=2, padding=(1, 1)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())

        self.layer12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer13 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=7, stride=1, padding=(2, 2)),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        # concatenated features fc layer

        self.fc0 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.ReLU())

        # final fc layer

        self.fc_final = nn.Sequential(
            nn.Linear(2816, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU())

        '''self.fc_soft = nn.Sequential(
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU())'''

    def forward(self, x):
        x0 = x.view(-1, self.num_flat_features(x))

        # for first column
        x3 = self.layer11(x)

        x3 = self.layer12(x3)

        x3 = self.layer13(x3)
        x13 = x3.view(-1, self.num_flat_features(x3))

        xz0 = self.fc0(x0)

        # all features concatenation

        xz = torch.cat((xz0, x13), 1)

        # final fc layer

        out = self.fc_final(xz)
        # out = self.fc_soft(out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Skip_Model()
if cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
if cuda:
    criterion.cuda()

total_step = len(train_loader)

if os.path.isfile(resume_weights):
    print("=> loading checkpoint '{}' ...".format(resume_weights))
    if cuda:
        checkpoint = torch.load(resume_weights)
    else:
        checkpoint = torch.load(resume_weights, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))

for epoch in range(num_epochs):
    print(learning_rate)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    if learning_rate >= 0.0003:
        learning_rate = learning_rate * 0.993

    train(model, optimizer, train_loader, criterion)
    acc = eval(model, val_loader)
    print('=> Validation set: Accuracy: {:.2f}%'.format(acc * 100))
    acc = torch.FloatTensor([acc])

    is_best = bool(acc.numpy() > best_accuracy.numpy())

    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))

    save_checkpoint({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, is_best)

    test_acc = eval(model, test_loader)
    print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

test_acc = eval(model, test_loader)
print('=> Test set: Accuracy: {:.2f}%'.format(test_acc * 100))

