from dataset_utils.dataset_sign_lang import get_train_test_loaders
import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
from NeuralNetwork import AlexNet


class Training:

    @staticmethod
    def __train_epoch(net, criterion, optimizer, train_loader, epoch):
        current_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs = Variable(data['image'].float())
            labels = Variable(data['label'].long())
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels[:, 0])
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            if i % 200 == 0:
                print("epoch: {0} =============== loss: {1}".format(epoch, current_loss / (i + 1)))

    @staticmethod
    def start_train(epochs, model_path):
        net = AlexNet().float()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        train_loader, _ = get_train_test_loaders()
        for epoch in range(epochs):
            Training.__train_epoch(net, criterion, optimizer, train_loader, epoch)
            scheduler.step()
        torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    for i in range(12):
        Training.start_train(i, "../resources/model" + str(i + 1) + ".pth")