import torch

from learning_nn.NeuralNetwork import AlexNet as Net

if __name__ == '__main__':
    net = Net().float().eval()
    pretrained_model = torch.load("../resources/model6.pth")
    net.load_state_dict(pretrained_model)
    path = "../application/model.onnx"
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(net, dummy, path, input_names=['input'])