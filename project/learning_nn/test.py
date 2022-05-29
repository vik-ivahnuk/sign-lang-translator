import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset_utils.dataset_sign_lang import get_train_test_loaders
from learning_nn.NeuralNetwork import AlexNet as Net


def validate(model_path):
    _, test_loader = get_train_test_loaders()
    net = Net().float().eval()
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model)
    score = n = 0.0
    for batch in test_loader:
        n += len(batch['image'])
        outputs = net(batch['image'])
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().numpy()
        label = batch['label'][:, 0].numpy()
        lh = np.argmax(outputs, axis=1)
        score += float(np.sum(lh == label))
    return score / n


if __name__ == '__main__':
    accuracy = []
    for i in range(9):
        accuracy.append(validate("../resources/model" + str(i + 1) + ".pth") * 100)
    plt.plot(accuracy)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()