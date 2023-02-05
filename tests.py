from ResNet20 import resnet20
from model_trainers import *
from data_utils import *
from datasets import CustomSubset as Subset
model = resnet20(16)
model.load_state_dict(torch.load("ckpt/ResNet20_B1_initial_pri.pt", map_location=torch.device('cpu')))
train, test = load_CIFAR10(root_dir='ckpt')
test = Subset(test, [0, 52, 295, 1299])

acc = test_network(model, test, batch_size=1)
print(acc)