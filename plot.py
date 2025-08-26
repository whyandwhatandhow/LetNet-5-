import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(28),
    transforms.Normalize((0.5,), (0.5,))
])

trainSet = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainLoader = DataLoader(trainSet, shuffle=True, batch_size=64)
testSet=torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
test_Loader=DataLoader(testSet,shuffle=True,batch_size=1)
for step, (b_x, b_y) in enumerate(trainLoader):
    if step > 0:
        break
    class_label = trainSet.classes
    print(class_label)

writer=SummaryWriter("./logs")
img_grid=torchvision.utils.make_grid(b_x[:16],nrow=4,normalize=True)
writer.add_image("MNIST",img_grid,0)

writer.close()