# import libraries
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


torch.cuda.empty_cache()
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
  print('CUDA is not available. Training on CPU.')
else:
  print('CUDA is available. Training on GPU.')
  
  









class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        self.expansion = 4
        
        self.conv1 = nn.Conv2d( in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        self.conv2 = nn.Conv2d(intermediate_channels,intermediate_channels,kernel_size=3, stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        self.conv3 = nn.Conv2d( intermediate_channels,intermediate_channels * self.expansion,kernel_size=1,stride=1,padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)






def ResNet50(img_channel=3, num_classes=12):
    return ResNet(block, [3, 4, 6, 3], img_channel, num_classes)
def ResNet101(img_channel=3, num_classes=12):
    return ResNet(block, [3, 4, 23, 3], img_channel, num_classes)
def ResNet152(img_channel=3, num_classes=12):
    return ResNet(block, [3, 8, 36, 3], img_channel, num_classes)






















  #8 15 20 13 16 20 19 13 20 20 14 16      test size
  
# imshow function for displaying images
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


# dataset directory
data_dir = r'C:\Users\Alex\Desktop\Uni work\Year 3\AI for Creative Technologies\PythonClassificationUniModule\Dataset'

# define batch size
batch_size = 194

# define transforms
transform = transforms.Compose([transforms.Resize(224), # resize to 224x?
                                transforms.CenterCrop(224), # take a square (224x224) crop from the centre
                                transforms.ToTensor(), # convert data to torch.FloatTensor
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]) # normalise for each colour channel

# choose the test dataset
test_data = datasets.ImageFolder(data_dir + '/Test', transform=transform)

# prepare the data loader
test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False)

# for t-SNE we will use a trained network to extract features for each image
# we will remove the final layer of the network so the output a 512 feature vector

# load model

# load a pre-trained ResNet network with 18 layers
#model = models.resnet18(pretrained=True)
myModel = None


ResNetType = 2
if ResNetType == 1:
    myModel = ResNet101()
elif ResNetType == 2:
    myModel = ResNet50()
else:
    myModel = ResNet152()
# # remove the final layer so the output of the network is now a 512 feature vector
#model = nn.Sequential(*list(model.children())[:-1])

# move tensors to GPU if CUDA is available
if train_on_gpu:
    myModel.cuda()

print(myModel)


# create a new model with ResNet18 architecture
# get the number of inputs for the final layer (fc) of the network
num_ftrs = myModel.fc.in_features
# replace the final layer so that the output is four classes
myModel.fc = nn.Linear(num_ftrs, 12)
# load previously trained model
myModel.load_state_dict(torch.load(r'C:\Users\Alex\Desktop\Uni work\Year 3\AI for Creative Technologies\PythonClassificationUniModule\ResNet_0.001_128_100_2.pt'))
# remove the final layer so the output of the network is now a 512 feature vector
myModel = nn.Sequential(*list(myModel.children())[:-1])

# move tensors to GPU if CUDA is available
if train_on_gpu:
    myModel.cuda()

print(myModel)

# Visualize Sample Test Results

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

# get sample outputs
imageNet_feats = myModel(images)
jellyfish_feats = myModel(images)

# put on cpu, convert to numpy array and squeeze to batchsize x 512
imageNet_feats = np.squeeze(imageNet_feats.cpu().detach().numpy())
jellyfish_feats = np.squeeze(jellyfish_feats.cpu().detach().numpy())

labels = labels.numpy()

print(imageNet_feats.shape)
print(jellyfish_feats.shape)
print(labels)


############################################################
# Fit and transform with a TSNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=1) 

############################################################
# Project the data in 2D
jellyfish_X_2d = tsne.fit_transform(jellyfish_feats)

############################################################


# Visualize the data
plt.figure(figsize=(6, 5))
for i in range(jellyfish_X_2d.shape[0]):
  if labels[i] == 0:
    class0 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='r')
  if labels[i] == 1:
    class1 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='g')
  if labels[i] == 2:
    class2 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='b')
  if labels[i] == 3:
    class3 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='c')
  if labels[i] == 4:
    class4 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='black')
  if labels[i] == 5:
    class5 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='brown')
  if labels[i] == 6:
    class6 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='blueviolet')
  if labels[i] == 7:
    class7 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='yellow')
  if labels[i] == 8:
    class8 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='orange')
  if labels[i] == 9:
    class9 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='pink')
  if labels[i] == 10:
    class10 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='purple')
  if labels[i] == 11:
    class11 = plt.scatter(jellyfish_X_2d[i,0], jellyfish_X_2d[i, 1], c='slategray')
    
plt.title("filename Here")
plt.legend((class0, class1, class2, class3,class4,class5,class6,class7,class8,class9,class10,class11), 
           ('0: Brimstone', '1: Common Blue', '2: Holly Blue', '3: Large White', '4: Meadow Brown', '5: Orange Tip', '6: Painted Lady', '7: Peacock', '8: Red Admiral', '9: Small Copper','10: Small Toirtoiseshell', '11: Speckled Wood')) 
plt.show()
 