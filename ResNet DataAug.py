import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms




import torch.optim as optim


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

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
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


def RunAI(lr, batch_size,n_epochs, ResNetType):

    model = None

    if ResNetType == 1:
      model = ResNet101()
    elif ResNetType == 2:
      model = ResNet50()
    else :
      model = ResNet152()





    
    savedFileName = "ResNet" + " " + str(lr) + " " + str(batch_size) + " " + str(n_epochs)  +  " "  + str(ResNetType) + ".pt"


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)


    # dataset directory
    data_dir = '/content/drive/MyDrive/Dataset'


# define transforms
# training (with random flips and rotation)
    train_transform = transforms.Compose([transforms.Resize(236), # resize to 236x?
                                transforms.RandomRotation(5), # random rotation
                                transforms.CenterCrop(224), # take a square (224x224) crop from the centre
                                transforms.RandomHorizontalFlip(), # randomly flip on horizontal axis
                                transforms.ToTensor(), # convert data to torch.FloatTensor
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]) # normalise for each colour channel

# validation and testing
    transform = transforms.Compose([transforms.Resize(224), # resize to 224x?
                                transforms.CenterCrop(224), # take a square (224x224) crop from the centre
                                transforms.ToTensor(), # convert data to torch.FloatTensor
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]) # normalise for each colour channel




    # choose the training, validation and test datasets
    train_data = datasets.ImageFolder(data_dir + '/Train', transform=transform)
    val_data = datasets.ImageFolder(data_dir + '/Validate', transform=transform)
    test_data = datasets.ImageFolder(data_dir + '/Test', transform=transform)

    # prepare the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)



    # initialise tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity

    # create empty lists to store the training and validation losses
    train_losses, val_losses = [], []

    for epoch in range(n_epochs):
    # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*data.size(0)

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for data, target in val_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss 
            valid_loss += loss.item()*data.size(0)

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)

        # store the training and validation losses for later visualisation
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
          epoch+1, 
          train_loss,
          valid_loss
          ))
            torch.save(model.state_dict(), savedFileName) # save in colab
            torch.save(model.state_dict(), '/content/drive/MyDrive/ResNet50.pt') # save in google drive
            valid_loss_min = valid_loss

    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend(frameon=False)

    model.load_state_dict(torch.load(savedFileName))



    #initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(12))
    class_total = list(0. for i in range(12))

    model.eval() # prep model for evaluation

    for data, target in test_loader:
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss 
        test_loss += loss.item()*data.size(0)
        # convert output probabilities to predicted class
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss/len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    for i in range(12):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))


print("running AI")

#lr   #bathc   #epoch    //   101 = 1     50 = 2     152 =3

RunAI(0.0001,64,100,2)
