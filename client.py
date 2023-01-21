import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28), layers=2):

        super().__init__()

        self.input_shape = input_shape
        self.layers = layers
        in_channels = 3
        
        ### Layer 1 ###

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=n1, kernel_size=(3, 3), stride=1, padding="same")
        self.bn1 = nn.BatchNorm2d(n1)
        self.activation1 = nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        
        avgpool1_padding = (0, 1, 0, 1) # padding="same"
        self.pad1 = nn.ZeroPad2d(padding=avgpool1_padding) # Pad 1 pixel on the right and bottom
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1))

        ### Layer 2 ###

        if layers == 2:

            self.conv2 = torch.nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(3, 3), stride=2, padding="valid")
            self.bn2 = nn.BatchNorm2d(n2)
            self.activation2 = nn.ReLU()
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)
            
            fc_in_features = int(n2 * ((input_shape[-1] - 2) / 2) ** 2)

        else:
            
            self.conv2 = torch.nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(2, 2), stride=2, padding="valid")
            self.bn2 = nn.BatchNorm2d(n2)
            self.activation2 = nn.ReLU()
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)
            self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2,2))

            ### Layer 3 ###
            
            self.conv3 = torch.nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=(3, 3), stride=2, padding="valid")
            self.bn3 = nn.BatchNorm2d(n3)
            self.activation3 = nn.ReLU()
            self.dropout3 = torch.nn.Dropout(p=dropout_rate)
            
            fc_in_features = int(n3 * ((input_shape[-1] - 8) / 8) ** 2)

        ### Fully connected layer ###
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=fc_in_features, out_features=n_classes, bias=False)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.pad1(x)
        x = self.avgpool1(x)

        x = self.conv2(x)
        x = self.bn2(x) 
        x = self.activation2(x) 
        x = self.dropout2(x)
        
        if self.layers == 3:
            x = self.avgpool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.activation3(x)
            x = self.dropout3(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


def cnn_2layers(n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28), **kwargs):
    return CNN(n_classes, n1, n2, None, dropout_rate, input_shape, layers=2)


def cnn_3layers(n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28), **kwargs):
    return CNN(n_classes, n1, n2, n3, dropout_rate, input_shape, layers=3)
