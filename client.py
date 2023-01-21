import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28), layers=2):

        super().__init__()

        self.input_shape = input_shape
        self.layers = layers
        in_channels = 3

        # y = Conv2D(filters=n1, kernel_size=(3, 3), strides=1, padding="same", activation=None)(y)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=n1, kernel_size=(3, 3), stride=1, padding="same")

        # y = BatchNormalization()(y)
        self.bn1 = nn.BatchNorm2d(n1)
        
        # y = Activation("relu")(y)
        self.activation1 = nn.ReLU()
        
        # y = Dropout(dropout_rate)(y)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        
        # y = AveragePooling2D(pool_size=(2, 2), strides=1, padding="same")(y)
        avgpool1_padding = (0, 1, 0, 1) # padding="same"
        self.pad1 = nn.ZeroPad2d(padding=avgpool1_padding) # Pad 1 pixel on the right and bottom
        self.avgpool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(1, 1), padding=avgpool1_padding)

        #####

        if layers == 2:

            # y = Conv2D(filters=n2, kernel_size=(3, 3), strides=2, padding="valid", activation=None)(y)
            self.conv2 = torch.nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(3, 3), stride=2, padding="valid")
            
            # y = BatchNormalization()(y)
            self.bn2 = nn.BatchNorm2d(n2)
        
            # y = Activation("relu")(y)
            self.activation2 = nn.ReLU()
        
            # y = Dropout(dropout_rate)(y)
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)
            
            fc_in_features = int(n2 * ((input_shape[-1] - 2) / 2) ** 2)

        else:

            # y = Conv2D(filters=n2, kernel_size=(2, 2), strides=2, padding="valid", activation=None)(y)
            self.conv2 = torch.nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=(2, 2), stride=2, padding="valid")
            
            # y = BatchNormalization()(y)
            self.bn2 = nn.BatchNorm2d(n2)
        
            # y = Activation("relu")(y)
            self.activation2 = nn.ReLU()
        
            # y = Dropout(dropout_rate)(y)
            self.dropout2 = torch.nn.Dropout(p=dropout_rate)
        
            # y = AveragePooling2D(pool_size=(2, 2), strides=2, padding="valid")(y)
            self.avgpool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2,2))

            # y = Conv2D(filters=n3, kernel_size=(3, 3), strides=2, padding="valid", activation=None)(y)
            self.conv3 = torch.nn.Conv2d(in_channels=n2, out_channels=n3, kernel_size=(3, 3), stride=2, padding="valid")
            
            # y = BatchNormalization()(y)
            self.bn3 = nn.BatchNorm2d(n3)
            
            # y = Activation("relu")(y)
            self.activation3 = nn.ReLU()
        
            # y = Dropout(dropout_rate)(y)
            self.dropout3 = torch.nn.Dropout(p=dropout_rate)
            
            # 64x64 -> 15x15
            # 32x32 -> 7x7
            
            fc_in_features = int(n3 * ((input_shape[-1] - 4) / 4) ** 2)

        #####

        #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

        # y = Flatten()(y)
        self.flatten = nn.Flatten()
        
        # y = Dense(units=n_classes, activation=None, use_bias=False,
        # TODO:          kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
        self.fc = nn.Linear(in_features=fc_in_features, out_features=n_classes, bias=False)
        
        # y = Activation("softmax")(y)
        # self.softmax = torch.nn.Softmax()

        # model_A = Model(inputs=x, outputs=y)

        # model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        #                 loss="sparse_categorical_crossentropy",
        #                 metrics=["accuracy"])
        # return model_A

    def forward(self, x):
       
        # if len(self.input_shape) == 2:
        #     # x = Reshape((input_shape[0], input_shape[1], 1))(x)
        #     x = x.reshape((self.input_shape[0], self.input_shape[1], 1))
        # else:
        #     # y = Reshape(input_shape)(x)
        #     x = x.reshape(self.input_shape)
            
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
        # x = self.softmax(x)
        return x


def cnn_2layers(n_classes, n1=128, n2=256, dropout_rate=0.2, input_shape=(28, 28)):
    return CNN(n_classes, n1, n2, None, dropout_rate, input_shape, layers=2)


def cnn_3layers(n_classes, n1=128, n2=192, n3=256, dropout_rate=0.2, input_shape=(28, 28)):
    return CNN(n_classes, n1, n2, n3, dropout_rate, input_shape, layers=3)
