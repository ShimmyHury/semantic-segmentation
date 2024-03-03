import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding='same')
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.batchnorm1 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding='same')
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, numberClasses=1, dropout=0.5):
        super(Unet, self).__init__()

        self.CB1 = ConvBlock(3, 64)
        self.CB2 = ConvBlock(64, 128)
        self.CB3 = ConvBlock(128, 256)
        self.CB4 = ConvBlock(256, 512)

        self.CB5 = ConvBlock(512, 1024)

        self.CB6 = ConvBlock(1024, 512)
        self.CB7 = ConvBlock(512, 256)
        self.CB8 = ConvBlock(256, 128)
        self.CB9 = ConvBlock(128, 64)

        self.pool = nn.MaxPool2d(2)

        self.dropout = nn.Dropout2d(dropout)

        self.UC1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.UC2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.UC3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.UC4 = nn.ConvTranspose2d(128, 64, 2, 2)

        self.head = nn.Conv2d(64, numberClasses, 1)

    def forward(self, x):
        # DOWN SAMPLING
        C1 = self.CB1(x)
        x = C1.clone()
        x = self.pool(x)
        x = self.dropout(x)

        C2 = self.CB2(x)
        x = C2.clone()
        x = self.pool(x)
        x = self.dropout(x)

        C3 = self.CB3(x)
        x = C3.clone()
        x = self.pool(x)
        x = self.dropout(x)

        C4 = self.CB4(x)
        x = C4.clone()
        x = self.pool(x)
        x = self.dropout(x)

        x = self.CB5(x)

        # UP SAMPLING
        x = self.UC1(x)

        x = torch.cat((x, C4), dim=1)
        x = self.CB6(x)
        x = self.dropout(x)

        x = self.UC2(x)
        x = torch.cat((x, C3), dim=1)
        x = self.CB7(x)
        x = self.dropout(x)

        x = self.UC3(x)
        x = torch.cat((x, C2), dim=1)
        x = self.CB8(x)
        x = self.dropout(x)

        x = self.UC4(x)
        x = torch.cat((x, C1), dim=1)
        x = self.CB9(x)
        x = self.head(x)
        return x
