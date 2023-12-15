import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        # Define your generator architecture
        self.fc = nn.Linear(input_size, 256)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_size, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  #generate images in the range [-1, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 1, 1)  #Reshape for convolutional layers
        x = self.conv_transpose(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        #Define your discriminator architecture
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  #Flatten for fully connected layer
        x = self.fc(x)
        return x
