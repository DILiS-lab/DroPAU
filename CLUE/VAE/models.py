from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from ..src.layers import (
    SkipConnection,
    ResBlock,
    MLPBlock,
    leaky_MLPBlock,
    preact_leaky_MLPBlock,
)


# MLP based model


class MLP_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_recognition_net, self).__init__()
        # input layer
        proposal_layers = [
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            proposal_layers.append(MLPBlock(width))
        # output layer
        proposal_layers.append(nn.Linear(width, latent_dim * 2))

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_generator_net, self).__init__()
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                leaky_MLPBlock(width)
            )
        # output layer
        generative_layers.extend(
            [
                nn.Linear(width, input_dim),
            ]
        )
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


# MLP fully linear residual path preact models


class MLP_preact_recognition_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_recognition_net, self).__init__()
        # input layer
        proposal_layers = [nn.Linear(input_dim, width)]
        # body
        for i in range(depth - 1):
            proposal_layers.append(preact_leaky_MLPBlock(width))
        # output layer
        proposal_layers.extend(
            [
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=width),
                nn.Linear(width, latent_dim * 2),
            ]
        )

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class MLP_preact_generator_net(nn.Module):
    def __init__(self, input_dim, width, depth, latent_dim):
        super(MLP_preact_generator_net, self).__init__()
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, width),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=width),
        ]
        # body
        for i in range(depth - 1):
            generative_layers.append(
                # skip-connection from prior network to generative network
                preact_leaky_MLPBlock(width)
            )
        # output layer
        generative_layers.extend(
            [
                nn.Linear(width, input_dim),
            ]
        )
        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)




class MNISTplus_recognition_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(MNISTplus_recognition_resnet, self).__init__()

        width_mul = 3

        proposal_layers = [
            # First layer: Doesn't downsample, just increases channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1),  # 64x64 -> 64x64
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            
            # Second layer: Downsample to 32x32
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 64x64 -> 32x32
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            
            # Third layer: Downsample to 16x16
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 32x32 -> 16x16
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            
            # Fourth layer: Downsample to 8x8
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2),  # 16x16 -> 8x8
            nn.LeakyReLU(inplace=True),
            
            # Flatten and connect to latent space
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=8 * 8 * 128),
            nn.Linear(8 * 8 * 128, latent_dim * 2),  # Adjust for new spatial size
            SkipConnection(
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=latent_dim * 2),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            ),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)



class MNISTplus_generator_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(MNISTplus_generator_resnet, self).__init__()

        width_mul = 3

        generative_layers = [
            leaky_MLPBlock(latent_dim),
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            small_MNIST_unFlatten(),  # 4x4 --128

            # Upsampling to 8x8
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, padding=1, stride=2
            ),  # 4x4 -> 8x8
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),

            # Upsampling to 16x16
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, stride=2
            ),  # 8x8 -> 16x16
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),

            # Upsampling to 32x32
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, stride=2
            ),  # 16x16 -> 32x32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),

            # Upsampling to 64x64
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, padding=1, stride=2
            ),  # 32x32 -> 64x64
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1),  # 64x64 -> 64x64 -- 1 channel
        ]

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)

class smallest_MNISTplus_recognition_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(smallest_MNISTplus_recognition_resnet, self).__init__()

        width_mul = 3

        proposal_layers = [
            # First layer: Doesn't downsample, just increases channels
            nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=1),  # 64x64 -> 64x64
            ResBlock(outer_dim=4, inner_dim=8 * width_mul),
            ResBlock(outer_dim=4, inner_dim=8 * width_mul),
            
            # Second layer: Downsample to 32x32
            nn.Conv2d(4, 4, kernel_size=4, padding=1, stride=2),  # 64x64 -> 32x32
            ResBlock(outer_dim=4, inner_dim=16 * width_mul),
            ResBlock(outer_dim=4, inner_dim=16 * width_mul),
            
            # Third layer: Downsample to 16x16
            nn.Conv2d(4, 4, kernel_size=4, padding=1, stride=2),  # 32x32 -> 16x16
            ResBlock(outer_dim=4, inner_dim=32 * width_mul),
            ResBlock(outer_dim=4, inner_dim=32 * width_mul),
            
            # Fourth layer: Downsample to 8x8
            nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=2),  # 16x16 -> 8x8
            nn.LeakyReLU(inplace=True),
            
            # Flatten and connect to latent space
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=8 * 8 * 4),
            nn.Linear(8 * 8 * 128, latent_dim * 2),  # Adjust for new spatial size
            SkipConnection(
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=latent_dim * 2),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            ),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        print("Input shape:", x.shape)  # Print input shape
        x = self.block[0](x)  # First conv layer
        print("After first conv:", x.shape)
        x = self.block[1](x)  # First res block
        print("After first res block:", x.shape)
        x = self.block[2](x)  # Second res block
        print("After second res block:", x.shape)
        x = self.block[3](x)  # Second conv layer
        print("After second conv:", x.shape)
        x = self.block[4](x)  # Third res block
        print("After third res block:", x.shape)
        x = self.block[5](x)  # Fourth res block
        print("After fourth res block:", x.shape)
        x = self.block[6](x)  # Third conv layer
        print("After third conv:", x.shape)
        x = self.block[7](x)  # Flatten
        print("After flatten:", x.shape)
        x = self.block[8](x)  # BatchNorm
        print("After batchnorm:", x.shape)
        x = self.block[9](x)  # First linear layer
        print("After first linear:", x.shape)
        x = self.block[10](x)  # Skip connection
        print("After skip connection:", x.shape)
        return x
    




class smallest_MNISTplus_generator_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(smallest_MNISTplus_generator_resnet, self).__init__()

        width_mul = 3

        generative_layers = [
            leaky_MLPBlock(latent_dim),
            nn.Linear(latent_dim, 4 * 4 * 4),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=4 * 4 * 4),
            small_MNIST_unFlatten(),  # 4x4 --128

            # Upsampling to 8x8
            nn.ConvTranspose2d(
                4, 128, kernel_size=4, padding=1, stride=2
            ),  # 4x4 -> 8x8
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),

            # Upsampling to 16x16
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, stride=2
            ),  # 8x8 -> 16x16
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),

            # Upsampling to 32x32
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, stride=2
            ),  # 16x16 -> 32x32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),

            # Upsampling to 64x64
            nn.ConvTranspose2d(
                32, 16, kernel_size=4, padding=1, stride=2
            ),  # 32x32 -> 64x64
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1, stride=1),  # 64x64 -> 64x64 -- 1 channel
        ]

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


# Small MNIST conv model
class small_MNIST_Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class small_MNIST_unFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 256, 8, 8)

class small_MNISTplus_recognition_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNISTplus_recognition_net, self).__init__()
        self.Nfilt1 = 64
        self.Nfilt2 = 128
        self.Nfilt3 = 256

        self.block = nn.Sequential(
            nn.Conv2d(1, self.Nfilt1, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt1),
            nn.Conv2d(self.Nfilt1, self.Nfilt2, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt2),
            nn.Conv2d(self.Nfilt2, self.Nfilt3, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=8 * 8 * self.Nfilt3),  # Flattened size
            nn.Linear(8 * 8 * self.Nfilt3, latent_dim * 2),  # Latent space output
        )

    def forward(self, x):
        return self.block(x)

class small_MNISTplus_generator_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNISTplus_generator_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 64

        self.block = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * self.Nfilt1),  # Latent space to 8x8x256
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=8 * 8 * self.Nfilt1),
            small_MNIST_unFlatten(),
            nn.ConvTranspose2d(
                self.Nfilt1, self.Nfilt2, kernel_size=4, stride=2, padding=1
            ),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt2),
            nn.ConvTranspose2d(
                self.Nfilt2, self.Nfilt3, kernel_size=4, stride=2, padding=1
            ),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt3),
            nn.ConvTranspose2d(self.Nfilt3, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid(),  # Ensure output is in range [0, 1]
        )

    def forward(self, x):
        print("Input shape:", x.shape)  # Print input shape
        x = self.block[0](x)  # First linear layer
        print("After linear:", x.shape)
        x = self.block[3](x)  # Apply small_MNIST_unFlatten
        print("After unFlatten:", x.shape)
        x = self.block[4](x)  # First transposed convolution
        print("After first conv:", x.shape)
        x = self.block[7](x)  # Second transposed convolution
        print("After second conv:", x.shape)
        x = self.block[10](x)  # Third transposed convolution
        print("After third conv:", x.shape)
        return x
    
class small_MNISTplus_generator_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNISTplus_generator_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 64

        self.block = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * self.Nfilt1),  # Latent space to 8x8x256
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=8 * 8 * self.Nfilt1),
            small_MNIST_unFlatten(),
            nn.ConvTranspose2d(
                self.Nfilt1, self.Nfilt2, kernel_size=4, stride=2, padding=1
            ),  # 8x8 -> 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt2),
            nn.ConvTranspose2d(
                self.Nfilt2, self.Nfilt3, kernel_size=4, stride=2, padding=1
            ),  # 16x16 -> 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt3),
            nn.ConvTranspose2d(self.Nfilt3, 1, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid(),  # Ensure output is in range [0, 1]
        )

    def forward(self, x):
        return self.block(x)



class small_MNIST_Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class small_MNIST_unFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 128, 4, 4)


class small_MNIST_recognition_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNIST_recognition_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 128

        proposal_layers = [
            nn.Conv2d(1, self.Nfilt1, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt1),
            nn.Conv2d(self.Nfilt1, self.Nfilt2, kernel_size=4, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt2),
            nn.Conv2d(self.Nfilt2, self.Nfilt3, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=4 * 4 * self.Nfilt3),
            nn.Linear(4 * 4 * self.Nfilt3, latent_dim * 2),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class small_MNIST_generator_net(nn.Module):
    def __init__(self, latent_dim):
        super(small_MNIST_generator_net, self).__init__()
        self.Nfilt1 = 256
        self.Nfilt2 = 128
        self.Nfilt3 = 128
        # input layer
        generative_layers = [
            nn.Linear(latent_dim, 4 * 4 * self.Nfilt3),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=4 * 4 * self.Nfilt3),
            small_MNIST_unFlatten(),
            nn.ConvTranspose2d(
                self.Nfilt3, self.Nfilt2, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt3),
            nn.ConvTranspose2d(
                self.Nfilt2, self.Nfilt1, kernel_size=4, padding=1, stride=2
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.Nfilt1),
            nn.ConvTranspose2d(self.Nfilt1, 1, kernel_size=4, padding=1, stride=2),
        ]

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)


######################################
# Models for Doodle at 64x64
######################################


class doodle_recognition_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(doodle_recognition_resnet, self).__init__()

        width_mul = 3

        proposal_layers = [
            nn.Conv2d(1, 32, kernel_size=1, padding=0, stride=1),  # 64x64 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),  # 32x32 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2),  # 16x16 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),  # 8x8 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.Conv2d(128, 128, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),  # 4x4 --128
            small_MNIST_Flatten(),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            nn.Linear(4 * 4 * 128, latent_dim * 2),
            SkipConnection(
                nn.LeakyReLU(),
                nn.BatchNorm1d(num_features=latent_dim * 2),
                nn.Linear(latent_dim * 2, latent_dim * 2),
            ),
        ]

        self.block = nn.Sequential(*proposal_layers)

    def forward(self, x):
        return self.block(x)


class doodle_generator_resnet(nn.Module):
    def __init__(self, latent_dim):
        super(doodle_generator_resnet, self).__init__()

        width_mul = 3

        generative_layers = [
            leaky_MLPBlock(latent_dim),
            nn.Linear(latent_dim, 4 * 4 * 128),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm1d(num_features=4 * 4 * 128),
            small_MNIST_unFlatten(),  # 4x4 --128
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, padding=1, stride=2
            ),  # 8x8 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.ConvTranspose2d(
                128, 128, kernel_size=4, padding=1, stride=2
            ),  # 16x16 --128
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            ResBlock(outer_dim=128, inner_dim=32 * width_mul),
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, padding=1, stride=2
            ),  # 32x32 --64
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            ResBlock(outer_dim=64, inner_dim=16 * width_mul),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, padding=1, stride=2
            ),  # 64x64 --32
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            ResBlock(outer_dim=32, inner_dim=8 * width_mul),
            nn.Conv2d(32, 1, kernel_size=1, padding=0, stride=1),
        ]  # 64x64 --1

        self.block = nn.Sequential(*generative_layers)

    def forward(self, x):
        return self.block(x)  # ResNet MNIST conv model
