import torch
import torch.nn as nn
import torch.nn.functional as F
import math

features = 16 * 3
latent = 32


# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
        # MINST
        # encoder
        # self.enc1 = nn.Linear(in_features=784, out_features=512)
        # self.enc2 = nn.Linear(in_features=512, out_features=features * 2)
        #
        # # decoder
        # self.dec1 = nn.Linear(in_features=features, out_features=512)
        # self.dec2 = nn.Linear(in_features=512, out_features=784)

        # celebA
        # encoder
        self.enc1 = nn.Linear(in_features=784 * 3, out_features=512 * 3)
        self.enc2 = nn.Linear(in_features=512 * 3, out_features=features * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=features, out_features=512 * 3)
        self.dec2 = nn.Linear(in_features=512 * 3, out_features=784 * 3)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x)
        x = x.view(-1, 2, features)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        return reconstruction, mu, log_var


# define a simple linear VAE
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # celebA
        # encoder
        self.enc1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.enc2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.enc3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.enc4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.enc5 = nn.Linear(in_features=64 * 4 * 4, out_features=256)
        self.enc6 = nn.Linear(in_features=256, out_features=latent * 2)

        # decoder
        self.dec1 = nn.Linear(in_features=latent, out_features=256)
        self.dec2 = nn.Linear(in_features=256, out_features=64 * 4 * 4)
        self.dec3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.dec4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.dec5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, padding=1, stride=2)
        self.dec6 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, padding=1, stride=2)
        self.init_weights()

    def init_weights(self):
        for conv in [self.enc1, self.enc2, self.enc3, self.enc4]:
            C_in = conv.weight.size(1)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 2.5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

        # TODO (part c): initialize parameters for fully connected layers
        nn.init.normal_(self.enc5.weight, 0.0, math.sqrt(1 / 256))
        nn.init.constant_(self.enc5.bias, 0.0)
        nn.init.normal_(self.enc6.weight, 0.0, math.sqrt(1 / 32))
        nn.init.constant_(self.enc6.bias, 0.0)

        nn.init.normal_(self.dec1.weight, 0.0, math.sqrt(1 / 32))
        nn.init.constant_(self.dec1.bias, 0.0)
        nn.init.normal_(self.dec2.weight, 0.0, math.sqrt(1 / 256))
        nn.init.constant_(self.dec2.bias, 0.0)

        for conv in [self.dec3, self.dec4, self.dec5, self.dec6]:
            C_in = conv.weight.size(0)
            nn.init.normal_(conv.weight, 0.0, 1 / math.sqrt(5 * 2.5 * C_in))
            nn.init.constant_(conv.bias, 0.0)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)  # `randn_like` as we need the same size
        sample = mu + (eps * std)  # sampling as if coming from the input space
        return sample

    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        size = x.size()
        x = x.view(x.size(0), -1)
        x = F.relu(self.enc5(x))
        x = self.enc6(x)
        x = x.view(-1, 2, latent)
        # get `mu` and `log_var`
        mu = x[:, 0, :]  # the first feature values as mean
        log_var = x[:, 1, :]  # the other feature values as variance
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # decoding
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = x.view(size)
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        x = F.relu(self.dec5(x))
        reconstruction = torch.sigmoid(self.dec6(x))
        return reconstruction, mu, log_var
