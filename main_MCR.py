import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model_MCR
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from loss import MaximalCodingRateReduction
import torch.optim.lr_scheduler as lr_scheduler

matplotlib.style.use('ggplot')

# %%
# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='batch size to train the VAE for')
parser.add_argument('-l', '--learning_rate', default=0.0001, type=float,
                    help='learning rate to train the VAE for')       
parser.add_argument('-beta', '--beta', default=10, type=int,
                    help='beta to train the VAE for')   
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')   
parser.add_argument('--gam1', type=float, default=1.0,
                    help='gamma1 for tuning empirical loss (default: 1.0)')
parser.add_argument('--gam2', type=float, default=10,
                    help='gamma2 for tuning empirical loss (default: 10)')
parser.add_argument('--eps', type=float, default=2,
                    help='eps squared (default: 2)')           
args = vars(parser.parse_args())
epochs = args['epochs']
batch_size = args['batch_size']
lr = args['learning_rate']
beta = args['beta']

# %%
# leanring parameters
#epochs = 0
# batch_size = 64
# lr = 0.0001
imgtrain_size = 64
# beta = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ",device)

# %%
# transforms
train_dir = '../celebA/train'
val_dir = '../celebA/val'
transform = transforms.Compose([
    transforms.Resize((imgtrain_size, imgtrain_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.24, 0.24, 0.24]),
])
# train and validation data

# train_data = datasets.MNIST(root='../input/MINST', train=True, download=True, transform=transform)
# val_data = datasets.MNIST(root='../input/MINST', train=False, download=True, transform=transform)
train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)

# training and validation data loaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model_MCR = model_MCR.VAE().to(device)
criterion = MaximalCodingRateReduction(gam1=1.0, gam2=10, eps=2)
optimizer = optim.SGD(model_MCR.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
# optimizer = optim.Adam(model_MCR.parameters(), lr=lr)
# criterion = nn.BCELoss(reduction='sum')


# %%
def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD


# %%
def fit(model_MCR, dataloader, beta=1):
    print("0")
    model_MCR.train()
    print("1")
    running_loss = 0.0
    running_x = 0.0
    running_z = 0.0
    with tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)) as t:
        for i, data in t:
        # for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
            t.close()
            data, _ = data
            data = data.to(device)
            # data = data.view(data.size(0), -1)
            optimizer.zero_grad()
            [reconstruction,z] = model_MCR(data)
            loss_x, loss_empi, loss_theo = criterion(reconstruction, data)
            reconstruction_inv,z_inv = model_MCR(reconstruction)
            loss_z, loss_empi, loss_theo = criterion(z_inv, z)
            loss=loss_x+loss_z

            # BCE, KLD = final_loss(bce_loss, mu, logvar)
            # loss = BCE + beta * KLD
            running_loss += loss.item()
            running_x += loss_x.item()
            running_z += loss_z.item()
            loss.backward()
            optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    train_x = running_x / len(dataloader.dataset)
    train_z = running_z / 16
    return train_loss, train_x, train_z


def validate(model_MCR, dataloader, beta=1):
    model_MCR.eval()
    running_loss = 0.0
    running_BCE = 0.0
    running_KLD = 0.0
    with torch.no_grad():
        with tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)) as t:
            for i, data in t:
            # for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
                t.close()
                data, _ = data
                data = data.to(device)
                # data = data.view(data.size(0), -1)
                reconstruction, mu, logvar = model_MCR(data)
                bce_loss = criterion(reconstruction, data)
                BCE, KLD = final_loss(bce_loss, mu, logvar)
                loss = BCE + beta * KLD
                running_loss += loss.item()
                running_KLD += KLD.item()
                running_BCE += BCE.item()

                # save the last batch input and output of every epoch
                if i == int(len(val_data) / dataloader.batch_size) - 1:
                    num_rows = 8
                    both = torch.cat((data.view(batch_size, 3, imgtrain_size, imgtrain_size)[:8],
                                      reconstruction.view(batch_size, 3, imgtrain_size, imgtrain_size)[:8]))
                    save_image(both.cpu(), './outputs/output'+str(epoch)+'.png', nrow=num_rows)
    val_loss = running_loss / len(dataloader.dataset)
    val_BCE = running_BCE / len(dataloader.dataset)
    val_KLD = running_KLD / len(dataloader.dataset)
    return val_loss, val_BCE, val_KLD


# %%
train_loss = []
train_BCE = []
train_KLD = []
val_loss = []
val_BCE = []
val_KLD = []
for epoch in range(epochs):
    print('Epoch '+str(epoch + 1)+' of '+str(epochs))
    train_epoch_loss, train_epoch_BCE, train_epoch_KLD = fit(model_MCR, train_loader, beta)
    val_epoch_loss, val_epoch_BCE, val_epoch_KLD = validate(model_MCR, val_loader, beta)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_BCE.append(train_epoch_BCE)
    train_KLD.append(train_epoch_KLD)
    val_BCE.append(val_epoch_BCE)
    val_KLD.append(val_epoch_KLD)
    print('Train Loss: '+str(train_epoch_loss)+', Train BCE: '+str(train_epoch_BCE)+', Train KLD: '+str(train_epoch_KLD))
    print('Val Loss: '+str(val_epoch_loss)+', Val BCE: '+str(val_epoch_BCE)+', Val KLD: '+str(val_epoch_KLD))
PATH = './beta-VAE.pth'
torch.save(model_MCR.state_dict(), PATH)

# Load model_MCR
# model_MCR.load_state_dict(torch.load(PATH))

# %%
x_axis = list(range(1, epochs + 1))
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
axes[0, 0].plot(x_axis, train_loss)
axes[0, 0].set_title('Train Loss')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Loss')

axes[0, 1].plot(x_axis, train_BCE)
axes[0, 1].set_title('Train BCE')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('BCE')

axes[0, 2].plot(x_axis, train_KLD)
axes[0, 2].set_title('Train KLD')
axes[0, 2].set_xlabel('Epochs')
axes[0, 2].set_ylabel('KLD')

axes[1, 0].plot(x_axis, val_loss)
axes[1, 0].set_title('Val Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].set_ylabel('Loss')

axes[1, 1].plot(x_axis, val_BCE)
axes[1, 1].set_title('Val BCE')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].set_ylabel('BCE')

axes[1, 2].plot(x_axis, val_KLD)
axes[1, 2].set_title('Val KLD')
axes[1, 2].set_xlabel('Epochs')
axes[1, 2].set_ylabel('KLD')
fig.tight_layout()
# plt.show()
fig.savefig('./result_loss'+str(epochs)+'.png')
