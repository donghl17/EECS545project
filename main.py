import torch
import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import model
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

matplotlib.style.use('ggplot')

# %%
# construct the argument parser and parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='number of epochs to train the VAE for')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    help='batch size to train the VAE for')
parser.add_argument('-l', '--learning_rate', default=0.0001, type=int,
                    help='learning rate to train the VAE for')       
parser.add_argument('-beta', '--beta', default=10, type=int,
                    help='beta to train the VAE for')                 
args = vars(parser.parse_args())
epochs = args['epochs']
batch_size = args['batch_size']
lr = args['learning_rate']
beta = args['beta']

# %%
# leanring parameters
# epochs = 100
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

model = model.VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')


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
def fit(model, dataloader, beta=1):
    model.train()
    running_loss = 0.0
    running_BCE = 0.0
    running_KLD = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        # data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        BCE, KLD = final_loss(bce_loss, mu, logvar)
        loss = BCE + beta * KLD
        running_loss += loss.item()
        running_KLD += KLD.item()
        running_BCE += BCE.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss / len(dataloader.dataset)
    train_KLD = running_KLD / len(dataloader.dataset)
    train_BCE = running_BCE / len(dataloader.dataset)
    return train_loss, train_BCE, train_KLD


def validate(model, dataloader, beta=1):
    model.eval()
    running_loss = 0.0
    running_BCE = 0.0
    running_KLD = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            # data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
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
                save_image(both.cpu(), f"./outputs/output{epoch}.png", nrow=num_rows)
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
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_epoch_BCE, train_epoch_KLD = fit(model, train_loader, beta)
    val_epoch_loss, val_epoch_BCE, val_epoch_KLD = validate(model, val_loader, beta)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_BCE.append(train_epoch_BCE)
    train_KLD.append(train_epoch_KLD)
    val_BCE.append(val_epoch_BCE)
    val_KLD.append(val_epoch_KLD)
    print(f"Train Loss: {train_epoch_loss:.4f}, Train BCE: {train_epoch_BCE:.4f}, Train KLD: {train_epoch_KLD:.4f}")
    print(f"Val Loss: {val_epoch_loss:.4f}, Val BCE: {val_epoch_BCE:.4f}, Val KLD: {val_epoch_KLD:.4f}")
PATH = './beta-VAE.pth'
torch.save(model.state_dict(), PATH)

# Load model
# model.load_state_dict(torch.load(PATH))

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
fig.savefig('./result_loss.png')
