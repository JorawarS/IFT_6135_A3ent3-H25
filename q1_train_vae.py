from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from q1_vae import *
import os

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_function(recon_x, x, mu, logvar):
    ## TO DO: Implement the loss function using your functions from q1_solution.py
    ## use the following structure:
    # kl = kl_gaussian_gaussian_analytic(mu_q=?, logvar_q=?, mu_p=?, logvar_p=?).sum()
    # recon_loss = (?).sum()
    # return recon_loss + kl

    kl = kl_gaussian_gaussian_analytic(mu_q=mu,
                                        logvar_q=logvar, 
                                        mu_p=torch.zeros_like(mu), 
                                        logvar_p=torch.zeros_like(logvar)).sum()
    recon_loss = -log_likelihood_bernoulli(recon_x, x).sum()
    return recon_loss + kl



train_losses = []
val_losses = []    

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    avg_train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(avg_train_loss)
    print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, avg_train_loss))

def validate():
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    val_losses.append(avg_val_loss)
    print('====> Validation set loss: {:.4f}'.format(avg_val_loss))

if __name__ == "__main__":
    if not os.path.exists('vae.pt'):
        # Initialize lists to store losses
        
        for epoch in range(1, args.epochs + 1):
            train(epoch)
            validate()
        
        # Plot training and validation losses
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, args.epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, args.epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('results/loss_plot_vae.png')
        plt.show()

        # Save the model
        torch.save(model, 'vae.pt')
    else:
        print("Model already trained. Loading model...")
        model = torch.load('vae.pt',weights_only=False)
        model.eval()

        # generate random images
        with torch.no_grad():
            for i in range(2):
                z = torch.randn(128, 20).to(device)  # Generate random latent vectors
                random_images = model.decode(z).view(128, 1, 28, 28)  # Decode to generate images
                save_image(random_images, f'results/random_images_{i}.png')  # Save the generated images
            print("random images saved to results folder")

        #Test the model on the test set
        test_loss = 0
        with torch.no_grad():  # Disable gradient computation
            for data, _ in test_loader:
                data = data.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move data to the appropriate device
                recon_batch, mu, logvar = model(data)  # Forward pass
                loss = loss_function(recon_batch, data, mu, logvar)  # Compute loss
                test_loss += loss.item()

        # Compute average loss
        test_loss /= len(test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

        # Latent Traversal
        with torch.no_grad():
            # Sample a latent vector z from the standard Gaussian distribution
            z = torch.randn(1, 20).to(device)
            latent_dim = 20  # Number of dimensions in the latent space
            num_samples = 30  # Number of samples to generate for each dimension
            epsilon = 0.5 # Step size for perturbation
            
            # Create a grid to store the generated images
            fig, axes = plt.subplots(latent_dim, num_samples, figsize=(num_samples, latent_dim))
            for i in range(latent_dim):
                for j in range(num_samples):
                    # Perturb the i-th dimension of z
                    z_perturbed = z.clone()
                    z_perturbed[0, i] += epsilon * j
                    
                    # Decode the perturbed latent vector
                    generated_image = model.decode(z_perturbed).view(28, 28).cpu().numpy()
                    
                    # Plot the generated image
                    axes[i, j].imshow(generated_image, cmap='gray')
                    axes[i, j].axis('off')
        
            plt.suptitle("Latent Traversals", y=1.001)  # Adjust the y position to prevent cutoff
            plt.tight_layout()
            plt.savefig('results/latent_traversal.png', bbox_inches='tight')  # Ensure everything fits in the saved image
            plt.show()



