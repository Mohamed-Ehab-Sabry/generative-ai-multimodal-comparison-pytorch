import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=16):
        super(VAE, self).__init__()
        
        # Encoder: 784 -> 400 -> 200
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc2 = nn.Linear(400, 200)
        
        # Latent space: 200 -> mu and logvar
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Decoder: 16 -> 200 -> 400 -> 784
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 400)
        self.fc5 = nn.Linear(400, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc_mu(h2), self.fc_logvar(h2)

    def reparameterize(self, mu, logvar):
        # std is sigma; computed from logvar for numerical stability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = F.relu(self.fc4(h3))
        # Sigmoid activation to output probabilities / binary values [0, 1]
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # BCE summed over the 784 dimensions (dim=1), then averaged over the batch
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=1).mean()
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    loss = BCE + KLD
    return loss, BCE, KLD