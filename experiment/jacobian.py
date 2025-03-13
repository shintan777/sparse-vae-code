import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from data.dataset import BaseDataset
from torch.utils.data import DataLoader
import argparse
from evaluation.evaluator import Evaluator
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=5, hidden_dim=128, beta=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z, mu, logvar  

    
    def reconstruction_loss(self, x_pred, x):
        return nn.MSELoss()(x_pred, x)
    
    def negative_log_likelihood(self, x, mixed_x):
        recon_loss = self.reconstruction_loss(mixed_x, x)
        return recon_loss
    
    def get_generator_mask(self):
    # Retrieve the last Linear layer instead of the last layer in encoder
        last_linear_layer = None
        for layer in reversed(self.encoder):
            if isinstance(layer, nn.Linear):
                last_linear_layer = layer
                break

        if last_linear_layer is None:
            raise ValueError("No Linear layer found in encoder.")

        return torch.eye(last_linear_layer.out_features, device=last_linear_layer.weight.device)


def parse_args():
    parser = argparse.ArgumentParser(description="VAE for Tabular Data")
    parser.add_argument("--lambda_sparsity", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--data", type=str, default="simulated", help="Dataset name")
    parser.add_argument("--datafile", type=str, default="", help="Path to raw data file")
    parser.add_argument("--procfile", type=str, default="", help="Path to processed data file")
    parser.add_argument("--is_discrete", action='store_true', help="Flag for discrete data")
    parser.add_argument("--num_folds", type=int, default=2, help="Number of dataset splits")
    parser.add_argument("--split", type=int, default=0, help="Fold to run experiment")
    return parser.parse_args()

def compute_generator_jacobian_optimized(model, embedding, epsilon_scale=0.001, device="cpu"):
    batch_size, latent_dim = embedding.shape
    encoding_rep = embedding.repeat(latent_dim + 1, 1).detach().clone()
    
    delta = torch.eye(latent_dim).repeat_interleave(batch_size, dim=0)
    delta = torch.cat((delta, torch.zeros(batch_size, latent_dim))).to(device)
    
    epsilon = epsilon_scale
    encoding_rep += epsilon * delta
    
    recons = model.decode(encoding_rep)
    recons = recons.view(latent_dim + 1, batch_size, -1)
    return (recons[:-1] - recons[-1]) / epsilon


def jacobian_loss_function(model, mu, logvar, device):
    jacobian = compute_generator_jacobian_optimized(model, mu, device=device)
    loss = torch.sum(torch.abs(jacobian)) / mu.shape[0]
    return loss


def jacobian_l2_loss_function(model, mu, logvar, device):
    jacobian = compute_generator_jacobian_optimized(model, mu, device=device)
    loss = torch.sum(torch.square(jacobian)) / mu.shape[0]
    return loss

def train_vae(model, dataloader, num_epochs, lr, device="cpu"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            x = batch['data'].to(device, dtype=torch.float)
            x_pred, z, mu, logvar = model(x)
            
            recon_loss = model.reconstruction_loss(x_pred, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
            jacobian_loss = jacobian_loss_function(model, mu, logvar, device)
            
            loss = recon_loss + model.beta * kl_loss + 0.1 * jacobian_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

def evaluate_metrics(model, dataset):
    evaluator = Evaluator(model, dataset, is_discrete=False)
    heldout_nll = evaluator.evaluate_heldout_nll()
    disentanglement_score = evaluator.evaluate_dci()

    # Convert dataset.tr_data to a PyTorch tensor
    data_tensor = torch.tensor(dataset.tr_data, dtype=torch.float32).to(next(model.parameters()).device)
    
    jacobian_matrix = compute_generator_jacobian_optimized(model, data_tensor)

    print(f"Heldout Negative Log-Likelihood: {heldout_nll}")
    print(f"Disentanglement Score: {disentanglement_score}")

    w_matrix = model.get_generator_mask().cpu().detach().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(w_matrix, cmap='hot', interpolation='nearest')
    plt.title("W Matrix")

    plt.subplot(1, 2, 2)
    plt.imshow(jacobian_matrix.mean(axis=1), cmap='hot', interpolation='nearest')
    plt.title("Jacobian Sparsity")
    plt.show()


# How to run this script
if __name__ == "__main__":
    args = parse_args()
    dataset = BaseDataset(args.data, data_file=args.datafile, processed_data_file=args.procfile, is_discrete_data=args.is_discrete)
    dataset.assign_splits(num_splits=args.num_folds)
    dataset.split_data(fold=args.split)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) if hasattr(dataset, 'tr_data') else None
    if dataloader is None:
        raise ValueError("Dataset does not contain training data. Ensure proper dataset splitting.")
    
    model = VAE(input_dim=dataset.get_num_features(), latent_dim=5, beta=1.0)
    model = model.to("cpu")  # Move to appropriate device if needed
    
    train_vae(model, dataloader, num_epochs=args.num_epochs, lr=args.lr, device="cpu")
    
    evaluate_metrics(model, dataset)