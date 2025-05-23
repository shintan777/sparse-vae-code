import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import argparse
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from data.dataset import BaseDataset
from evaluation.evaluator import Evaluator
import torch.distributions as dist
import FrEIA.framework as Ff
import FrEIA.modules as Fm

def parse_args():
    parser = argparse.ArgumentParser(description="Nonlinear ICA with Structural Sparsity")
    parser.add_argument("--lambda_sparsity", type=float, default=0.1, help="Regularization strength for MCP penalty")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training")
    parser.add_argument("--data", type=str, default="simulated", help="Dataset name")
    parser.add_argument("--datafile", type=str, default="", help="Path to raw data file")
    parser.add_argument("--procfile", type=str, default="", help="Path to processed data file")
    parser.add_argument("--is_discrete", action='store_true', help="Flag for discrete data")
    parser.add_argument("--num_folds", type=int, default=2, help="Number of dataset splits")
    parser.add_argument("--split", type=int, default=0, help="Fold to run experiment")
    parser.add_argument("--reg_type", type=str, default="mcp", choices=["mcp", "l1", "scad"], help="Regularization type")
    return parser.parse_args()

class NonlinearICA(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.flow = self.build_glow(latent_dim)
        self.input_dim = input_dim
    
    def build_glow(self, latent_dim):
        def subnet_fc(c_in, c_out):
            return nn.Sequential(nn.Linear(c_in, 128), nn.ReLU(), nn.Linear(128, c_out), nn.BatchNorm1d(c_out))
        nodes = [Ff.InputNode(latent_dim, name='input')]
        for i in range(10):
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}, name=f'actnorm_{i}'))  # Normalize activations
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock, {'subnet_constructor': subnet_fc}, name=f'coupling_{i}'))
        nodes.append(Ff.OutputNode(nodes[-1], name='output'))
        return Ff.ReversibleGraphNet(nodes, verbose=False)
    
    
    def whiten_zca(self, x):
        """ Apply ZCA whitening to stabilize latent sources """
        mean = x.mean(dim=0, keepdim=True)
        cov = (x - mean).T @ (x - mean) / (x.shape[0] - 1)
        U, S, V = torch.svd(cov)
        W = U @ torch.diag(1.0 / (torch.sqrt(S) + 1e-5)) @ U.T
        return (x - mean) @ W
    

    def forward(self, x):
        sources = self.encoder(x)
        sources = (sources - sources.mean(dim=0)) / (sources.std(dim=0) + 1e-6)  # Normalize sources
        sources = self.whiten_zca(sources)
        mixed_x, _ = self.flow(sources)
        
        if mixed_x.shape[1] != self.input_dim:
            mixed_x = nn.Linear(mixed_x.shape[1], self.input_dim).to(mixed_x.device)(mixed_x)
        
        return mixed_x, sources, 0, 0
    
    def compute_jacobian(self, x):
        x.requires_grad_(True)
        sources = self.encoder(x)
        jacobian = [grad(sources[:, i], x, grad_outputs=torch.ones_like(sources[:, i]), create_graph=True, retain_graph=True)[0] for i in range(sources.shape[1])]
        return torch.stack(jacobian, dim=1)
    
    def reconstruction_loss(self, x_pred, x):
        return nn.MSELoss()(x_pred, x)
    
    def negative_log_likelihood(self, x, mixed_x):
        recon_loss = self.reconstruction_loss(mixed_x, x)
        return recon_loss
    
    def get_generator_mask(self):
        return torch.eye(self.encoder[-1].out_features, device=self.encoder[-1].weight.device)
    

def apply_regularization(jacobian, lambda_sparsity=0.1, reg_type="mcp"):
    abs_jacobian = torch.abs(jacobian)
    gamma = 0.1
    if reg_type == "mcp":
        mask = abs_jacobian <= gamma * lambda_sparsity
        return torch.where(mask, lambda_sparsity * abs_jacobian - (jacobian**2) / (2 * gamma), (gamma * lambda_sparsity**2) / 2).sum()
    elif reg_type == "l1":
        return lambda_sparsity * abs_jacobian.sum()
    elif reg_type == "scad":
        a = 3.7
        mask1 = abs_jacobian <= lambda_sparsity
        mask2 = (abs_jacobian > lambda_sparsity) & (abs_jacobian <= a * lambda_sparsity)
        scad_penalty = torch.where(mask1, lambda_sparsity * abs_jacobian, 
                                   torch.where(mask2, (a * lambda_sparsity * abs_jacobian - 0.5 * (abs_jacobian**2 + lambda_sparsity**2)) / (a - 1), 
                                               0.5 * (a + 1) * lambda_sparsity**2))
        return scad_penalty.sum()
    
def source_independence_loss(sources):
    """ Compute covariance loss to enforce independence between sources """
    batch_size, latent_dim = sources.shape
    sources = sources - sources.mean(dim=0)  # Center the sources
    cov_matrix = (sources.T @ sources) / (batch_size - 1)  # Covariance matrix
    cov_matrix -= torch.eye(latent_dim, device=sources.device)  # Remove diagonal (self-correlation)
    return torch.norm(cov_matrix, p="fro")  # Frobenius norm to penalize non-diagonal entries


def train_model(args, model, dataloader):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            observed_data = batch['data'].to(torch.float)
            optimizer.zero_grad()
            mixed_x, sources, _, _ = model(observed_data)

            # Compute Losses
            nll_loss = model.negative_log_likelihood(observed_data, mixed_x)
            jacobian = model.compute_jacobian(observed_data)
            sparsity_loss = apply_regularization(jacobian, lambda_sparsity=args.lambda_sparsity, reg_type=args.reg_type)
            independence_loss = source_independence_loss(sources)  # New term for decorrelation

            # Combine Losses
            total_loss = nll_loss + sparsity_loss + 0.1 * independence_loss  # Weight for independence loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss.item():.4f}, "
                  f"Negative Log-Likelihood: {nll_loss.item():.4f}, "
                  f"Sparsity Loss: {sparsity_loss.item():.4f}, "
                  f"Independence Loss: {independence_loss.item():.4f}")
    
    return model


if __name__ == "__main__":
    args = parse_args()
    dataset = BaseDataset(args.data, data_file=args.datafile, processed_data_file=args.procfile, is_discrete_data=args.is_discrete)
    
    # Ensure dataset has training data
    dataset.assign_splits(num_splits=args.num_folds)
    dataset.split_data(fold=args.split)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) if hasattr(dataset, 'tr_data') else None
    
    if dataloader is None:
        raise ValueError("Dataset does not contain training data. Ensure proper dataset splitting.")
    
    model = NonlinearICA(dataset.get_num_features(), latent_dim=5)
    model = train_model(args, model, dataloader)
    evaluator = Evaluator(model, dataset, is_discrete=args.is_discrete)
    heldout_nll = evaluator.evaluate_heldout_nll()
    print(f"Heldout Negative Log-Likelihood: {heldout_nll}")
