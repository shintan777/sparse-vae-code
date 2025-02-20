import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import argparse
from scipy.stats import pearsonr

def parse_args():
    parser = argparse.ArgumentParser(description="Nonlinear ICA with Structural Sparsity")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of data samples")
    parser.add_argument("--num_sources", type=int, default=2, help="Number of hidden sources")
    parser.add_argument("--num_observed", type=int, default=5, help="Number of observed variables")
    parser.add_argument("--lambda_sparsity", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of training epochs")
    return parser.parse_args()

def generate_synthetic_data(args):
    variances = torch.FloatTensor(args.num_sources).uniform_(0.5, 3)
    sources = torch.randn(args.num_samples, args.num_sources) * torch.sqrt(variances)
    
    mixing_matrix = torch.randn(args.num_sources, args.num_observed)
    observed_data = sources @ mixing_matrix + 0.1 * torch.randn(args.num_samples, args.num_observed)
    
    return sources, observed_data, mixing_matrix

class NonlinearTransform(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        return self.model(x)

def compute_jacobian(model, x):
    x.requires_grad_(True)
    y = model(x)
    jacobian = [grad(y[:, i], x, grad_outputs=torch.ones_like(y[:, i]), create_graph=True, retain_graph=True)[0] 
                for i in range(y.shape[1])]
    return torch.stack(jacobian, dim=1)

def mcp_penalty(x, gamma=1.0, lambda_=1.0):
    abs_x = torch.abs(x)
    mask = abs_x <= gamma * lambda_
    return torch.where(mask, lambda_ * abs_x - (x**2)/(2*gamma), (gamma * lambda_**2)/2).sum()

def train_model(args, model, observed_data, true_sources):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.num_epochs):
        optimizer.zero_grad()
        estimated_sources = model(observed_data)
        
        log_p_z = -0.5 * torch.sum(estimated_sources**2, dim=1)
        log_likelihood = torch.mean(log_p_z)
        
        sparsity_loss = mcp_penalty(compute_jacobian(model, observed_data), lambda_=args.lambda_sparsity)
        total_loss = -log_likelihood + sparsity_loss
        
        total_loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss.item():.4f}, "
                  f"Log-Likelihood: {log_likelihood.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
    
    return model

def compute_mcc(true, estimated):
    return np.mean([pearsonr(true[:, i].numpy(), estimated[:, i].numpy())[0] 
                    for i in range(true.shape[1])])

if __name__ == "__main__":
    args = parse_args()
    true_sources, observed_data, _ = generate_synthetic_data(args)
    model = NonlinearTransform(observed_data.shape[1])
    trained_model = train_model(args, model, observed_data, true_sources)
    estimated_sources = trained_model(observed_data).detach()
    print(f"MCC: {compute_mcc(true_sources, estimated_sources):.4f}")
