import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import numpy as np
import argparse
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from data.dataset import BaseDataset
from data.process_data import load_simulated_data
from evaluation.evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Nonlinear ICA with Structural Sparsity")
    parser.add_argument("--lambda_sparsity", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    return parser.parse_args()

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

def train_model(args, model, dataloader):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            observed_data = batch[0]  # Fix to use correct indexing for TensorDataset
            optimizer.zero_grad()
            estimated_sources = model(observed_data)
            
            log_p_z = -0.5 * torch.sum(estimated_sources**2, dim=1)
            log_likelihood = torch.mean(log_p_z)
            
            sparsity_loss = mcp_penalty(compute_jacobian(model, observed_data), lambda_=args.lambda_sparsity)
            total_loss = -log_likelihood + sparsity_loss
            
            total_loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {total_loss.item():.4f}, "
                  f"Log-Likelihood: {log_likelihood.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}")
    
    return model

def compute_mcc(true, estimated):
    return np.mean([pearsonr(true[:, i].numpy(), estimated[:, i].numpy())[0] 
                    for i in range(true.shape[1])])

if __name__ == "__main__":
    args = parse_args()
    data, metadata = load_simulated_data(N=1000, sigma_true=0.5, rho=0.0)
    make_data_from_scratch = (FLAGS.data == 'simulated')
	
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    model = NonlinearTransform(data.shape[1])
    trained_model = train_model(args, model, dataloader)
    
    estimated_sources = trained_model(torch.tensor(data, dtype=torch.float)).detach()
    print(f"MCC: {compute_mcc(torch.tensor(metadata, dtype=torch.float), estimated_sources):.4f}")
    
    evaluator = Evaluator(model, dataset)
    heldout_nll = evaluator.evaluate_heldout_nll()
    print(f"Heldout Negative Log-Likelihood: {heldout_nll}")
