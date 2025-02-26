import torch
import numpy as np
from torch.utils.data import DataLoader
from model.models import SparseVAESpikeSlab, VAE  # Import VAE for baseline comparison
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
from data.dataset import BaseDataset
from experiment.j_experiment import NonlinearICA, train_model, apply_regularization
import argparse
import os

# Argument Parser
parser = argparse.ArgumentParser(description="Compare Nonlinear ICA and Sparse VAE")
parser.add_argument("--data", type=str, default="simulated", choices=["peerread", "movielens", "simulated"], help="Dataset name")
parser.add_argument("--datafile", type=str, default="", help="Path to raw data file")
parser.add_argument("--procfile", type=str, default="", help="Path to processed data file")
parser.add_argument("--num_epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--lambda_sparsity", type=float, default=0.01, help="Regularization strength for MCP penalty")
parser.add_argument("--reg_type", type=str, default="mcp", choices=["mcp", "l1", "scad"], help="Regularization type")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--latent_dim", type=int, default=5, help="Number of latent components")
parser.add_argument("--hidden_dim", type=int, default=300, help="Hidden layer dimensions")
parser.add_argument("--lambda0", type=float, default=10.0, help="Lambda0 regularization for sparse VAE")
parser.add_argument("--lambda1", type=float, default=1.0, help="Lambda1 regularization for sparse VAE")
parser.add_argument("--outdir", type=str, default="./results", help="Output directory")
args = parser.parse_args()

# Create output directory
os.makedirs(args.outdir, exist_ok=True)

# Load dataset

dataset = BaseDataset(args.data, is_discrete_data=False, processed_data_file=args.procfile, make_data_from_scratch=True)
dataset.assign_splits(num_splits=2)
dataset.split_data(fold=0)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
input_dim = dataset.get_num_features()

# Define models
print("Initializing models...")


ica_model = NonlinearICA(input_dim, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)

# Sparse VAE Model
sparse_vae = SparseVAESpikeSlab(
    args.batch_size,
    input_dim,
    args.latent_dim,
    hidden_dim=args.hidden_dim,
    lambda0=args.lambda0,
    lambda1=args.lambda1
)

# Train models
print("Training Nonlinear ICA...")
# trainer_ica = ModelTrainer(ica_model, "NonlinearICA")
# trainer_ica.train(dataloader, epochs=args.epochs, lr=args.lr)
ica_model = train_model(args, ica_model, dataloader)


print("Training Sparse VAE...")
trainer_vae = ModelTrainer(sparse_vae, "SparseVAE")
trainer_vae.train(dataloader, epochs=args.epochs, lr=args.lr)

# Evaluate models
evaluator_ica = Evaluator(ica_model, dataset)
evaluator_vae = Evaluator(sparse_vae, dataset)

ica_nll = evaluator_ica.evaluate_heldout_nll()
vae_nll = evaluator_vae.evaluate_heldout_nll()

print(f"Nonlinear ICA Heldout NLL: {ica_nll}")
print(f"Sparse VAE Heldout NLL: {vae_nll}")

# Save results
np.save(os.path.join(args.outdir, "ica_nll.npy"), np.array([ica_nll]))
np.save(os.path.join(args.outdir, "vae_nll.npy"), np.array([vae_nll]))

# # Additional Metrics
# print("Evaluating disentanglement and sparsity...")
# dci_ica = evaluator_ica.evaluate_dci()
# dci_vae = evaluator_vae.evaluate_dci()
# print(f"Nonlinear ICA DCI Score: {dci_ica}")
# print(f"Sparse VAE DCI Score: {dci_vae}")

# # Save additional results
# np.save(os.path.join(args.outdir, "ica_dci.npy"), np.array([dci_ica]))
# np.save(os.path.join(args.outdir, "vae_dci.npy"), np.array([dci_vae]))

print("Experiment completed! Results saved.")
