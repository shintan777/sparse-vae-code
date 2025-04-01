# Import necessary modules
import torch
import numpy as np
from torch.utils.data import DataLoader
from model.models import SparseVAESpikeSlab
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
from data.dataset import BaseDataset
from experiment.jacobian import VAE, train_vae  # Import from jacobian
from experiment.j_experiment import NonlinearICA, train_model
import argparse
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming

# Argument Parser
parser = argparse.ArgumentParser(description="Compare Nonlinear ICA, Sparse VAE, and Jacobian-Regularized VAE")
parser.add_argument("--data", type=str, default="simulated", choices=["peerread", "movielens", "simulated"], help="Dataset name")
parser.add_argument("--datafile", type=str, default="", help="Path to raw data file")
parser.add_argument("--procfile", type=str, default="", help="Path to processed data file")
parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--latent_dim", type=int, default=5, help="Number of latent components")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimensions")
parser.add_argument("--lambda0", type=float, default=10.0, help="Lambda0 regularization for sparse VAE")
parser.add_argument("--lambda1", type=float, default=1.0, help="Lambda1 regularization for sparse VAE")
parser.add_argument("--outdir", type=str, default="./results", help="Output directory")
parser.add_argument("--num_folds", type=int, default=2, help="Number of data splits")
parser.add_argument("--split", type=int, default=0, help="Which split to use")
parser.add_argument("--is_discrete", action='store_true', help="Indicates if the data is discrete")
parser.add_argument("--beta", type=float, default=1.0, help="Beta-VAE parameter")
parser.add_argument("--lambda_sparsity", type=float, default=0.01, help="Regularization strength for sparsity loss")
parser.add_argument("--reg_type", type=str, default="mcp", choices=["mcp", "l1", "scad"], help="Regularization type")

args = parser.parse_args()

# Create output directory
os.makedirs(args.outdir, exist_ok=True)

# Load dataset
dataset = BaseDataset(
    args.data, 
    data_file=args.datafile, 
    processed_data_file=args.procfile, 
    is_discrete_data=args.is_discrete
)
dataset.assign_splits(num_splits=args.num_folds)
dataset.split_data(fold=args.split)

dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True) if hasattr(dataset, 'tr_data') else None
if dataloader is None:
    raise ValueError("Dataset does not contain training data. Ensure proper dataset splitting.")

input_dim = dataset.get_num_features()

# Define models
print("Initializing models...")

# Nonlinear ICA Model
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

# Jacobian-Regularized VAE Model (from experiment.jacobian)
vae_model = VAE(input_dim=dataset.get_num_features(), latent_dim=args.latent_dim, beta=args.beta)
vae_model = vae_model.to("cpu")  # Move to CPU

# Train models
print("Training Nonlinear ICA...")
ica_model = train_model(args, ica_model, dataloader)

print("Training Sparse VAE...")
trainer_sparse_vae = ModelTrainer(sparse_vae, "SparseVAE")
trainer_sparse_vae.train(dataloader, epochs=args.num_epochs, lr=args.lr)

print("Training Jacobian-Regularized VAE...")
train_vae(vae_model, dataloader, num_epochs=args.num_epochs, lr=args.lr, device="cpu")

# Evaluate models
evaluator_ica = Evaluator(ica_model, dataset)
evaluator_sparse_vae = Evaluator(sparse_vae, dataset)
evaluator_vae = Evaluator(vae_model, dataset)

ica_nll = evaluator_ica.evaluate_heldout_nll()
sparse_vae_nll = evaluator_sparse_vae.evaluate_heldout_nll()
vae_nll = evaluator_vae.evaluate_heldout_nll()

print(f"Nonlinear ICA Heldout NLL: {ica_nll}")
print(f"Sparse VAE Heldout NLL: {sparse_vae_nll}")
print(f"Jacobian-Regularized VAE Heldout NLL: {vae_nll}")

# Save results
# np.save(os.path.join(args.outdir, "ica_nll.npy"), np.array([ica_nll]))
# np.save(os.path.join(args.outdir, "sparse_vae_nll.npy"), np.array([sparse_vae_nll]))
# np.save(os.path.join(args.outdir, "vae_nll.npy"), np.array([vae_nll]))


def binarize(matrix, threshold=1e-3):
    return (np.abs(matrix) > threshold).astype(int)

def compute_decoder_jacobian_matrix(model, z_dim, input_dim):
    model.eval()
    z = torch.eye(z_dim, requires_grad=True)
    z = z.to("cpu")

    jacobian_rows = []
    for i in range(input_dim):
        x_i = model.decode(z)[:, i]  # shape: [z_dim]
        grads = torch.autograd.grad(
            outputs=x_i,
            inputs=z,
            grad_outputs=torch.ones_like(x_i),
            create_graph=False,
            retain_graph=True
        )[0]  # shape: [z_dim, z_dim]
        if grads is None:
            print(f"Failed to compute gradients for x_{i}")
            return None
        jacobian_rows.append(grads.abs().sum(dim=1).cpu().detach().numpy())  # shape: [z_dim]

    return np.stack(jacobian_rows, axis=0)  # final shape: [input_dim, latent_dim]


# Replace W_vae with the jacobian-based structure
W_vae = compute_decoder_jacobian_matrix(vae_model, args.latent_dim, input_dim)
print("Jacobian min:", np.min(W_vae))
print("Jacobian max:", np.max(W_vae))
print(W_vae)

W_sparse_vae = sparse_vae.get_generator_mask().detach().cpu().numpy()

if W_vae is None:
    raise RuntimeError("Jacobian computation failed. Check model.decode() or autograd.")



# --- Optional: Load ground truth W if available ---
W_true = getattr(dataset, "ground_truth_W", None)
if isinstance(W_true, torch.Tensor):
    W_true = W_true.detach().cpu().numpy()

# --- Binarization helper ---


# --- Binarize matrices ---
W_sparse_vae_bin = binarize(W_sparse_vae)
# W_vae_bin = binarize(W_vae)
W_vae_bin = binarize(W_vae, threshold=2.5)
W_true_bin = binarize(W_true) if W_true is not None else None

# --- Visualization ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Sparse VAE W (binary)")
plt.imshow(W_sparse_vae_bin, cmap="Greys", aspect="auto")

plt.subplot(1, 3, 2)
plt.title("Jacobian VAE W (binary)")
plt.imshow(W_vae_bin, cmap="Greys", aspect="auto")

if W_true_bin is not None:
    plt.subplot(1, 3, 3)
    plt.title("True W (binary)")
    plt.imshow(W_true_bin, cmap="Greys", aspect="auto")

plt.tight_layout()
plt.savefig(os.path.join(args.outdir, "W_sparsity_comparison.png"))
plt.show()

# --- Hamming Distance ---
def compute_hamming(W_est, W_true):
    if W_true is None:
        return None
    return hamming(W_est.flatten(), W_true.flatten()) * W_true.size

hamming_sparse = compute_hamming(W_sparse_vae_bin, W_true_bin)
hamming_vae = compute_hamming(W_vae_bin, W_true_bin)

if hamming_sparse is not None:
    print(f"Hamming Distance (Sparse VAE vs True): {hamming_sparse:.2f}")
    print(f"Hamming Distance (Jacobian VAE vs True): {hamming_vae:.2f}")
else:
    print("Ground truth W not available — skipping Hamming distance.")


print("Experiment completed! Results saved.")
