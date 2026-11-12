import time
import torch
import torch.nn.functional as F
import pandas as pd
from src.nade_model import NADE
from src.vae_model import VAE

def generate_4x4_optimizer_matrix(all_results):
    """
    Extracts the final test loss (-ELBO) for each optimizer across all datasets
    to build the required 4x4 summary matrix.
    """
    matrix = {}
    for dataset, opt_results in all_results.items():
        matrix[dataset] = {}
        for opt_name, history in opt_results.items():
            # Extract final epoch's test loss
            matrix[dataset][opt_name] = history["test_loss"][-1]
            
    df = pd.DataFrame(matrix).T
    df.index.name = "Dataset"
    return df

def train_nade(dataset_name, train_loader, test_loader, best_opt_name, device="cpu"):
    """
    Trains NADE using the best optimizer identified from the VAE runs.
    """
    print(f"\n--- Training NADE on {dataset_name} | Optimizer: {best_opt_name} ---")
    torch.manual_seed(42)
    
    epochs = 60 if dataset_name == "D4" else 20
    model = NADE().to(device)
    
    # Replicate the exact optimizer settings from the VAE phase
    if best_opt_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif best_opt_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    elif best_opt_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif best_opt_name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_time = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.exact_nll(x)
            loss.backward()
            optimizer.step()
            
    total_train_time = time.perf_counter() - start_time
    
    # Evaluate exact NLL on test set
    model.eval()
    total_nll = 0.0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            total_nll += model.exact_nll(x).item() * x.size(0)
            
    test_nll = total_nll / len(test_loader.dataset)
    
    return model, test_nll, total_train_time

def time_generation_100_samples(model, model_type, device="cpu"):
    """
    Times the generation of 100 samples. 
    VAE decodes instantly in parallel; NADE generates autoregressively feature-by-feature.
    """
    model.eval()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        if model_type == "VAE":
            # Generate 100 samples from latent dimension 16
            z = torch.randn(100, 16).to(device)
            samples = model.decode(z)
            
        elif model_type == "NADE":
            # Autoregressive generation over 784 dimensions
            samples = torch.zeros(100, 784).to(device)
            a = model.c.unsqueeze(0).expand(100, -1).clone()
            for d in range(784):
                h_d = torch.sigmoid(a)
                p_d = torch.sigmoid(F.linear(h_d, model.W[d:d+1, :], model.b[d:d+1]))
                # Sample 0 or 1 based on probability p_d
                samples[:, d] = torch.bernoulli(p_d).squeeze()
                if d < 784 - 1:
                    a = a + F.linear(samples[:, d:d+1], model.V[d:d+1, :])
                    
    generation_time = time.perf_counter() - start_time
    return samples, generation_time

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def build_comparison_table(dataset_name, vae_metrics, nade_metrics):
    """
    Constructs the VAE vs NADE comparison metrics table.
    """
    data = {
        "Metric": [
            "Test negative log-likelihood (nats / example)",
            "Number of trainable parameters",
            "Training time (total) [sec]",
            "Time to generate 100 samples [sec]"
        ],
        "VAE": [
            f"{vae_metrics['test_loss']:.4f} (-ELBO upper bound)",
            vae_metrics['params'],
            f"{vae_metrics['train_time']:.2f}",
            f"{vae_metrics['gen_time']:.4f}"
        ],
        "NADE": [
            f"{nade_metrics['test_loss']:.4f} (Exact NLL)",
            nade_metrics['params'],
            f"{nade_metrics['train_time']:.2f}",
            f"{nade_metrics['gen_time']:.4f}"
        ]
    }
    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)
    return df