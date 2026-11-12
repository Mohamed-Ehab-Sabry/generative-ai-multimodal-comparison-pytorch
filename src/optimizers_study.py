import time
import torch
import torch.optim as optim
from src.vae_model import VAE, vae_loss_function

def get_optimizers(model):
    """
    Returns the 4 required optimizers with hyperparameters specified in the assignment.
    """
    return {
        "SGD": optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
        "RMSprop": optim.RMSprop(model.parameters(), lr=1e-3),
        "Adam": optim.Adam(model.parameters(), lr=1e-3),
        "AdamW": optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    }

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_bce, total_kld = 0.0, 0.0, 0.0
    
    for x in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, logvar = model(x)
        loss, bce, kld = vae_loss_function(recon_x, x, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        # Multiply by batch size to accumulate correctly before averaging
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_bce += bce.item() * batch_size
        total_kld += kld.item() * batch_size
        
    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, total_bce / n_samples, total_kld / n_samples

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            recon_x, mu, logvar = model(x)
            # We primarily need the overall test loss (-ELBO) for evaluation
            loss, _, _ = vae_loss_function(recon_x, x, mu, logvar)
            total_loss += loss.item() * x.size(0)
            
    return total_loss / len(dataloader.dataset)

def run_optimizer_study(dataset_name, train_loader, test_loader, device="cpu"):
    """
    Executes the training loop across all 4 optimizers for a given dataset.
    Logs train loss, test loss, BCE, KL, and epoch times.
    """
    # 60 epochs for D4 (speech), 20 epochs for D1/D2/D3
    epochs = 60 if dataset_name == "D4" else 20
    results = {}

    for opt_name in ["SGD", "RMSprop", "Adam", "AdamW"]:
        print(f"\n--- Training VAE on {dataset_name} | Optimizer: {opt_name} ---")
        
        # Fix the random seed per run for fair parameter initialization comparison
        torch.manual_seed(42) 
        
        model = VAE().to(device)
        optimizer = get_optimizers(model)[opt_name]
        
        history = {
            "train_loss": [], "test_loss": [], 
            "train_bce": [], "train_kld": [], "epoch_times": []
        }
        
        for epoch in range(epochs):
            start_time = time.perf_counter()
            
            train_loss, train_bce, train_kld = train_epoch(model, train_loader, optimizer, device)
            test_loss = evaluate(model, test_loader, device)
            
            epoch_time = time.perf_counter() - start_time
            
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_bce"].append(train_bce)
            history["train_kld"].append(train_kld)
            history["epoch_times"].append(epoch_time)
            
            # Log progress every 5 epochs or on the last epoch
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:02d}/{epochs} | Time: {epoch_time:.2f}s | "
                      f"Train Loss: {train_loss:.4f} (BCE:{train_bce:.4f}, KL:{train_kld:.4f}) | "
                      f"Test Loss: {test_loss:.4f}")
                
        results[opt_name] = history
        
    return results