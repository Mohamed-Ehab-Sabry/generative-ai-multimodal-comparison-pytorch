import torch
import matplotlib.pyplot as plt
import numpy as np

def display_8_original_samples(dataset_name, samples, vocab=None, original_texts=None):
    """
    Displays 8 original samples from the dataset.
    Images/Speech (D1, D2, D4): reshaped to 28x28 and plotted in a 2x4 grid.
    Text (D3): prints the original headline (if provided) and the active words.
    """
    print(f"\n--- Original Samples: {dataset_name} ---")
    
    if dataset_name in ["D1", "D2", "D4"]:
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle(f"8 Original Samples - {dataset_name}")
        for i, ax in enumerate(axes.flatten()):
            if i < len(samples):
                img = samples[i].view(28, 28).cpu().numpy()
                ax.imshow(img, cmap='gray')
                ax.axis('off')
        plt.tight_layout()
        plt.show()
        
    elif dataset_name == "D3":
        assert vocab is not None, "Vocabulary list required to map text indices."
        for i in range(min(8, len(samples))):
            active_indices = torch.where(samples[i] > 0.5)[0].tolist()
            active_words = [vocab[idx] for idx in active_indices]
            
            print(f"\nSample {i+1}:")
            if original_texts and i < len(original_texts):
                print(f"Original: {original_texts[i]}")
            print(f"Active words: {', '.join(active_words)}")

def plot_test_loss_curves(dataset_name, results_dict):
    """
    Plots the 4 test-loss curves (for SGD, RMSprop, Adam, AdamW) per dataset.
    """
    plt.figure(figsize=(8, 5))
    for opt_name, history in results_dict.items():
        epochs = range(1, len(history["test_loss"]) + 1)
        plt.plot(epochs, history["test_loss"], label=opt_name, linewidth=2)
        
    plt.title(f"Test Loss (-ELBO) over Epochs - {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss (nats)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def render_generated_samples(dataset_name, vae_samples, nade_samples, vocab=None):
    """
    Renders 16 VAE vs. 16 NADE generated samples.
    Images/Speech: Plotted as side-by-side 4x4 grids.
    Text: Prints the active words for the 16 generations.
    """
    print(f"\n--- Generated Samples (VAE vs NADE): {dataset_name} ---")
    
    if dataset_name in ["D1", "D2", "D4"]:
        fig, axes = plt.subplots(4, 8, figsize=(16, 8))
        fig.suptitle(f"Generated Samples - {dataset_name} (Left 4 cols: VAE | Right 4 cols: NADE)")
        
        for i in range(16):
            row = i // 4
            col_vae = i % 4
            col_nade = (i % 4) + 4
            
            # VAE side
            img_vae = vae_samples[i].view(28, 28).cpu().numpy()
            axes[row, col_vae].imshow(img_vae, cmap='gray')
            axes[row, col_vae].axis('off')
            
            # NADE side
            img_nade = nade_samples[i].view(28, 28).cpu().numpy()
            axes[row, col_nade].imshow(img_nade, cmap='gray')
            axes[row, col_nade].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    elif dataset_name == "D3":
        assert vocab is not None, "Vocabulary list required to map text indices."
        print("\n--- VAE Generations ---")
        for i in range(16):
            active_indices = torch.where(vae_samples[i] > 0.5)[0].tolist()
            active_words = [vocab[idx] for idx in active_indices]
            print(f"VAE {i+1}: {', '.join(active_words)}")
            
        print("\n--- NADE Generations ---")
        for i in range(16):
            active_indices = torch.where(nade_samples[i] > 0.5)[0].tolist()
            active_words = [vocab[idx] for idx in active_indices]
            print(f"NADE {i+1}: {', '.join(active_words)}")