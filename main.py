import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime, timedelta
from time import time_ns

import matplotlib.pyplot as plt
import numpy as np
import torch
from alpaca.data.timeframe import TimeFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from ds.bars import Conversion
from ds.config import Config
from ds.data import PreTokenizedCurveDataset
from ds.discretization import Discretizer
from ds.fit import fit, print_labels
from ds.generate import save_dataset_from_results
from ds.labeler import Labeler
from ds.models.model import InformerStack
from ds.retrieval import Security
from ds.thresholds import Threshold
from ds.visualization import Visualizer


def main() -> None:
    # Configure a 7-day, 1-minute retrieval window with all parameters
    config = Config(
        start=datetime.now() - timedelta(days=365),
        step=TimeFrame.Minute,
        plot_enabled=True,
        plot_save=True,
        plot_show=False,
        conversion=Conversion.DOLLAR,
        threshold_des_scaling_factor=0.1,
        max_time_series_len=390, # Approx one trading day
        threshold_strategy=Threshold.EMA,
        sg_window_length=50,
        sg_polyorder=2,
    )

    # Instantiate and run the offline pipeline
    symbols: list[str] = [
        "META",
        "AAPL",
        "GOOG",
        # "GOOGL",
        "NVDA",
        "TSM",
        "ORCL",
        "PLTR",
        "XOM",
        "RY",
        "DB",
        "JPM",
    ]

    print("Loading dataset...")
    try:
        dataset_data = torch.load("dataset.pt")
    except FileNotFoundError:
        start = time_ns()
        results = fit(symbols, Security.EQUITIES, config)
        end = time_ns()

        # print_labels(results)
    
        # Generate dataset
        save_dataset_from_results(results, config, "dataset.pt")

        print(f"Fitting completed in {end - start} nanoseconds")

    dataset_data = torch.load("dataset.pt")

    # --- Training Loop ---

    # Hyperparameters
    seq_len = 4
    label_len = 4
    pred_len = 4
    batch_size = 512
    d_model = 32
    learning_rate = 1e-4
    epochs = 5_000
    plot_step = max(1, epochs // 10)
    patience = 5_000

    # Initialize Discretizer to get vocab size
    discretizer = Discretizer(
        min_power=config.discretization_log_min_power,
        max_power=config.discretization_log_max_power,
        points_per_magnitude=config.discretization_points_per_magnitude
    )
    vocab_size = discretizer.vocab_size
    num_types = len(Discretizer.CURVE_TYPES)
    
    # Output size: 5 tokens * max(vocab_size, num_types)
    # We will use a single vocab size that covers both types and params for simplicity in the output layer
    # and then mask/ignore invalid indices during loss calculation if needed.
    # Or better: just use the larger vocab size (params) for all outputs, as types are few.
    output_vocab_size = max(vocab_size, num_types)
    c_out = 5 * output_vocab_size

    # Split data into train and validation (80/20)
    train_data = []
    val_data = []
    split_ratio = 0.8
    
    for item in dataset_data:
        tokens = item['tokens']
        split_idx = int(len(tokens) * split_ratio)
        
        # Ensure we have enough data for at least one sequence in both sets
        min_len = seq_len + pred_len + label_len + 1
        print(f"Symbol: {item['symbol']}, Tokens: {len(tokens)}, SplitIdx: {split_idx}, MinLen: {min_len}")
        if split_idx < min_len or (len(tokens) - split_idx) < min_len:
            print(f"Warning: Not enough data to split symbol {item['symbol']}. Using all for training.")
            train_data.append(item)
            continue
            
        train_tokens = tokens[:split_idx]
        val_tokens = tokens[split_idx:]
        
        train_data.append({'symbol': item['symbol'], 'tokens': train_tokens})
        val_data.append({'symbol': item['symbol'], 'tokens': val_tokens})

    train_dataset = PreTokenizedCurveDataset(train_data, seq_len, label_len, pred_len)
    val_dataset = PreTokenizedCurveDataset(val_data, seq_len, label_len, pred_len)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    if len(train_dataset) == 0:
        print("Train dataset is empty. Skipping training.")
        return

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    print("Initializing model...")
    model = InformerStack(
        enc_in=5, # Not used for curve embedding
        dec_in=5, # Not used for curve embedding
        c_out=c_out, # Output logits for 5 tokens
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len,
        d_model=d_model,
        embed='curve',
        attn='full', # Use full attention for short sequences
        dropout=0.1, # Add dropout
        vocab_size=vocab_size,
        num_types=num_types,
        device=device
    )
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5) # Add weight decay
    
    losses = []
    val_losses = []
    train_epoch_losses = []
    
    labeler = Labeler()
    visualizer = Visualizer(save_dir="plots/steps")

    # Early Stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "best_model.pth"

    print("Starting training...")
    for epoch in tqdm(range(epochs)):
        # Training Phase
        model.train()
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # batch_x: [B, SeqLen, 5]
            # batch_y: [B, LabelLen+PredLen, 5]
            
            # Decoder input
            # Mask the future (prediction) part with zeros to prevent leakage
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).long()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).long().to(device)
            
            optimizer.zero_grad()
            
            # Forward
            # Note: We need to pass x_mark as None
            outputs = model(batch_x, None, dec_inp, None)
            
            # outputs: [B, PredLen, c_out]
            # Reshape to [B, PredLen, 5, output_vocab_size]
            outputs = outputs.view(outputs.shape[0], pred_len, 5, output_vocab_size)
            
            # Target: [B, PredLen, 5]
            target_tokens = batch_y[:, -pred_len:, :]
            
            # Calculate loss
            # Flatten for CrossEntropyLoss: [N, C] vs [N]
            # N = B * PredLen * 5
            # C = output_vocab_size
            
            loss = criterion(outputs.reshape(-1, output_vocab_size), target_tokens.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            losses.append(loss.item())
            
            global_step = epoch * len(train_loader) + i
            if global_step % plot_step == 0:
                # Visualize on training data
                visualizer.visualize(batch_y, outputs, global_step, discretizer, labeler, pred_len)
                
        avg_train_loss = total_loss / len(train_loader)
        train_epoch_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(val_loader):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).long()
                dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).long().to(device)
                
                outputs = model(batch_x, None, dec_inp, None)
                outputs = outputs.view(outputs.shape[0], pred_len, 5, output_vocab_size)
                target_tokens = batch_y[:, -pred_len:, :]
                
                loss = criterion(outputs.reshape(-1, output_vocab_size), target_tokens.reshape(-1))
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_losses.append(avg_val_loss)
        
        # tqdm.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # tqdm.write(f"Validation loss improved. Saved model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print("Training finished.")
    
    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")

    # Plot loss
    plt.figure()
    plt.plot(train_epoch_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss.png")
    print("Loss curve saved to plots/loss.png")


if __name__ == "__main__":
    main()
