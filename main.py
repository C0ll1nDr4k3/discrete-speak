from datetime import datetime, timedelta
from time import time_ns

from alpaca.data.timeframe import TimeFrame

from ds.bars import Conversion
from ds.config import Config
from ds.fit import fit, print_labels
from ds.retrieval import Security
from ds.thresholds import Threshold
from ds.generate import save_dataset_from_results


def main() -> None:
    # Configure a 7-day, 1-minute retrieval window with all parameters
    config = Config(
        start=datetime.now() - timedelta(days=120),
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
        # "META",
        "AAPL",
        # "GOOG",
        # "GOOGL",
        "NVDA",
        # "TSM",
        # "ORCL",
        "PLTR",
        # "XOM",
        # "RY",
        # "DB",
        # "JPM",
    ]

    start = time_ns()
    results = fit(symbols, Security.EQUITIES, config)
    end = time_ns()

    print_labels(results)
    
    # Generate dataset
    save_dataset_from_results(results, config, "dataset.pt")

    print(f"Training completed in {end - start} nanoseconds")

    # --- Training Loop ---
    import torch
    from torch.utils.data import DataLoader
    from ds.data import PreTokenizedCurveDataset
    from ds.models.model import InformerStack
    from ds.discretization import Discretizer

    print("Loading dataset...")
    try:
        dataset_data = torch.load("dataset.pt")
    except FileNotFoundError:
        print("Dataset not found. Skipping training.")
        return

    # Hyperparameters
    seq_len = 32
    label_len = 16
    pred_len = 8
    batch_size = 32
    d_model = 512
    learning_rate = 1e-4
    epochs = 30
    plot_step = 10

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

    dataset = PreTokenizedCurveDataset(dataset_data, seq_len, label_len, pred_len)
    if len(dataset) == 0:
        print("Dataset is empty. Skipping training.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        vocab_size=vocab_size,
        num_types=num_types,
        device=device
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    print("Starting training...")
    model.train()
    losses = []
    
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from ds.labeler import Labeler
    
    labeler = Labeler()

    def visualize_predictions(batch_y, outputs, step, discretizer, labeler):
        """
        Visualizes actual vs predicted curves.
        batch_y: [B, LabelLen+PredLen, 5]
        outputs: [B, PredLen, 5, VocabSize]
        """
        # Take the last sample in the batch
        target_tokens = batch_y[-1, -pred_len:, :].cpu() # [PredLen, 5]
        
        # Get predicted tokens
        # outputs: [B, PredLen, 5, VocabSize] -> [PredLen, 5, VocabSize]
        pred_logits = outputs[-1] 
        pred_tokens = torch.argmax(pred_logits, dim=-1).cpu() # [PredLen, 5]
        
        # Reconstruct curves
        # We need to plot them sequentially.
        # Let's assume each segment is length 10 (arbitrary for visualization if we don't have original length)
        # Or we can just plot the functions.
        
        segment_len = 10
        total_len = pred_len * segment_len
        x_axis = np.arange(total_len)
        
        actual_y = []
        predicted_y = []
        
        for i in range(pred_len):
            # Actual
            t_type_idx = target_tokens[i, 0].item()
            t_params_idx = target_tokens[i, 1:].tolist()
            
            t_type = discretizer.get_curve_type(t_type_idx)
            t_params = [discretizer.get_param_value(p) for p in t_params_idx]
            
            # Predicted
            p_type_idx = pred_tokens[i, 0].item()
            p_params_idx = pred_tokens[i, 1:].tolist()
            
            # Handle potential out of bounds for predicted indices
            try:
                p_type = discretizer.get_curve_type(p_type_idx)
            except ValueError:
                p_type = "constant" # Fallback
                
            try:
                p_params = [discretizer.get_param_value(p) for p in p_params_idx]
            except ValueError:
                p_params = [0.0] * 4 # Fallback
            
            # Generate points
            seg_x = np.arange(1, segment_len + 1)
            
            # Determine number of params needed
            # linear: 2, quadratic: 3, cubic: 4, exponential: 2
            param_counts = {
                "linear": 2,
                "quadratic": 3,
                "cubic": 4,
                "exponential": 2,
                "constant": 1 # Special case handled in labeler but useful here
            }
            
            if t_type in labeler.functions:
                n_params = param_counts.get(t_type, 4)
                actual_seg = labeler.functions[t_type](seg_x, *t_params[:n_params])
            else:
                actual_seg = np.zeros(segment_len)
                
            if p_type in labeler.functions:
                n_params = param_counts.get(p_type, 4)
                predicted_seg = labeler.functions[p_type](seg_x, *p_params[:n_params])
            else:
                predicted_seg = np.zeros(segment_len)
                
            actual_y.extend(actual_seg)
            predicted_y.extend(predicted_seg)
            
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, actual_y, label="Actual", linestyle="-")
        plt.plot(x_axis, predicted_y, label="Predicted", linestyle="--")
        plt.title(f"Actual vs Predicted (Step {step})")
        plt.legend()
        
        os.makedirs("plots/steps", exist_ok=True)
        plt.savefig(f"plots/steps/step_{step}.png")
        plt.close()

    for epoch in range(epochs):
        total_loss = 0
        for i, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # batch_x: [B, SeqLen, 5]
            # batch_y: [B, LabelLen+PredLen, 5]
            
            # Decoder input
            dec_inp = batch_y[:, :label_len+pred_len, :]
            
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
            
            if i % plot_step == 0:
                print(f"Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item():.4f}")
                visualize_predictions(batch_y, outputs, epoch * len(dataloader) + i, discretizer, labeler)
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")

    print("Training finished.")

    # Plot loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("plots/loss.png")
    print("Loss curve saved to plots/loss.png")


if __name__ == "__main__":
    main()
