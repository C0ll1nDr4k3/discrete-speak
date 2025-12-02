import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from .discretization import Discretizer
from .labeler import Labeler


class Visualizer:
    def __init__(self, save_dir: str = "plots/steps"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Determine number of params needed for each curve type
        self.param_counts = {
            "linear": 2,
            "quadratic": 3,
            "cubic": 4,
            "exponential": 2,
            "constant": 1
        }

    def visualize(self, batch_y: torch.Tensor, outputs: torch.Tensor, step: int, discretizer: Discretizer, labeler: Labeler, pred_len: int):
        """
        Visualizes actual vs predicted curves.
        
        Args:
            batch_y: Target tokens [B, LabelLen+PredLen, 5]
            outputs: Predicted logits [B, PredLen, 5, VocabSize]
            step: Current training step
            discretizer: Instance of Discretizer for decoding
            labeler: Instance of Labeler for curve functions
            pred_len: Length of prediction sequence
        """
        # Take the last sample in the batch
        target_tokens = batch_y[-1, -pred_len:, :].cpu() # [PredLen, 5]
        
        # Get predicted tokens
        # outputs: [B, PredLen, 5, VocabSize] -> [PredLen, 5, VocabSize]
        pred_logits = outputs[-1] 
        pred_tokens = torch.argmax(pred_logits, dim=-1).cpu() # [PredLen, 5]
        
        # Reconstruct curves
        segment_len = 10 # Arbitrary length for visualization
        total_len = pred_len * segment_len
        x_axis = np.arange(total_len)
        
        actual_y = []
        predicted_y = []
        
        for i in range(pred_len):
            # Actual
            t_type_idx = target_tokens[i, 0].item()
            t_params_idx = target_tokens[i, 1:].tolist()
            
            try:
                t_type = discretizer.get_curve_type(t_type_idx)
                t_params = [discretizer.get_param_value(p) for p in t_params_idx]
            except ValueError:
                t_type = "constant"
                t_params = [0.0] * 4

            # Predicted
            p_type_idx = pred_tokens[i, 0].item()
            p_params_idx = pred_tokens[i, 1:].tolist()
            
            try:
                p_type = discretizer.get_curve_type(p_type_idx)
            except ValueError:
                p_type = "constant" 
                
            try:
                p_params = [discretizer.get_param_value(p) for p in p_params_idx]
            except ValueError:
                p_params = [0.0] * 4
            
            # Generate points
            seg_x = np.arange(1, segment_len + 1)
            
            # Generate Actual Segment
            if t_type in labeler.functions:
                n_params = self.param_counts.get(t_type, 4)
                actual_seg = labeler.functions[t_type](seg_x, *t_params[:n_params])
            else:
                actual_seg = np.zeros(segment_len)
                
            # Generate Predicted Segment
            if p_type in labeler.functions:
                n_params = self.param_counts.get(p_type, 4)
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
        
        plt.savefig(os.path.join(self.save_dir, f"step_{step}.png"))
        plt.close()
