import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple
import numpy as np
from .discretization import Discretizer

class CurveTokenizer:
    def __init__(self, discretizer: Discretizer):
        self.discretizer = discretizer

    def tokenize(self, label_info: Dict[str, Any]) -> List[int]:
        """
        Converts a label dictionary to a list of token IDs.
        Format: [CurveTypeID, Param1ID, Param2ID, Param3ID, Param4ID]
        """
        curve_type = label_info["label"]
        params = label_info["params"]
        
        type_id = self.discretizer.CURVE_TYPE_MAP.get(curve_type, -1)
        if type_id == -1:
            raise ValueError(f"Unknown curve type: {curve_type}")
            
        param_ids = [self.discretizer.get_param_index(p) for p in params]
        
        # Pad parameters to MAX_PARAMS
        padding_needed = self.discretizer.MAX_PARAMS - len(param_ids)
        # Use 0 (or a specific padding token) for padding. 
        # Since 0 is a valid index in discrete_range, we might need a separate padding token 
        # or just reuse one if it doesn't matter (e.g. if the model learns to ignore padded params based on type).
        # For now, let's use 0, but ideally we should have a PAD token.
        # However, the plan didn't specify a PAD token for params. 
        # Let's assume the model will learn from the curve type which params to attend to.
        param_ids.extend([0] * padding_needed)
        
        return [type_id] + param_ids

    def tokenize_sequence(self, sequence: List[Dict[str, Any]]) -> List[List[int]]:
        return [self.tokenize(item) for item in sequence]

class CurveDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], discretizer: Discretizer, seq_len: int, label_len: int, pred_len: int):
        """
        Args:
            data: List of label dictionaries (output of fit).
            discretizer: Instance of Discretizer.
            seq_len: Input sequence length.
            label_len: Start token length for decoder.
            pred_len: Prediction sequence length.
        """
        self.tokenizer = CurveTokenizer(discretizer)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Flatten data into a single sequence of tokens if it's a list of segments
        # But `fit` returns a dict of symbols -> results.
        # We probably want to process one symbol's data or a list of all segments.
        # Assuming `data` is a list of segments (labels) from one or multiple symbols concatenated.
        
        self.tokens = []
        for item in data:
             self.tokens.append(self.tokenizer.tokenize(item))
             
        self.tokens = np.array(self.tokens) # [TotalSegments, 5]

    def __len__(self):
        return len(self.tokens) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.tokens[s_begin:s_end]
        seq_y = self.tokens[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.long), torch.tensor(seq_y, dtype=torch.long)

class PreTokenizedCurveDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], seq_len: int, label_len: int, pred_len: int):
        """
        Args:
            data: List of dicts {'symbol': str, 'tokens': Tensor} (loaded from .pt).
            seq_len: Input sequence length.
            label_len: Start token length for decoder.
            pred_len: Prediction sequence length.
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        # Flatten all tokens into a single sequence for now
        # Ideally we should respect symbol boundaries, but for simplicity let's concatenate
        # or just list all valid windows.
        # Let's concatenate for now to maximize data usage, assuming breaks are rare/acceptable noise.
        # Better: Keep them separate and index into them.
        
        self.sequences = []
        for item in data:
            self.sequences.append(item['tokens'])
            
        # Create a mapping from global index to (sequence_index, local_index)
        self.index_map = []
        for seq_idx, seq in enumerate(self.sequences):
            num_windows = len(seq) - seq_len - pred_len + 1
            if num_windows > 0:
                for i in range(num_windows):
                    self.index_map.append((seq_idx, i))
                    
    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        seq_idx, s_begin = self.index_map[index]
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq = self.sequences[seq_idx]
        
        seq_x = seq[s_begin:s_end]
        seq_y = seq[r_begin:r_end]

        return seq_x, seq_y
