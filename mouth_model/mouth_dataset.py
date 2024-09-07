import random
import torch
import numpy as np
import os
import pickle

class SequenceDataset:
    def __init__(self, dataset_path, batch_size, val_size):
        if not dataset_path.endswith(".pickle"):
            sequences = [] 
            for i, file in enumerate(os.listdir(dataset_path)):
                sequence = torch.tensor(np.load(os.path.join(dataset_path,file)).astype(np.float32))
                sequences.append(sequence)
                if i%1000 == 0:
                    print(f"Loaded {i}")
            #store as pickle
            with open(dataset_path+".pickle", 'wb') as f: 
                pickle.dump(sequences,f)
        else:
            with open(dataset_path, 'rb') as f:
                sequences = pickle.load(f)


            
        random.shuffle(sequences)
        self.sequences, self.normalization = self.normalize_features(sequences)
        initial_len = len(self.sequences)
        self.sequences = [seq for seq in self.sequences if seq.shape[0] > 80]
        print(f"{len(self.sequences)/initial_len} left after filtering by > 80")

        self.batch_size = batch_size
        self.val_batch_size = val_size
        self.val_sequences = self.sequences[:val_size]
        self.sequences = self.sequences[val_size:]

        self.validation_batch = self.get_val_batch().cuda()

    def get_val_batch(self):
        val = []
        min_length = min([len(s) for s in self.val_sequences])
        for sequence in self.val_sequences:
            val.append(sequence[:min_length])
        
        print("Validation length: ", min_length)
        return torch.stack(val)


    def get_batch(self, batch_sequence_length):

        batch = []
        while len(batch) < self.batch_size:
            sequence = random.choice(self.sequences)
            
            if len(sequence) >= batch_sequence_length:
                start_idx = random.randint(0, len(sequence) - batch_sequence_length)
                subsequence = sequence[start_idx:start_idx + batch_sequence_length]
                batch.append(subsequence)
        
        return torch.stack(batch)

    def scale_features(self, data):
        concatenated_data = torch.cat(data, dim=0)
        lower_percentile = torch.quantile(concatenated_data[:5000], 0.01, dim=0)
        upper_percentile = torch.quantile(concatenated_data[:5000], 0.99, dim=0)
        scaled_data = [(seq - lower_percentile) / (upper_percentile - lower_percentile) for seq in data]
        scaled_data = [torch.nan_to_num(torch.clamp(seq, 0, 1)) for seq in scaled_data]
        
        return scaled_data, (lower_percentile, upper_percentile)

    def normalize_features(self, data):
        filtered_data = [seq for seq in data if not torch.isnan(seq).any()]
        concatenated_data = torch.cat(filtered_data, dim=0)
        mean = concatenated_data.mean(dim=0)
        std = concatenated_data.std(dim=0)
        
        normalized_data = [torch.nan_to_num((seq - mean) / std) for seq in filtered_data]
        
        print(f"Removed {100*(1-len(filtered_data) / len(data)):.2f} due nan values")
        
        return normalized_data, (mean, std)
    
    def scale_and_normalize_features(self, data, x=80):
        concatenated_data = torch.cat(data, dim=0)
        # Scale the first x features
        lower_percentile = torch.quantile(concatenated_data[:5000, :x], 0.01, dim=0)
        upper_percentile = torch.quantile(concatenated_data[:5000, :x], 0.99, dim=0)
        
        # Normalize the remaining features
        mean = concatenated_data[:, x:].mean(dim=0)
        std = concatenated_data[:, x:].std(dim=0)
        
        scaled_normalized_data = []
        for seq in data:
            scaled_part = (seq[:, :x] - lower_percentile) / (upper_percentile - lower_percentile)
            scaled_part = torch.clamp(scaled_part, 0, 1)  # Ensure values are within [0, 1]
            
            normalized_part = (seq[:, x:] - mean) / std
            normalized_part = torch.nan_to_num(normalized_part)  # Handle any potential NaN values
            
            combined = torch.cat([scaled_part, normalized_part], dim=1)
            scaled_normalized_data.append(combined)
        
        return scaled_normalized_data, (lower_percentile, upper_percentile, mean, std)
                
