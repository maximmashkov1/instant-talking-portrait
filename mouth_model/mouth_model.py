import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 

class SpeechToLip(nn.Module):

    def __init__(self, normalization=None, n_mels=80, hidden_size=32, num_layers=2, linear_size=64, morph_target_count=7, dropout=0.1, original_fps=30):
        super().__init__()
        self.original_fps = original_fps
        self.n_mels = n_mels
        self.hidden_size = hidden_size
        self.morph_target_count = morph_target_count

        self.normalization = normalization
        
        self.gru = nn.GRU(input_size=n_mels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.morph_blending = nn.Sequential(nn.Linear(hidden_size*2, linear_size), nn.BatchNorm1d(linear_size), nn.LeakyReLU(), nn.Dropout(dropout),
                                           nn.Linear(linear_size, morph_target_count))
        
        self.morph_mapping = nn.Linear(morph_target_count,25, bias=False) #each weight column represents a morph target

    def get_morph_targets(self):
        return [self.morph_mapping.weight[:, n].detach().cpu() for n in range(self.morph_target_count)]

    def get_loss(self, batch):
        x = batch[:, :, :self.n_mels]
        target = batch[:, :, self.n_mels:].view(-1, 25)
        morph_blending = self.forward(x).reshape(-1, self.morph_target_count)
        morph_mapping = self.morph_mapping(morph_blending)
        return F.mse_loss(morph_mapping, target, reduction='mean')
    
    def forward(self, x):

        #expects x normalized with self.normalization
        
        gru_output = self.gru(x)[0]
        shape = gru_output.shape
        gru_output = gru_output.reshape(-1, self.hidden_size*2)
        morph_blending = self.morph_blending(gru_output)
        morph_blending.reshape(shape[0], -1, self.morph_target_count)
        return morph_blending