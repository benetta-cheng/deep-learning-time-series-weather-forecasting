# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
        super(EncoderGRU, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers = num_layers,
                          dropout = dropout if num_layers > 1 else 0.0,
                          batch_first = False)

    def forward(self, input_seq):
        # input_seq: (seq_len, batch_size, input_size)
        output, hidden = self.gru(input_seq)
        return hidden
    

class DecoderGRU(nn.Module):
    def __init__(self, hidden_size, feature_size, num_layers = 1, dropout = 0.0):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size # 4 features
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size=feature_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          dropout = dropout if num_layers > 1 else 0.0,
                          batch_first=False)

        self.linear = nn.Linear(hidden_size, feature_size)

    def forward(self, x, hidden, output_length, target=None):
        outputs = []
        decoder_input = x  # shape: (1, batch_size, 4)

        for i in range(output_length):
            # One step forward through GRU
            decoder_output, hidden = self.gru(decoder_input, hidden)
            decoder_output = self.linear(decoder_output)  # shape: (1, batch_size, 4)
            outputs.append(decoder_output)

            if target is not None:
                # Teacher forcing: use actual target
                decoder_input = target[i].unsqueeze(0)  # shape: (1, batch_size, 4)
            else:
                # Auto-regressive: use own prediction
                decoder_input = decoder_output.detach()  # detach to prevent gradient flow

        return torch.cat(outputs, dim=0)  # shape: (output_length, batch_size, 4)
    
class Seq2SeqGRU(nn.Module):
    def __init__(self, hidden_size, feature_size, output_size, num_layers=1, dropout=0.0):
        super(Seq2SeqGRU, self).__init__()

        self.output_length = output_size

        self.encoder = EncoderGRU(input_size=feature_size, # 4 features
                                  hidden_size=hidden_size,
                                  num_layers= num_layers,
                                  dropout=dropout)

        self.decoder = DecoderGRU(hidden_size=hidden_size,
                                  feature_size=feature_size,
                                  num_layers=num_layers,
                                  dropout=dropout)  # 4 features
        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)

    def forward(self, inputs, outputs=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, input_size=4)
            outputs: optional teacher forcing target, shape (output_length, batch_size, 1)
        """
        hidden = self.encoder(inputs)  # returns (1, batch_size, hidden_size)

        # Initial decoder input: zeros (1, batch_size, 4)
        decoder_input = torch.zeros(1, inputs.shape[1], self.decoder.feature_size).to(inputs.device)

        # Pass to decoder
        output = self.decoder(decoder_input, hidden, self.output_length, outputs)
        return output  # shape: (output_length, batch_size, 1)