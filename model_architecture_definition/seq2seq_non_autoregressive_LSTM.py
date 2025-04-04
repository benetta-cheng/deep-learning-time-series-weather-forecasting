# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1, dropout = 0.0):
        super(EncoderLSTM, self).__init__() 
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, 
                            num_layers = num_layers, 
                            dropout = dropout if num_layers > 1 else 0.0, 
                            batch_first = False)

    def forward(self, input_seq):
        # For this encoder, we ignore the outputs if we only need the final hidden state(s)
        outputs, hidden = self.lstm(input_seq)
        return hidden
    

class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0.0):
        super(DecoderLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.num_layers = num_layers
        
        self.output_size=output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, 
                            num_layers = num_layers, 
                            dropout = dropout if num_layers > 1 else 0.0, 
                            batch_first = False)
        
        self.linear = nn.Linear(hidden_size, self.output_size)

    def forward(self, input_seq, hidden,output_length,target=None):
        outputs = []
        decoder_input = input_seq 

        for i in range(output_length):
            # One step forward through LSTM
            decoder_output, hidden = self.lstm(decoder_input, hidden)
            
            decoder_output = self.linear(decoder_output)  
            outputs.append(decoder_output)

            if target is not None:
                decoder_input = target[i].unsqueeze(0)
            else:
                decoder_input = decoder_output.detach() 

        return torch.cat(outputs, dim=0)  # shape: (output_length, batch_size, 1)
    
class Seq2SeqLSTM(nn.Module):
    def __init__(self, hidden_size, input_size, output_size,output_length,num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        
        self.output_length = output_length
        
        self.encoder = EncoderLSTM(input_size=input_size, 
                                   hidden_size=hidden_size,
                                   num_layers=num_layers)  # 4 features
        
        self.decoder = DecoderLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   output_size=output_size,
                                   num_layers=num_layers)  # 4 features

    def forward(self, inputs, outputs=None):
        hidden = self.encoder(inputs)  # returns (1, batch_size, hidden_size)

        # Initial decoder input: zeros (1, batch_size, 4)
        decoder_input = torch.zeros(1, inputs.shape[1], self.decoder.output_size).to(inputs.device)

        # Pass to decoder
        output = self.decoder(decoder_input, hidden, self.output_length, outputs)
        
        return output  # shape: (output_length, batch_size, 1)