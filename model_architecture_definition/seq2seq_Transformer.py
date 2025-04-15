import torch
import torch.nn as nn
import math

class TimeSeriesEmbedding(nn.Module):
    """
    Maps each 4D input vector to a higher-dimensional embedding (d_model)
    """
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)

    def forward(self, x):
        # Shape of input x: (batch_size, seq_len, input_dim)
        return self.linear(x)  # (batch_size, seq_len, d_model)


class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to the embedded vectors
    to incorporate sequence order into the Transformer
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (seq_len, batch_size, d_model)
        We add the positional encoding up to x.size(0) (the seq_len)
        """
        seq_len = x.size(0)
        return x + self.pe[:seq_len, :]


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diagonal
    Used to mask future positions in the decoder for auto‐regressive or
    teacher-forcing style training
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Seq2SeqTransformer(nn.Module):
    """
    A standard Transformer-based model for time-series forecasting:
    - Embeds the (batch, seq_len, 4) input to (seq_len, batch, d_model)
    - Feeds it into the Encoder
    - Uses the Decoder with an optional target to produce predictions
    - Outputs a final linear projection to get back to 4 features
    """
    def __init__(self,
                 input_dim,
                 d_model,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout,
                 output_dim):
        super(Seq2SeqTransformer, self).__init__()

        # Embedding layers for source and target
        self.input_emb = TimeSeriesEmbedding(input_dim, d_model)
        self.target_emb = TimeSeriesEmbedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer blocks
        # Encoder block
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder block
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dropout=dropout,
                                                   batch_first=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Final linear projection back to (4) features
        self.output_linear = nn.Linear(d_model, output_dim)

        self.d_model = d_model

    def forward(self, src, tgt):
        """
        src: (seq_len_in, batch_size, input_dim=4)
        tgt: (seq_len_out, batch_size, input_dim=4)
        Returns: (seq_len_out, batch_size, output_dim=4)
        """
        # Encode - Embed and add positional info
        src_emb = self.input_emb(src)   # (seq_len_in, batch, d_model)
        src_emb = self.pos_encoder(src_emb)
        # Pass through Transformer Encoder
        memory = self.encoder(src_emb)  # (seq_len_in, batch, d_model)

        # Decode - Similar embedding for target
        tgt_emb = self.target_emb(tgt)  # (seq_len_out, batch, d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Generate a subsequent mask for the target to prevent it from 'seeing' future positions
        seq_len_out = tgt.size(0)
        tgt_mask = generate_square_subsequent_mask(seq_len_out).to(tgt.device)
        # Pass through Transformer Decoder
        decoder_output = self.decoder(tgt=tgt_emb,
                                      memory=memory,
                                      tgt_mask=tgt_mask)
        # Final projection
        out = self.output_linear(decoder_output)  # shape: (seq_len_out, batch, 4)
        return out
