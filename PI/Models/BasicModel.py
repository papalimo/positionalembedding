import torch
import torch.nn as nn
import math
from Embedding import PI,LPI,RePE,RPE,TSE



class TransformerModel(nn.Module):
    def __init__(self, pe,ntoken: int, d_model: int, nhead: int, nhid: int, nlayers: int, dropout: float = 0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

        self.model_type = 'Transformer'
        self.src_mask = None
        if pe=='PI':
            self.pos_encoder = PI(d_model, dropout)
        elif pe=='LPI':
            self.pos_encoder = LPI(d_model, dropout)
        self.encoder = nn.Embedding(ntoken, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        decoder_layers = TransformerDecoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if src_mask is None or tgt_mask is None:
            device = src.device
            src_mask = self.generate_square_subsequent_mask(src.size(0)).to(device)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(device)

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        output = self.decoder(output)
        return output