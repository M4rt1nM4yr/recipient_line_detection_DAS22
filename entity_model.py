from torch import nn
import pytorch_lightning as pl
from module_factory import produce_feature_extractor, produce_adaptive_2d_PE, produce_transformer, produce_1d_PE

class EntityModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=2048, n_head=1, dropout=0.1):
        super(EntityModel, self).__init__()
        ## ENCODER
        # cut R50 as encoder (1/32 scaling in height and width direction)
        self.feature_extractor = produce_feature_extractor()
        # conv2d to match the dimension size of the transformer encoder
        self.conv_hidden = nn.Conv2d(512, hidden_size, (1,1), stride=(1,1))
        # adaptive 2d positional encoding
        self.pe2d = produce_adaptive_2d_PE(d_model=hidden_size)
        # one transformer for all
        self.transformer = produce_transformer(d_model=hidden_size, n_head=n_head, dim_feedforwad=hidden_size, dropout=dropout)

        ## DECODER
        # embedding and 1d pe
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.pe1d = produce_1d_PE(d_model=hidden_size, dropout=dropout)
        # linear layer to output alphabet
        self.predictor = nn.Linear(hidden_size,vocab_size)

    def forward(self, x, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        # visual
        features = self.conv_hidden(self.feature_extractor(x))
        features_pe = self.pe2d(features)
        s = features_pe.shape
        seq_in = features_pe.reshape(s[0],s[1],-1).permute(2,0,1)
        # text
        y_emb = self.pe1d(self.embedder(tgt)).permute(1, 0, 2)
        transformer_out = self.transformer(src=seq_in, tgt=y_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        seq_out = transformer_out[0].permute(1,0,2)
        attention_map = transformer_out[1]
        pred = self.predictor(seq_out)
        return pred, attention_map, seq_out
