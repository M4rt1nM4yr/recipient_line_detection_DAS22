import numpy as np
import torch
from torch import nn


class LinRecClassifierCnn(nn.Module):

        def __init__(self,
                     padding_idx=1,
                     content_length=102,
                     embed_dim=300,
                     filter_sizes=[3, 4, 5],
                     num_filters=[64, 128, 256],
                     num_classes=1,
                     dropout=0.5,):
            super(LinRecClassifierCnn, self).__init__()
            self.embed_dim = embed_dim
            self.embedding = nn.Embedding(num_embeddings=content_length,
                                          embedding_dim=self.embed_dim,
                                          padding_idx=padding_idx)

            self.conv1d_list = nn.ModuleList([
                nn.Conv1d(in_channels=self.embed_dim,
                          out_channels=num_filters[i],
                          kernel_size=filter_sizes[i])
                for i in range(len(filter_sizes))
            ])

            self.fc = nn.Linear(np.sum(num_filters), num_classes)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, input_ids, padding=None):
            x_embed = self.embedding(input_ids.long()).float()
            x_reshaped = x_embed.permute(0, 2, 1)
            x_conv_list = [torch.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
            x_pool_list = [torch.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                           for x_conv in x_conv_list]
            x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                             dim=1)
            return logits = self.fc(self.dropout(x_fc))
