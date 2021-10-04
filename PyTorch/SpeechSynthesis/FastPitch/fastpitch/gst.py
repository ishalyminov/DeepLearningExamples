import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from fastpitch.transformer import FFTransformer, MultiHeadAttn


class GST(nn.Module):
    def __init__(
        self,
        gst_n_layers,
        gst_n_heads,
        n_mel_channels,
        gst_d_head,
        gst_conv1d_filter_size,
        gst_conv1d_kernel_size,
        p_gst_dropout,
        p_gst_dropatt,
        p_gst_dropemb,
    ):
        super(GST, self).__init__()

        self.encoder = FFTransformer(
            n_layer=gst_n_layers,
            n_head=gst_n_heads,
            d_model=n_mel_channels,
            d_head=gst_d_head,
            d_inner=gst_conv1d_filter_size,
            kernel_size=gst_conv1d_kernel_size,
            dropout=p_gst_dropout,
            dropatt=p_gst_dropatt,
            dropemb=p_gst_dropemb,
            embed_input=False
        )
        self.gru = nn.GRU(input_size=n_mel_channels,
                          hidden_size=n_mel_channels // 2,
                          batch_first=True)

        # additional layer to be used in GST
        # !!!!!!!!!!!!!!!!!!!!!!
        # INSERT YOUR CODE HERE!
        # !!!!!!!!!!!!!!!!!!!!!!
        self.stl = STL(
            n_mel_channels,
            gst_d_head,
            gst_n_heads,
        )

    def forward(self, mels, mel_lengths):
        """
        Should take GT mels and return style embeddings and attention probs
        """
        mels_enc, alphas = self.encoder(mels, mel_lengths)
        _, mels_enc_gru = self.gru(mels_enc)
        mels_enc_gru = mels_enc_gru.squeeze()

        """
        mel-invariant tokens should be keys in the attention here
        and encoder output -- the query


        Please note that you should prevent unlimited growth of tokens magnitude somehow
        E.g. you can shrink them with some restrictive function:
            attention_keys = tanh(tokens_embeddings)
        """

        # !!!!!!!!!!!!!!!!!!!!!!
        # INSERT YOUR CODE HERE!
        # !!!!!!!!!!!!!!!!!!!!!!
        style_embed = self.stl(mels_enc_gru)

        style_embed = style_embed.repeat(mels.size(0) // style_embed.size(0), mels.size(1) // style_embed.size(1), mels.size(2) // style_embed.size(2))
        return style_embed, alphas


class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(
            self,
            token_num,
            embedding_dim,
            num_heads,
    ):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(token_num, embedding_dim // num_heads))
        d_q =token_num // 2
        d_k = embedding_dim // num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = GSTMultiHeadAttn(query_dim=d_q, key_dim=d_k, num_units=embedding_dim, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class GSTMultiHeadAttn(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
