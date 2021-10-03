import torch
import torch.nn as nn
import torch.nn.init as init
import torch.functional as F
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
            n_symbols=None,
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

        # additional layer to be used in GST
        # !!!!!!!!!!!!!!!!!!!!!!
        # INSERT YOUR CODE HERE!
        # !!!!!!!!!!!!!!!!!!!!!!
        self.stl = STL(
            gst_d_head,
            gst_n_heads,
        )

    def forward(self, mels, mel_lengths):
        """
        Should take GT mels and return style embeddings and attention probs
        """
        mels_enc = self.encoder(mels, mel_lengths)
        # ...

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
        style_embed = self.stl(mels_enc)

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
        d_q = embedding_dim // 2
        d_k = embedding_dim // num_heads
        # self.attention = MultiHeadAttention(hp.num_heads, d_model, d_q, d_v)
        self.attention = MultiHeadAttn(query_dim=d_q, key_dim=d_k, num_units=embedding_dim, num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E//2]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed
