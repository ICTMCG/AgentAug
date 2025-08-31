import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(d_k)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        attn = self.softmax(scores)
        output = torch.matmul(attn, v)
        return output, attn

class _MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout):
        super(_MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.n_heads = n_heads

        self.w_q = nn.Linear(d_model, d_k * n_heads)
        self.w_k = nn.Linear(d_model, d_k * n_heads)
        self.w_v = nn.Linear(d_model, d_v * n_heads)

    def forward(self, q, k, v):
        b_size = q.size(0)

        # q_s, k_s, v_s: [b_size x n_heads x len_{q,k,v} x d_{k,v}]
        q_s = self.w_q(q).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.w_k(k).view(b_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.w_v(v).view(b_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 返回处理后的 Q, K, V
        return q_s, k_s, v_s


class CrossModal_Attention(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model,  fea_v_dim, fea_s_dim):
        super(CrossModal_Attention, self).__init__()
        self.n_heads = n_heads
        self.d_v = d_v
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

        self.multihead_attn_v = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.multihead_attn_s = _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)
        self.attention = ScaledDotProductAttention(d_k, dropout)
        self.linear_v = nn.Linear(in_features=fea_v_dim, out_features=d_model)
        self.linear_s = nn.Linear(in_features=fea_s_dim, out_features=d_model)

        self.proj_v = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm_v = nn.LayerNorm(d_model)

        self.relu=nn.ReLU()

    def forward(self, v, s): #用s引导v
        b_size = v.size(0)

        v,s=self.linear_v(v),self.linear_s(s)
        
        q_v, k_v, v_v = self.multihead_attn_v(v, v, v)
        q_s, k_s, v_s = self.multihead_attn_s(s, s, s)

        # A模态引导B模态（例如，s引导v）
        context_v, attn_v = self.attention(q_v, k_s, v_s)

        context_v = context_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        output_v = self.dropout(self.proj_v(context_v))

        return self.layer_norm_v(self.relu(v + output_v))

class CrossModal_Attention_wo_residual(nn.Module):
    def __init__(self, d_k, d_v, n_heads, dropout, d_model,  fea_q, fea_kv):
        super(CrossModal_Attention_wo_residual, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)

        self.get_qkv= _MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout)

        self.attention = ScaledDotProductAttention(d_v, dropout)

        self.linear_q_input = nn.Linear(in_features=fea_q, out_features=d_model)
        self.linear_kv_input = nn.Linear(in_features=fea_kv, out_features=d_model)

        self.proj_v = nn.Linear(n_heads * d_v, d_model)

        self.layer_norm_v = nn.LayerNorm(d_model)

        self.relu=nn.ReLU()

    def forward(self, v, q): #用s引导v
        b_size = v.size(0)

        v,q=self.linear_kv_input(v),self.linear_q_input(s)
        
        projected_q, projected_k, projected_v = self.multihead_attn_v(q, v, v)

        guided_v, attn_score = self.attention(projected_q, projected_k, projected_v)

        guided_v = guided_v.transpose(1, 2).contiguous().view(b_size, -1, self.n_heads * self.d_v)

        output_v = self.dropout(self.proj_v(guided_v))

        return output_v

