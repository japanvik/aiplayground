import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()

    def forward(self, x):
        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)

        out = self.gamma * attn + x
        return out


class SelfAttention(AttentionModel):
    # block to allow convolutions to take a sneak peak at other areas of the image
    def __init__(self, input_c):
        super(SelfAttention, self).__init__()
        self.query = spectral_norm(nn.Conv1d(input_c, input_c // 8, 1))
        self.key = spectral_norm(nn.Conv1d(input_c, input_c // 8, 1))
        self.value = spectral_norm(nn.Conv1d(input_c, input_c, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))


class SelfAttention2D(nn.Module):
    # block to allow convolutions to take a sneak peak at other areas of the image
    # 2dConv to see if it is any different
    def __init__(self, input_c):
        super(SelfAttention2D, self).__init__()
        output_c = input_c // 8
        self.query = nn.Conv2d(input_c, output_c, 1)
        self.key = nn.Conv2d(input_c, output_c, 1)
        self.value = nn.Conv2d(input_c, input_c, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        dims = (x.size(0), -1, x.size(2) * x.size(3))
        out_query = self.query(x).view(dims)
        out_key = self.key(x).view(dims).permute(0, 2, 1)
        attn = F.softmax(torch.bmm(out_key, out_query), dim=-1)
        out_value = self.value(x).view(dims)
        out_value = torch.bmm(out_value, attn).view(x.size())
        out = self.gamma * out_value + x
        return out

