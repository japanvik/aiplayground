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


class SelfAttention2D(AttentionModel):
    # block to allow convolutions to take a sneak peak at other areas of the image
    # 2dConv to see if it is any different
    def __init__(self, input_c):
        super(self.__class__, self)
        self.query = spectral_norm(nn.Conv2d(input_c, input_c // 8, 1))
        self.key = spectral_norm(nn.Conv2d(input_c, input_c // 8, 1))
        self.value = spectral_norm(nn.Conv2d(input_c, input_c, 1))
        self.gamma = nn.Parameter(torch.tensor(0.0))


