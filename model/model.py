from transformers import XLMRobertaModel
from torch import nn
import torch
import math


class ReluOrGelu(nn.Module):
    def __init__(self, activate_type: str):
        super(ReluOrGelu, self).__init__()
        self.activate_type = activate_type

    def forward(self, x):
        if self.activate_type == 'relu':
            return torch.relu(x)
        elif self.activate_type == 'gelu':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TextEmbedding(nn.Module):
    def __init__(self, text_weight_name, out_dim):
        self.roberta = XLMRobertaModel.from_pretrained(text_weight_name)
        self.aligner = nn.Sequential(
            nn.Linear(self.roberta.get_output_dim(), out_dim),
            ReluOrGelu("gelu")
        )

    def forward(self, input_ids, attention_mask):
        feature = self.roberta(input_ids, attention_mask=attention_mask)
        out = self.aligner(feature.last_hidden_state)
        return out


class PTAMSC(nn.Module):
    def __init__(self, text_model_num, image_model_num):
        super(PTAMSC, self).__init__()
        self.text_embedding = TextEmbedding(text_model_num, 768)

    def forward(self):
        pass