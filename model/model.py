from transformers import XLMRobertaModel, SwinModel
from torch import nn
from model.roberta import RobertaEncoder
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
        super(TextEmbedding, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(text_weight_name)
        self.aligner = nn.Sequential(
            nn.Linear(768, out_dim),
            ReluOrGelu("gelu")
        )

    def forward(self, input_ids, attention_mask):
        feature = self.roberta(input_ids, attention_mask=attention_mask)
        out = self.aligner(feature.last_hidden_state)
        return out


class ImageEmbbedding(nn.Module):
    def __init__(self, img_weight_name, out_dim):
        super(ImageEmbbedding, self).__init__()
        self.swin = SwinModel.from_pretrained(img_weight_name)
        self.aligner = nn.Sequential(
            nn.Linear(1024, out_dim),
            ReluOrGelu("gelu")
        )

    def forward(self, pixel_values):
        feature = self.swin(pixel_values=pixel_values)
        out = self.aligner(feature.last_hidden_state)
        return out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            ReluOrGelu('gelu'),
            nn.Linear(768, 3)
        )

    def forward(self, feature):
        return self.linear(feature)


class PTAMSC(nn.Module):
    def __init__(self, text_model_name, img_model_name):
        super(PTAMSC, self).__init__()
        self.text_backbone_name = text_model_name
        self.img_backbone_name = img_model_name
        self.text_embedding = TextEmbedding(text_model_name, 768)
        self.img_embedding = ImageEmbbedding(img_model_name, 768)
        self.fuse_config = self.text_embedding.roberta.config
        self.fuser = RobertaEncoder(self.fuse_config)
        self.output_attention = nn.Sequential(
            nn.Linear(768, 768 // 2),
            ReluOrGelu('gelu'),
            nn.Linear(768 // 2, 1)
        )
        self.classifier = Classifier()

    def forward(self, text_input_ids, text_attention_mask, img_pixel_values):
        text_feature = self.text_embedding(text_input_ids, text_attention_mask)
        img_feature = self.img_embedding(img_pixel_values)
        concat_feature = torch.cat((text_feature, img_feature), dim=1)
        fused_feature = self.fuser(concat_feature)

        text_image_output = fused_feature.last_hidden_state.contiguous()
        text_image_alpha = self.output_attention(text_image_output)
        text_image_alpha = text_image_alpha.squeeze(-1)
        text_image_alpha = torch.softmax(text_image_alpha, dim=-1)

        text_image_output = (text_image_alpha.unsqueeze(-1) * text_image_output).sum(dim=1)
        out = self.classifier(text_image_output)
        return out
