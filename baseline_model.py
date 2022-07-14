from torch import nn
from transformers import BertModel, XLMRobertaModel, SwinModel


class PureBert(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PureBert, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        feature = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        out = self.dense(feature.pooler_output)
        return out


class PureRoberta(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PureRoberta, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        feature = self.roberta(input_ids, attention_mask=attention_mask)
        out = self.dense(feature.pooler_output)
        return out


class PureSwinTransformer(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PureSwinTransformer, self).__init__()
        self.swin = SwinModel.from_pretrained(model_name)
        self.dense = nn.Linear(1024, num_classes)

    def forward(self, pixel_values):
        feature = self.swin(pixel_values=pixel_values)
        out = self.dense(feature.pooler_output)
        return out
