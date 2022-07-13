from torch import nn
from transformers import XLMRobertaModel


class PureRoberta(nn.Module):
    def __init__(self, model_name, num_classes):
        super(PureRoberta, self).__init__()
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        self.dense = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        feature = self.roberta(input_ids, attention_mask=attention_mask)
        out = self.dense(feature.pooler_output)
        return out