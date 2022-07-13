from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, AutoTokenizer
import torch


class MSRDataset(Dataset):
    def __init__(self, text_dict: dict, label_dict: dict, label_mapping: dict, tokenizer_name: str):
        self.index_id_map = {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = []
        self.labels = []
        index_cnt = 0
        for key in text_dict:
            text = text_dict[key]
            label = label_dict[key]
            text_token = self.tokenizer.encode_plus(text, max_length=256, padding='max_length', truncation=True)
            self.texts.append(text_token)
            self.labels.append(label_mapping[label])
            self.index_id_map[index_cnt] = key
            index_cnt += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text_instance = self.texts[index]
        input_ids = torch.tensor(text_instance['input_ids'])
        attention_mask = torch.tensor(text_instance['attention_mask'])
        return input_ids, attention_mask, self.labels[index]
