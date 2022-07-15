from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer, AutoTokenizer, AutoFeatureExtractor
import torch


class MSRDataset(Dataset):
    def __init__(self, text_dict: dict, img_dict: dict, label_dict, label_mapping: dict, text_tokenizer_name: str, img_tokenizer_name: str):
        self.index_id_map = {}
        self.tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_name)
        self.extractor = AutoFeatureExtractor.from_pretrained(img_tokenizer_name)
        self.texts = []
        self.imgs = []
        self.labels = []
        if label_dict is not None:
            self.mode = 'train'
        else:
            self.mode = 'test'
        index_cnt = 0
        for key in text_dict:
            text = text_dict[key]
            text_token = self.tokenizer.encode_plus(text, max_length=256, padding='max_length', truncation=True)
            self.texts.append(text_token)
            img = img_dict[key]
            img = self.extractor(images=img)
            self.imgs.append(img)
            if label_dict is not None:
                label = label_dict[key]
                self.labels.append(label_mapping[label])
            self.index_id_map[index_cnt] = key
            index_cnt += 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_instance = self.texts[index]
        input_ids = torch.tensor(text_instance['input_ids'])
        attention_mask = torch.tensor(text_instance['attention_mask'])
        if self.mode == 'train':
            return input_ids, attention_mask, self.imgs[index], self.labels[index]
        else:
            return input_ids, attention_mask, self.imgs[index], None


class MSRTextOnlyDataset(Dataset):
    def __init__(self, text_dict: dict, label_dict, label_mapping: dict, tokenizer_name: str):
        self.index_id_map = {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = []
        self.imgs = []
        self.labels = []
        if label_dict is not None:
            self.mode = 'train'
        else:
            self.mode = 'test'
        index_cnt = 0
        for key in text_dict:
            text = text_dict[key]
            text_token = self.tokenizer.encode_plus(text, max_length=256, padding='max_length', truncation=True)
            self.texts.append(text_token)
            if label_dict is not None:
                label = label_dict[key]
                self.labels.append(label_mapping[label])
            self.index_id_map[index_cnt] = key
            index_cnt += 1

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text_instance = self.texts[index]
        input_ids = torch.tensor(text_instance['input_ids'])
        attention_mask = torch.tensor(text_instance['attention_mask'])
        token_type_ids = torch.zeros(input_ids.shape)
        if 'token_type_ids' in text_instance:
            token_type_ids = torch.tensor(text_instance['token_type_ids'])
        if self.mode == 'train':
            return input_ids, attention_mask, token_type_ids, self.labels[index]
        else:
            return input_ids, attention_mask, token_type_ids, None


class MSRImageOnlyDataset(Dataset):
    def __init__(self, img_dict: dict, label_dict, label_mapping: dict, tokenizer_name=None):
        self.index_id_map = {}
        self.imgs = []
        self.labels = []
        if label_dict is not None:
            self.mode = 'train'
        else:
            self.mode = 'test'
        if tokenizer_name is not None:
            self.extractor = AutoFeatureExtractor.from_pretrained(tokenizer_name)
        index_cnt = 0
        for key in img_dict:
            img = img_dict[key]
            if tokenizer_name is not None:
                img = self.extractor(images=img)
            self.imgs.append(img)
            if label_dict is not None:
                label = label_dict[key]
                self.labels.append(label_mapping[label])
            self.index_id_map[index_cnt] = key
            index_cnt += 1

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.imgs[index], self.labels[index]
        else:
            return self.imgs[index], None