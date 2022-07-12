from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label
from data_pipe import MSRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaForSequenceClassification, AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import torch


if __name__ == "__main__":
    text_embed_model_name = 'xlm-roberta-base'
    train_iteration = 4
    all_text_dict = read_folder_text('data/source')
    train_label_dict = read_train_tag('data/train.txt')
    test_label_dict = read_train_tag('data/test_target.txt')
    train_text_dict, test_text_dict = separate_train_test_by_label(all_text_dict, train_label_dict, test_label_dict)
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_dataset = MSRDataset(train_text_dict, train_label_dict, label_map, text_embed_model_name)
    test_dataset = MSRDataset(test_text_dict, test_label_dict, label_map, text_embed_model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=16)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = XLMRobertaForSequenceClassification.from_pretrained(text_embed_model_name, num_labels=3)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_func = CrossEntropyLoss()
    model = model.to(device)
    loss_func = loss_func.to(device)
    for epoch in range(train_iteration):
        print('Training epoch {0}!'.format(epoch))
        for step, item in enumerate(tqdm(train_dataloader)):
            texts_input_ids, text_attention_mask, labels = item
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            optimizer.zero_grad()
            labels = labels.to(device)
            output = model(input_ids=texts_input_ids, attention_mask=text_attention_mask)
            loss = loss_func(output.logits, labels)
            loss.backward()
            optimizer.step()
        torch.cuda.empty_cache()
        print('Testing!')
        test_label, test_output = [], []
        for step, item in enumerate(tqdm(test_dataloader)):
            texts_input_ids, text_attention_mask, labels = item
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            test_label.extend(labels.tolist())
            output = model(input_ids=texts_input_ids, attention_mask=text_attention_mask)
            output_label = torch.argmax(output.logits, dim=1)
            print(output_label.detach().cpu().tolist())
            test_output.extend(output_label.detach().cpu().tolist())
        precision = precision_score(test_label, test_output, average='macro')
        recall = recall_score(test_label, test_output, average='macro')
        f1 = f1_score(test_label, test_output, average='macro')
        print('Precision: {0}, Recall: {1}, F1: {2}'.format(precision, recall, f1))

