from data_pipe import MSRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from baseline_model import PureRoberta
import torch


def train_xlm(device: str, loss_log_file: str, eval_log_file: str, model_name: str, train_text_dict: dict, train_label_dict: dict, test_text_dict: dict, test_label_dict: dict, label_map: dict, iteration: int, batch_size: int, lr1: float, lr2: float, warm: int):
    train_dataset = MSRDataset(train_text_dict, train_label_dict, label_map, model_name)
    test_dataset = MSRDataset(test_text_dict, test_label_dict, label_map, model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = PureRoberta(model_name, len(label_map))
    optimizer = AdamW([
        {
            'params': model.roberta.parameters(),
            'lr': lr1
        },
        {
            'params': model.dense.parameters(),
            'lr': lr2
        }
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warm,
                                                num_training_steps=len(train_dataloader) * iteration)
    loss_func = CrossEntropyLoss()
    model = model.to(device)
    loss_func = loss_func.to(device)
    f_loss_log = open(loss_log_file, 'w')
    f_eval_log = open(eval_log_file, 'w')
    print('Model: Pure-XLMRoberta, Weight: {0}'.format(model_name))
    for epoch in range(iteration):
        print('Training epoch {0}'.format(epoch + 1))
        model.train()
        for step, item in enumerate(tqdm(train_dataloader)):
            texts_input_ids, text_attention_mask, labels = item
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            model.zero_grad()
            labels = labels.to(device)
            output = model(input_ids=texts_input_ids, attention_mask=text_attention_mask)
            loss = loss_func(output, labels)
            f_loss_log.write('Step: {0},Loss: {1}\n'.format(epoch * len(train_dataloader) + step, loss.cpu().item()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
        torch.cuda.empty_cache()
        print('Testing Epoch {0}'.format(epoch + 1))
        test_label, test_output = [], []
        model.eval()
        for step, item in enumerate(tqdm(test_dataloader)):
            texts_input_ids, text_attention_mask, labels = item
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            test_label.extend(labels.tolist())
            output = model(input_ids=texts_input_ids, attention_mask=text_attention_mask)
            output_label = torch.argmax(output, dim=1)
            # print(output_label.detach().cpu().tolist())
            test_output.extend(output_label.detach().cpu().tolist())
        precision = precision_score(test_label, test_output, average='macro')
        recall = recall_score(test_label, test_output, average='macro')
        f1 = f1_score(test_label, test_output, average='macro')
        print('Precision: {0}, Recall: {1}, F1: {2}'.format(precision, recall, f1))
        f_eval_log.write('Epoch: {0},Precision: {1},Recall: {2},F1: {3}\n'.format(epoch + 1, precision, recall, f1))
        if epoch >= 2 and precision < 0.2:
            break
    f_loss_log.close()
    f_eval_log.close()