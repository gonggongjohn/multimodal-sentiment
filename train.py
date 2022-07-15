from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch


def train_ptamsc(device: str, loss_log_file: str, eval_log_file: str, model, train_dataloader: DataLoader, eval_dataloader: DataLoader, iteration: int, lr_finetune: float, lr_downstream: float, warmup: int):
    optimizer = AdamW([
        {
            'params': model.text_embedding.roberta.parameters(),
            'lr': lr_finetune
        },
        {
            'params': model.img_embedding.swin.parameters(),
            'lr': lr_finetune
        },
        {
            'params': model.text_embedding.aligner.parameters(),
            'lr': lr_downstream
        },
        {
            'params': model.img_embedding.aligner.parameters(),
            'lr': lr_downstream
        },
        {
            'params': model.fuser.parameters(),
            'lr': lr_downstream
        }
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup,
                                                num_training_steps=len(train_dataloader) * iteration)
    loss_func = CrossEntropyLoss()
    model = model.to(device)
    loss_func = loss_func.to(device)
    f_loss_log = open(loss_log_file, 'w')
    f_eval_log = open(eval_log_file, 'w')
    print('Model: PTAMSC, Text weight: {0}, Image weight: {1}'.format(model.text_backbone_name, model.img_backbone_name))
    for epoch in range(iteration):
        print('Training epoch {0}'.format(epoch + 1))
        model.train()
        for step, item in enumerate(tqdm(train_dataloader)):
            texts_input_ids, text_attention_mask, imgs, labels = item
            pixels = imgs['pixel_values'][0]
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            pixels = pixels.to(device)
            model.zero_grad()
            labels = labels.to(device)
            output = model(texts_input_ids, text_attention_mask, pixels)
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
        for step, item in enumerate(tqdm(eval_dataloader)):
            texts_input_ids, text_attention_mask, imgs, labels = item
            pixels = imgs['pixel_values'][0]
            texts_input_ids = texts_input_ids.to(device)
            text_attention_mask = text_attention_mask.to(device)
            pixels = pixels.to(device)
            test_label.extend(labels.tolist())
            output = model(texts_input_ids, text_attention_mask, pixels)
            output_label = torch.argmax(output, dim=1)
            test_output.extend(output_label.detach().cpu().tolist())
        accuracy = accuracy_score(test_label, test_output)
        precision = precision_score(test_label, test_output, average='macro')
        recall = recall_score(test_label, test_output, average='macro')
        f1 = f1_score(test_label, test_output, average='macro')
        print('Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}'.format(accuracy, precision, recall, f1))
        f_eval_log.write(
            'Epoch: {0},Accuracy: {1},Precision: {2},Recall: {3},F1: {4}\n'.format(epoch + 1, accuracy, precision,
                                                                                   recall, f1))
    f_loss_log.close()
    f_eval_log.close()
