from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label, read_img_by_id
from data_pipe import MSRImageOnlyDataset
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Linear
from torch.optim.adamw import AdamW
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def train_resnet(device: str, loss_log_file: str, eval_log_file: str, train_img_dict: dict, train_label_dict: dict, test_img_dict: dict, test_label_dict: dict, label_map: dict, iteration: int, batch_size: int, lr2: float):
    train_dataset = MSRImageOnlyDataset(train_img_dict, train_label_dict, label_map)
    test_dataset = MSRImageOnlyDataset(test_img_dict, test_label_dict, label_map)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model = resnet50(pretrained=True)
    del model.fc
    model.add_module('fc', Linear(2048, 3))
    loss_func = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr2)
    model = model.to(device)
    loss_func = loss_func.to(device)
    print('Model: ResNet-50')
    for epoch in range(iteration):
        print('Training epoch {0}'.format(epoch + 1))
        model.train()
        for step, item in enumerate(tqdm(train_dataloader)):
            imgs, labels = item
            model.zero_grad()
            imgs = imgs.to(device)
            labels = labels.to(device)
            output = model(imgs)
            loss = loss_func(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        torch.cuda.empty_cache()
        print('Testing Epoch {0}'.format(epoch + 1))
        test_label, test_output = [], []
        model.eval()
        for step, item in enumerate(tqdm(test_dataloader)):
            imgs, labels = item
            imgs = imgs.to(device)
            test_label.extend(labels.tolist())
            output = model(imgs)
            output_label = torch.argmax(output, dim=1)
            test_output.extend(output_label.detach().cpu().tolist())
        accuracy = accuracy_score(test_label, test_output)
        precision = precision_score(test_label, test_output, average='macro')
        recall = recall_score(test_label, test_output, average='macro')
        f1 = f1_score(test_label, test_output, average='macro')
        print('Accuracy: {0}, Precision: {1}, Recall: {2}, F1: {3}'.format(accuracy, precision, recall, f1))

