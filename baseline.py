from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label, read_img_by_id, split_train_dev
from data_pipe import MSRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from baseline_model import PureRoberta
from xlm_train import train_xlm
import torch
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bert')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--img_scale_size', type=int, default=224)
    parser.add_argument('--text_model_name', type=str, default='bert-base-cased')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Path and params construction
    dataset_source_path = os.path.join(args.data_path, 'source/')
    train_label_path = os.path.join(args.data_path, 'train.txt')
    test_item_path = os.path.join(args.data_path, 'test_without_label.txt')
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Preparing dataset
    dataset_text_dict = read_folder_text(dataset_source_path)
    train_all_label_dict = read_train_tag(train_label_path)
    test_label_dict = read_train_tag(test_item_path)
    train_all_text_dict, test_text_dict = separate_train_test_by_label(dataset_text_dict, train_all_label_dict,
                                                                       test_label_dict)
    train_all_img_dict = read_img_by_id(dataset_source_path, train_all_label_dict, args.img_scale_size)
    test_img_dict = read_img_by_id(dataset_source_path, test_label_dict, args.img_scale_size)
    train_text_dict, eval_text_dict, train_img_dict, eval_img_dict, train_label_dict, eval_label_dict = split_train_dev(
        train_all_text_dict, train_all_img_dict, train_all_label_dict)
    train_dataset = MSRDataset(train_text_dict, train_img_dict, train_label_dict, label_map, args.text_model_name)
    eval_dataset = MSRDataset(eval_text_dict, eval_img_dict, eval_label_dict, label_map, args.text_model_name)
    test_dataset = MSRDataset(test_text_dict, test_img_dict, test_label_dict, label_map, args.text_model_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)