from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label, read_img_by_id, split_train_dev
from data_pipe import MSRDataset
from torch.utils.data import DataLoader
import torch
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', type=bool, default=False)
    parser.add_argument('--do_eval', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--img_scale_size', type=int, default=224)
    parser.add_argument('--text_model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--img_model_name', type=str, default='microsoft/swin-base-patch4-window7-224')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=6)
    parser.add_argument('--lr_finetune', type=float, default=1e-5)
    parser.add_argument('--lr_downstream', type=float, default=2e-3)
    parser.add_argument('--warmup_step', type=int, default=10)
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
    train_all_text_dict, test_text_dict = separate_train_test_by_label(dataset_text_dict, train_all_label_dict, test_label_dict)
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


