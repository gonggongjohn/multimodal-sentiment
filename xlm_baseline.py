from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label
from xlm_train import train_xlm
import torch


if __name__ == "__main__":
    text_embed_model_name = 'xlm-roberta-base'
    train_iteration = 4
    all_text_dict = read_folder_text('data/source')
    train_label_dict = read_train_tag('data/train.txt')
    test_label_dict = read_train_tag('data/test_target.txt')
    train_text_dict, test_text_dict = separate_train_test_by_label(all_text_dict, train_label_dict, test_label_dict)
    label_map = {'neutral': 0, 'negative': 1, 'positive': 2}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr1 = 1e-5
    lr2 = 2e-3
    warm = 10
    loss_log_name = 'log/loss/xlm_loss_{0}_{1}_{2}.log'.format(lr1, lr2, warm)
    eval_log_name = 'log/eval/xlm_eval_{0}_{1}_{2}.log'.format(lr1, lr2, warm)
    train_xlm(device, loss_log_name, eval_log_name, text_embed_model_name, train_text_dict, train_label_dict, test_text_dict, test_label_dict, label_map, 6, 16, lr1, lr2, warm)


