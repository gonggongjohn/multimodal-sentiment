from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label
from baselines.xlm_train import train_xlm
import torch


if __name__ == "__main__":
    text_embed_model_name = 'xlm-roberta-base'
    train_iteration = 4
    all_text_dict = read_folder_text('../data/source')
    train_label_dict = read_train_tag('../data/train.txt')
    test_label_dict = read_train_tag('../data/test_target.txt')
    train_text_dict, test_text_dict = separate_train_test_by_label(all_text_dict, train_label_dict, test_label_dict)
    label_map = {'neutral': 0, 'negative': 1, 'positive': 2}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr1_list = [5e-6, 1e-5, 2e-5, 5e-5]
    lr2_list = [5e-5, 5e-4, 1e-3, 5e-3, 1e-2]
    warm_list = [0]
    cnt = 0
    for lr1 in lr1_list:
        for lr2 in lr2_list:
            for warm in warm_list:
                if lr1 == 1e-6 or lr2 > 1e-3:
                    if warm > 0:
                        continue
                cnt += 1
                loss_log_name = 'log/xlm_loss_{0}_{1}_{2}.log'.format(lr1, lr2, warm)
                eval_log_name = 'log/xlm_eval_{0}_{1}_{2}.log'.format(lr1, lr2, warm)
                train_xlm(device, loss_log_name, eval_log_name, text_embed_model_name, train_text_dict,
                          train_label_dict, test_text_dict, test_label_dict, label_map, 6, 16, lr1, lr2, warm)
