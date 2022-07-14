from data_utils import read_train_tag, read_img_by_id
from swin_train import train_swin
from data_pipe import MSRImageOnlyDataset
from baseline_model import PureSwinTransformer
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score


if __name__ == "__main__":
    model_weight_name = 'microsoft/swin-base-patch4-window7-224'
    iteration = 6
    train_label_dict = read_train_tag('data/train.txt')
    test_label_dict = read_train_tag('data/test_target.txt')
    train_img_dict = read_img_by_id('data/source/', train_label_dict, 224)
    test_img_dict = read_img_by_id('data/source/', test_label_dict, 224)
    label_map = {'neutral': 0, 'negative': 1, 'positive': 2}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr1 = 2e-5
    lr2 = 2e-3
    warmup = 10
    train_swin(device, 'log/loss/swin_loss_{0}_{1}_{2}.log'.format(lr1, lr2, warmup), 'log/eval/swin_eval_{0}_{1}_{2}.log'.format(lr1, lr2, warmup), model_weight_name, train_img_dict, train_label_dict, test_img_dict, test_label_dict, label_map, 6, 16, lr1, lr2, warmup)

