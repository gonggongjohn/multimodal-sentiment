from data_utils import read_folder_text, read_train_tag, separate_train_test_by_label
from data_pipe import MSRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import XLMRobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from baseline_model import PureRoberta
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


