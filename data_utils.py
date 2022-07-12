import os
import chardet
from sklearn.metrics import precision_score, recall_score, f1_score


def read_train_tag(path):
    """Parse train.txt into a training tag dictionary"""
    tag_dict = {}
    with open(path, 'r') as f:
        f.readline()  # Skip the title line
        line = f.readline()
        while line:
            items = line.strip().split(',')
            key = int(items[0])
            value = items[1]
            tag_dict[key] = value
            line = f.readline()
    return tag_dict


def read_train_text(path):
    """Parse text files in the training data folder"""
    text_dict = {}
    cnt = 0
    for item in os.scandir(path):
        if item.name[-4:] == '.txt':
            key = item.name[:-4]
            try:
                with open(item.path, 'r', encoding='utf-8') as f_text:
                    text = f_text.readline()
            except UnicodeDecodeError:
                try:
                    with open(item.path, 'r', encoding='cp1252') as f_text:
                        text = f_text.readline()
                        print(text)
                except:
                    with open(item.path, 'rb') as f_text:
                        text = f_text.readline()
                    encoder = chardet.detect(text).get('encoding')
                    cnt += 1
                    print('Read error!', item.name, cnt)


def evaluate_result(output_path, target_path):
    target_dict = {}
    target_vector = []
    output_vector = []
    mapping = {'positive': 1, 'neutral': 2, 'negative': 3, 'null': 4}
    with open(target_path, 'r') as f:
        f.readline()  # Skip the title line
        line = f.readline()
        while line:
            items = line.strip().split(',')
            key = int(items[0])
            value = items[1]
            target_dict[key] = value
            target_vector.append(mapping[value])
            line = f.readline()
    with open(output_path, 'r') as f:
        f.readline()
        line = f.readline()
        while line:
            items = line.strip().split(',')
            key = int(items[0])
            value = items[1]
            output_vector.append(mapping[value])
            line = f.readline()
    p = precision_score(target_vector, output_vector, average='macro')
    r = recall_score(target_vector, output_vector, average='macro')
    f1 = f1_score(target_vector, output_vector, average='macro')
    return p, r, f1


if __name__ == "__main__":
    # precision, recall, f1 = evaluate_result('data/test_manual.txt', 'data/test_target.txt')
    # print(precision, recall, f1)
    read_train_text('data/source')