import os
import chardet
import json

def read_train_tag(path: str):
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


def read_folder_text(path: str):
    """Parse text files in the training data folder"""
    text_dict = {}
    cnt = 0
    for item in os.scandir(path):
        if item.name[-4:] == '.txt':
            key = int(item.name[:-4])
            try:
                with open(item.path, 'r', encoding='utf-8') as f_text:
                    text = f_text.readline()
                    text_dict[key] = text
            except UnicodeDecodeError:
                try:
                    with open(item.path, 'r', encoding='cp1252') as f_text:
                        text = f_text.readline()
                        text_dict[key] = text
                except:
                    with open(item.path, 'rb') as f_text:
                        text = f_text.readline()
                    encoder = chardet.detect(text).get('encoding')
                    cnt += 1
                    print('Reading error! Filename: {0}, Guess Encoding: {1}, Error Count: {2}'.format(item.name, encoder, cnt))
    return text_dict


def dataset_translate(all_text_dict: dict, train_tag_dict: dict, test_tag_dict: dict, output_path):
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    train_path = os.path.join(output_path, 'train.json')
    dev_path = os.path.join(output_path, 'dev.json')
    test_path = os.path.join(output_path, 'test.json')
    train_list, dev_list, test_list = [], [], []
    for key in train_tag_dict:
        item_dict = {'id': str(key), 'text': all_text_dict[key].strip(), 'emotion_label': label_map[train_tag_dict[key]]}
        train_list.append(item_dict)
    with open(train_path, 'w') as f_train:
        json.dump(train_list, f_train, ensure_ascii=False)
    for key in test_tag_dict:
        item_dict = {'id': str(key), 'text': all_text_dict[key].strip(), 'emotion_label': label_map[test_tag_dict[key]]}
        dev_list.append(item_dict)
        test_list.append(item_dict)
    with open(dev_path, 'w') as f_dev:
        json.dump(test_list, f_dev, ensure_ascii=False)
    with open(test_path, 'w') as f_test:
        json.dump(test_list, f_test, ensure_ascii=False)


if __name__ == "__main__":
    all_text_dict = read_folder_text('source/')
    train_tag_dict = read_train_tag('train.txt')
    test_tag_dict = read_train_tag('test_target.txt')
    dataset_translate(all_text_dict, train_tag_dict, test_tag_dict, 'clmlf/')