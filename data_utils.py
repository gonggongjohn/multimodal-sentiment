from sklearn.metrics import precision_score, recall_score, f1_score
from PIL import Image
from torchvision import transforms
import os
import chardet


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


def write_test_output(output_dict: dict, path: str):
    with open(path, 'w') as f:
        f.write('guid,tag\n')
        for key in output_dict:
            f.write('{0},{1}\n'.format(key, output_dict[key]))


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


def separate_train_test_by_label(all_data_dict: dict, train_label_dict: dict, test_label_dict: dict):
    train_dict, test_dict = {}, {}
    for key in all_data_dict:
        if key in train_label_dict:
            train_dict[key] = all_data_dict[key]
        if key in test_label_dict:
            test_dict[key] = all_data_dict[key]
    return train_dict, test_dict


def read_img_by_id(dir_path: str, data_dict: dict, target_size: int):
    img_dict = {}
    img_transformer = transforms.Compose(
        [
            transforms.Resize(img_aligned_scale(target_size)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    for key in data_dict:
        path = os.path.join(dir_path, '{0}.jpg'.format(key))
        img = Image.open(path)
        img.load()
        img_transformed = img_transformer(img)
        img_dict[key] = img_transformed
    return img_dict


def img_aligned_scale(target_size):
    for i in range(20):
        if 2 ** i >= target_size:
            return 2 ** i
    return target_size


def evaluate_result(output_path: str, target_path: str):
    target_dict = {}
    target_vector = []
    output_vector = []
    mapping = {'negative': 0, 'neutral': 1, 'positive': 2, 'null': 4}
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
    read_folder_text('data/source')
