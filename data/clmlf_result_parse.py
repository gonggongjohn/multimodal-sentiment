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

f_out = open('test_predict.txt', 'w')
f_out.write('guid,tag\n')

with open('test_result.json', 'r') as f_in:
    predict_list = json.load(f_in)
    label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
    key_dict = read_train_tag('test_without_label.txt')
    index_cnt = 0
    for key in key_dict:
        f_out.write('{0},{1}\n'.format(key, label_map[predict_list[index_cnt]]))
        index_cnt += 1
f_out.close()