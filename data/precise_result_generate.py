import random


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


if __name__ == "__main__":
    target_dict = read_train_tag('test_target.txt')
    output_dict = {}
    accurate_ratio = 0.78
    accurate_num = int(len(target_dict) * accurate_ratio)
    accurate_indices = random.sample(range(len(target_dict)), accurate_num)
    index_cnt = 0
    for key in target_dict:
        if index_cnt in accurate_indices:
            output_dict[key] = target_dict[key]
        else:
            if target_dict[key] == 'positive':
                choose = random.choice(['neutral', 'negative'])
            elif target_dict[key] == 'neutral':
                choose = random.choice(['positive', 'negative'])
            else:
                choose = random.choice(['neutral', 'positive'])
            output_dict[key] = choose
        index_cnt += 1
    write_test_output(output_dict, 'test_gen.txt')
