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


def read_target_tag(path):
    tag_dict = {}
    with open(path, 'r') as f:
        f.readline()  # Skip the title line
        line = f.readline()
        while line:
            items = line.strip().split('\t')
            key = int(items[0])
            value = items[1]
            value_split = value.split(',')
            tag_dict[key] = (value_split[0], value_split[1])
            line = f.readline()
    return tag_dict


if __name__ == "__main__":
    test_tag_dict = read_train_tag('test_without_label.txt')
    target_dict = read_target_tag('labelAll.txt')
    with open('test_target.txt', 'w') as f_out:
        f_out.write('guid,tag\n')
        for key in test_tag_dict:
            target_label = target_dict[key]
            if (target_label[0] == 'positive' and target_label[1] == 'negative') or (target_label[0] == 'negative' and target_label[1] == 'positive'):
                print('Contradict!id=', key)
            else:
                if target_label[0] == 'neutral':
                    result_label = target_label[1]
                else:
                    result_label = target_label[0]
                f_out.write('{0},{1}\n'.format(key, result_label))