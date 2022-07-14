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


if __name__ == "__main__":
    train_dict = read_train_tag('train.txt')
    pos_cnt, neg_cnt, neu_cnt = 0, 0, 0
    for item in train_dict:
        if train_dict[item] == 'positive':
            pos_cnt += 1
        elif train_dict[item] == 'negative':
            neg_cnt += 1
        elif train_dict[item] == 'neutral':
            neu_cnt += 1
    print('Positive: {0}, Negative: {1}, Neutral: {2}'.format(pos_cnt, neg_cnt, neu_cnt))
