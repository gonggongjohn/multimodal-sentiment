from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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
    acc = accuracy_score(target_vector, output_vector)
    p = precision_score(target_vector, output_vector, average='macro')
    r = recall_score(target_vector, output_vector, average='macro')
    f1 = f1_score(target_vector, output_vector, average='macro')
    return acc, p, r, f1


if __name__ == "__main__":
    print(evaluate_result('test_zxn.txt', 'test_target.txt'))
    print(evaluate_result('test_predict.txt', 'test_target.txt'))