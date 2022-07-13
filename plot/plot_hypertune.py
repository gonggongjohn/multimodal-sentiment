import matplotlib.pyplot as plt
import os

'''
for item in os.scandir('../log/eval/hyper/'):
    name = item.name
    if name[-4:] == '.log':
        name_items = name[:-4].split('_')
        lr1 = name_items[2]
        lr2 = name_items[3]
        warm = name_items[4]
'''

path = '../log/eval/hyper/'
lr1_list = [0, 10, 20, 50]
epoch = [i for i in range(1, 7)]
f1_list = []
for item in lr1_list:
    with open(path + 'xlm_eval_2e-06_0.001_{0}.log'.format(item), 'r') as f:
        line = f.readline()
        while line:
            f1_item = line.strip().split(',')[3]
            f1_list.append(float(f1_item[4:]))
            line = f.readline()
        if len(f1_list) < 6:
            while len(f1_list) < 6:
                f1_list.append(f1_list[-1])
        plt.plot(epoch, f1_list, label='warmup={0}'.format(item))
        f1_list = []

plt.title('Classification performance with different warmup(lr1=2e-6, lr2=1e-3)')
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('hypertune_warmup.png', dpi=300)
