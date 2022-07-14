import matplotlib.pyplot as plt

f1_list = []
acc_list = []
with open('../log/eval/clmlf_1657786874_eval.log', 'r') as f_eval:
    line = f_eval.readline()
    while line:
        items = line.strip().split(',')
        acc = float(items[0][10:])
        f1 = float(items[3][4:])
        f1_list.append(f1)
        acc_list.append(acc)
        f_eval.readline()
        f_eval.readline()
        f_eval.readline()
        line = f_eval.readline()
epoch = [i for i in range(1, 11)]
plt.plot(epoch, f1_list, marker='o', label='PTAMSC')
plt.title('F1 score on validation set With Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('ptamsc_f1.png', dpi=300)

plt.clf()

plt.plot(epoch, acc_list, marker='o', label='PTAMSC')
plt.title('Accuracy score on validation set With Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('ptamsc_acc.png', dpi=300)

plt.clf()

loss_list = []
# step_list = []
with open('../log/loss/clmlf_1657786874_loss.log', 'r') as f_loss:
    line = f_loss.readline()
    line_cnt = 1
    while line:
        items = line.strip().split(',')
        # step = int(items[0][6:])
        loss = float(items[1][6:])
        loss_list.append(loss)
        # step_list.append(step)
        if line_cnt == 125:
            for i in range(375):
                f_loss.readline()
            line_cnt = 0
        line = f_loss.readline()
        line_cnt += 1
step_list = [i for i in range(1, 1251)]
plt.plot(step_list, loss_list, label='PTAMSC')
plt.title('Training Loss with Steps(Batch Size=32)')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('ptamsc_loss.png', dpi=300)
