import matplotlib.pyplot as plt

accuracy = [0.59556, 0.64222, 0.69222, 0.70222, 0.71778, 0.69036, 0.73111, 0.73556]
epoch = [i for i in range(1, 8)]
plt.plot(epoch, accuracy, marker='o', label='IT-Fuse')
plt.title('F1 score on validation set With Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 score')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('fuse_accuracy.png', dpi=300)
