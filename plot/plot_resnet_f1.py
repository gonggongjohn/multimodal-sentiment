import matplotlib.pyplot as plt

accuracy = [0.48572396438250104, 0.4991428598454905, 0.5099677095494098, 0.5160201530157438, 0.5300542373236133, 0.5054942620646651, 0.5246052875755387, 0.5255248704249756, 0.5276876250764064, 0.5180230397621702]
epoch = [i for i in range(1, 11)]
plt.plot(epoch, accuracy, marker='o', label='ResNet-50')
plt.title('F1 score on validation set With Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.legend()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.savefig('resnet_f1.png', dpi=300)
