import numpy as np
import matplotlib.pyplot as plt

from fkp_input import load_train_set

datasets = load_train_set(valid=0.0)

for i, img in enumerate(datasets.train.images):
	plt.figure()
	plt.imshow(np.reshape(img, (96, 96)), cmap='gray')
	plt.axis('off')
	plt.savefig('faces/Img%04d.png' % i)
