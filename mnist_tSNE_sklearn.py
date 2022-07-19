import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

 # Utility function to visualize the outputs of PCA and t-SNE
def mnist_tSNE_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts


mnist = load_digits()

time_start = time.time()

fashion_tsne = TSNE(random_state=123).fit_transform(mnist.data)

print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

mnist_tSNE_scatter(fashion_tsne, mnist.target)
plt.show()