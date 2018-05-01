from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.style.use('ggplot')

class Prior(object):
    def __init__(self, type):
        self.type = type

    def sample(self, shape):
        if self.type == "uniform":
            return np.random.uniform(-1.0, 1.0, shape)
        else:
            return np.random.normal(0, 1, shape)

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

def make_batches(size, batch_size):
    '''Returns a list of batch indices (tuples of indices).
    '''
    return [(i, min(size, i + batch_size)) for i in range(0, size, batch_size)]

def create_image_grid(x, img_size, tile_shape):
    assert (x.shape[0] == tile_shape[0] * tile_shape[1])
    assert (x[0].shape == img_size)

    img = np.zeros((img_size[0] * tile_shape[0] + tile_shape[0] - 1,
                    img_size[1] * tile_shape[1] + tile_shape[1] - 1,
                    3))

    for t in range(x.shape[0]):
        i, j = t // tile_shape[1], t % tile_shape[1]
        img[i * img_size[0] + i : (i + 1) * img_size[0] + i, j * img_size[1] + j : (j + 1) * img_size[1] + j] = x[t]

    return img


def disp_scatter(x, g, gen, num_gens, fig=None, ax=None):
    colors = ['darkblue', 'yellow', 'indigo', 'darkgreen', 'purple',
              'dodgerblue', 'lime', 'brown', 'darkcyan', 'deeppink']

    if ax is None:
        fig, ax = plt.subplots()

    ax.cla()
    ax.scatter(x[:, 0], x[:, 1], s=10, marker='+', color='r', alpha=0.8)
    for i in range(num_gens):
        ax.scatter(g[gen == i, 0], g[gen == i, 1], s=10, marker='o',
                   color=colors[i], alpha=0.8)
    ax.legend(["real data"] + ['gen {}'.format(i) for i in range(num_gens)])
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    return fig, ax