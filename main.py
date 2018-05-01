from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import pickle
import argparse
import numpy as np
import tensorflow as tf

from models import MGAN

FLAGS = None


def main(_):
    tmp = pickle.load(open("data/cifar10_train.pkl", "rb"))
    x_train = tmp['data'].astype(np.float32).reshape([-1, 32, 32, 3]) / 127.5 - 1.
    model = MGAN(
        num_z=FLAGS.num_z,
        beta=FLAGS.beta,
        num_gens=FLAGS.num_gens,
        d_batch_size=FLAGS.d_batch_size,
        g_batch_size=FLAGS.g_batch_size,
        z_prior=FLAGS.z_prior,
        learning_rate=FLAGS.learning_rate,
        img_size=(32, 32, 3),
        num_conv_layers=FLAGS.num_conv_layers,
        num_gen_feature_maps=FLAGS.num_gen_feature_maps,
        num_dis_feature_maps=FLAGS.num_dis_feature_maps,
        num_epochs=FLAGS.num_epochs,
        sample_fp="samples/samples_{epoch:04d}.png",
        sample_by_gen_fp="samples_by_gen/samples_{epoch:04d}.png",
        random_seed=6789)
    model.fit(x_train)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_z', type=int, default=100,
                        help='Number of latent units.')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='Diversity parameter beta.')
    parser.add_argument('--num_gens', type=int, default=10,
                        help='Number of generators.')
    parser.add_argument('--d_batch_size', type=int, default=64,
                        help='Minibatch size for the discriminator.')
    parser.add_argument('--g_batch_size', type=int, default=12,
                        help='Minibatch size for the generators.')
    parser.add_argument('--z_prior', type=str, default="uniform",
                        help='Prior distribution of the noise (uniform/gaussian).')
    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning rate.')
    parser.add_argument('--num_conv_layers', type=int, default=3,
                        help='Number of convolutional layers.')
    parser.add_argument('--num_gen_feature_maps', type=int, default=128,
                        help='Number of feature maps of Generator.')
    parser.add_argument('--num_dis_feature_maps', type=int, default=128,
                        help='Number of feature maps of Discriminator.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
