from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from functools import partial

import os
import numpy as np
import tensorflow as tf
from ops import lrelu, linear, conv2d, deconv2d
from utils import make_batches, Prior, conv_out_size_same, create_image_grid

batch_norm = partial(tf.contrib.layers.batch_norm,
                     decay=0.9,
                     updates_collections=None,
                     epsilon=1e-5,
                     scale=True)


class MGAN(object):
    """Mixture Generative Adversarial Nets
    """

    def __init__(self,
                 model_name='MGAN',
                 beta=1.0,
                 num_z=128,
                 num_gens=4,
                 d_batch_size=64,
                 g_batch_size=32,
                 z_prior="uniform",
                 same_input=True,
                 learning_rate=0.0002,
                 img_size=(32, 32, 3),  # (height, width, channels)
                 num_conv_layers=3,
                 num_gen_feature_maps=128,  # number of feature maps of generator
                 num_dis_feature_maps=128,  # number of feature maps of discriminator
                 sample_fp=None,
                 sample_by_gen_fp=None,
                 num_epochs=25000,
                 random_seed=6789):
        self.beta = beta
        self.num_z = num_z
        self.num_gens = num_gens
        self.d_batch_size = d_batch_size
        self.g_batch_size = g_batch_size
        self.z_prior = Prior(z_prior)
        self.same_input = same_input
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.img_size = img_size
        self.num_conv_layers = num_conv_layers
        self.num_gen_feature_maps = num_gen_feature_maps
        self.num_dis_feature_maps = num_dis_feature_maps
        self.sample_fp = sample_fp
        self.sample_by_gen_fp = sample_by_gen_fp
        self.random_seed = random_seed

    def _init(self):
        self.epoch = 0

        # TensorFlow's initialization
        self.tf_graph = tf.Graph()
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.log_device_placement = False
        self.tf_config.allow_soft_placement = True
        self.tf_session = tf.Session(config=self.tf_config, graph=self.tf_graph)

        np.random.seed(self.random_seed)
        with self.tf_graph.as_default():
            tf.set_random_seed(self.random_seed)

    def _build_model(self):
        arr = np.array([i // self.g_batch_size for i in range(self.g_batch_size * self.num_gens)])
        d_mul_labels = tf.constant(arr, dtype=tf.int32)

        self.x = tf.placeholder(tf.float32, [None,
                                             self.img_size[0], self.img_size[1], self.img_size[2]],
                                name="real_data")
        self.z = tf.placeholder(tf.float32, [self.g_batch_size * self.num_gens, self.num_z], name='noise')

        # create generator G
        self.g = self._create_generator(self.z)

        # create sampler to generate samples
        self.sampler = self._create_generator(self.z, train=False, reuse=True)

        # create discriminator D
        d_bin_x_logits, d_mul_x_logits = self._create_discriminator(self.x)
        d_bin_g_logits, d_mul_g_logits = self._create_discriminator(self.g, reuse=True)

        # define loss functions
        self.d_bin_x_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_x_logits, labels=tf.ones_like(d_bin_x_logits)),
            name='d_bin_x_loss')
        self.d_bin_g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_g_logits, labels=tf.zeros_like(d_bin_g_logits)),
            name='d_bin_g_loss')
        self.d_bin_loss = tf.add(self.d_bin_x_loss, self.d_bin_g_loss, name='d_bin_loss')
        self.d_mul_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=d_mul_g_logits, labels=d_mul_labels),
            name="d_mul_loss")
        self.d_loss = tf.add(self.d_bin_loss, self.d_mul_loss, name="d_loss")

        self.g_bin_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_bin_g_logits, labels=tf.ones_like(d_bin_g_logits)),
            name="g_bin_loss")
        self.g_mul_loss = tf.multiply(self.beta, self.d_mul_loss, name='g_mul_loss')
        self.g_loss = tf.add(self.g_bin_loss, self.g_mul_loss, name="g_loss")

        # create optimizers
        self.d_opt = self._create_optimizer(self.d_loss, scope='discriminator',
                                            lr=self.learning_rate)
        self.g_opt = self._create_optimizer(self.g_loss, scope='generator',
                                            lr=self.learning_rate)

    def _create_generator(self, z, train=True, reuse=False, name="generator"):
        out_size = [(conv_out_size_same(self.img_size[0], 2),
                     conv_out_size_same(self.img_size[1], 2),
                     self.num_gen_feature_maps)]
        for i in range(self.num_conv_layers - 1):
            out_size = [(conv_out_size_same(out_size[0][0], 2),
                         conv_out_size_same(out_size[0][1], 2),
                         out_size[0][2] * 2)] + out_size

        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            z_split = tf.split(z, self.num_gens, axis=0)
            h0 = []
            for i, var in enumerate(z_split):
                h0.append(tf.nn.relu(batch_norm(linear(var, out_size[0][0] * out_size[0][1] * out_size[0][2],
                                                       scope='g_h0_linear{}'.format(i), stddev=0.02),
                                                is_training=train,
                                                scope="g_h0_bn{}".format(i)),
                                     name="g_h0_relu{}".format(i)))

            h = []
            for var in h0:
                h.append(tf.reshape(var, [self.g_batch_size, out_size[0][0], out_size[0][1], out_size[0][2]]))
            h = tf.concat(h, axis=0, name="g_h0_relu")

            for i in range(1, self.num_conv_layers):
                h = tf.nn.relu(
                    batch_norm(
                        deconv2d(h,
                                 [self.g_batch_size  * self.num_gens, out_size[i][0], out_size[i][1], out_size[i][2]],
                                 stddev=0.02, name="g_h{}_deconv".format(i)),
                        is_training=train,
                        center=False,
                        scope="g_h{}_bn".format(i)),
                    name="g_h{}_relu".format(i))

            g_out = tf.nn.tanh(
                deconv2d(h,
                         [self.g_batch_size * self.num_gens, self.img_size[0], self.img_size[1], self.img_size[2]],
                         stddev=0.02, name="g_out_deconv"),
                name="g_out_tanh")
            return g_out

    def _create_discriminator(self, x, train=True, reuse=False, name="discriminator"):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            h = x
            for i in range(self.num_conv_layers):
                h = lrelu(batch_norm(conv2d(h, self.num_dis_feature_maps * (2 ** i),
                                            stddev=0.02, name="d_h{}_conv".format(i)),
                                     is_training=train,
                                     scope="d_bn{}".format(i)))

            dim = h.get_shape()[1:].num_elements()
            h = tf.reshape(h, [-1, dim])
            d_bin_logits = linear(h, 1, scope='d_bin_logits')
            d_mul_logits = linear(h, self.num_gens, scope='d_mul_logits')
        return d_bin_logits, d_mul_logits

    def _create_optimizer(self, loss, scope, lr):
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
        opt = tf.train.AdamOptimizer(lr, beta1=0.5)
        grads = opt.compute_gradients(loss, var_list=params)
        train_op = opt.apply_gradients(grads)
        return train_op

    def fit(self, x):
        if (not hasattr(self, 'epoch')) or self.epoch == 0:
            self._init()
            with self.tf_graph.as_default():
                self._build_model()
                self.tf_session.run(tf.global_variables_initializer())

        num_data = x.shape[0] - x.shape[0] % self.d_batch_size
        batches = make_batches(num_data, self.d_batch_size)
        best_is = 0.0
        while (self.epoch < self.num_epochs):
            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                batch_size = batch_end - batch_start

                x_batch = x[batch_start:batch_end]
                if self.same_input:
                    z_batch = self.z_prior.sample([self.g_batch_size, self.num_z]).astype(np.float32)
                    z_batch = np.vstack([z_batch] * self.num_gens)
                else:
                    z_batch = self.z_prior.sample([self.g_batch_size * self.num_gens, self.num_z]).astype(np.float32)

                # update discriminator D
                d_bin_loss, d_mul_loss, d_loss, _ = self.tf_session.run(
                    [self.d_bin_loss, self.d_mul_loss, self.d_loss, self.d_opt],
                    feed_dict={self.x: x_batch, self.z: z_batch})

                # update generator G
                g_bin_loss, g_mul_loss, g_loss, _ = self.tf_session.run(
                    [self.g_bin_loss, self.g_mul_loss, self.g_loss, self.g_opt],
                    feed_dict={self.z: z_batch})

            self.epoch += 1
            print("Epoch: [%4d/%4d] d_bin_loss: %.5f, d_mul_loss: %.5f, d_loss: %.5f,"
                  " g_bin_loss: %.5f, g_mul_loss: %.5f, g_loss: %.5f" % (self.epoch, self.num_epochs,
                                d_bin_loss, d_mul_loss, d_loss, g_bin_loss, g_mul_loss, g_loss))
            self._samples(self.sample_fp.format(epoch=self.epoch+1))
            self._samples_by_gen(self.sample_by_gen_fp.format(epoch=self.epoch+1))

    def _generate(self, num_samples=100):
        sess = self.tf_session
        batch_size = self.g_batch_size * self.num_gens
        num = ((num_samples - 1) // batch_size + 1) * batch_size
        z = self.z_prior.sample([num, self.num_z]).astype(np.float32)
        x = np.zeros([num, self.img_size[0], self.img_size[1], self.img_size[2]],
                     dtype=np.float32)
        batches = make_batches(num, batch_size)
        for batch_idx, (batch_start, batch_end) in enumerate(batches):
            z_batch = z[batch_start:batch_end]
            x[batch_start:batch_end] = sess.run(self.sampler,
                                                feed_dict={self.z: z_batch})
        idx = np.random.permutation(num)[:num_samples]
        x = (x[idx] + 1.0) / 2.0
        return x

    def _samples(self, filepath, tile_shape=(10, 10)):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        num_samples = tile_shape[0] * tile_shape[1]
        x = self._generate(num_samples)
        imgs = create_image_grid(x, img_size=self.img_size, tile_shape=tile_shape)
        import scipy.misc
        scipy.misc.imsave(filepath, imgs)

    def _samples_by_gen(self, filepath):
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))

        num_samples = self.num_gens * 10
        tile_shape = (self.num_gens, 10)

        sess = self.tf_session
        img_per_gen = num_samples // self.num_gens
        x = np.zeros([num_samples, self.img_size[0], self.img_size[1], self.img_size[2]],
                     dtype=np.float32)
        for i in range(0, img_per_gen, self.g_batch_size):
            z_batch = self.z_prior.sample([self.g_batch_size * self.num_gens, self.num_z]).astype(np.float32)
            samples = sess.run(self.sampler, feed_dict={self.z: z_batch})

            for gen in range(self.num_gens):
                x[gen * img_per_gen + i:gen * img_per_gen + min(i + self.g_batch_size, img_per_gen)] = \
                    samples[
                    gen * self.g_batch_size:gen * self.g_batch_size + min(self.g_batch_size, img_per_gen)]

        x = (x + 1.0) / 2.0
        imgs = create_image_grid(x, img_size=self.img_size, tile_shape=tile_shape)
        import scipy.misc
        scipy.misc.imsave(filepath, imgs)
