import tensorflow as tf
from gcn import gcn
import numpy as np
import uuid
import time

sample_size = 100000
inputs_num = 3
embed_size = 2
gcn_output = 2
edge_labels_num = 5
bias_labels_num = 4
x = tf.placeholder(shape=[None, inputs_num, embed_size], dtype=tf.float32)
e = tf.placeholder(shape=[None, inputs_num, inputs_num,
                          edge_labels_num], dtype=tf.float32)
b = tf.placeholder(shape=[None, inputs_num, inputs_num,
                          bias_labels_num], dtype=tf.float32)
inputs = [x, e, b]
# nn = tf.layers.dense(x, inputs_num, activation=tf.nn.sigmoid)
nn = gcn(inputs, activation=tf.nn.sigmoid)
# nn = tf.reshape(nn, [-1, gcn_output * inputs_num])
nn = tf.Print(nn, [tf.shape(nn), nn], message="gcn result")
nn = gcn([nn, e, b], gcn_output, activation=tf.nn.sigmoid)
nn = tf.Print(nn, [tf.shape(nn), nn], message="gcn result")

nn = tf.reshape(nn, [-1, gcn_output * inputs_num])
encoded = tf.layers.dense(nn, 2, activation=tf.nn.sigmoid)
nn = tf.layers.dense(encoded, 5, activation=tf.nn.sigmoid)
nn = tf.layers.dense(nn, inputs_num * embed_size, activation=tf.nn.sigmoid)
nn = tf.reshape(nn, (-1, inputs_num, embed_size))

cost = tf.reduce_mean((nn - x)**2)
optimizer = tf.train.RMSPropOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()

tf.summary.scalar("cost", cost)
merged_summary_op = tf.summary.merge_all()
now = time.time()
with tf.Session() as sess:
    sess.run(init)
    uniq_id = "/tmp/tensorboard-layers-api/" + \
        uuid.uuid1().__str__()[:inputs_num]
    summary_writer = tf.summary.FileWriter(
        uniq_id, graph=tf.get_default_graph())
    x_vals = np.random.normal(0, 1, (sample_size, inputs_num, embed_size))
    edges = np.ones((sample_size, inputs_num, inputs_num, edge_labels_num))

    for bsm, i, n, l in np.ndindex(edges.shape):
        if n == 0:
            edges[bsm, i, n, l] = 0
    biases = np.ones((sample_size, inputs_num, inputs_num, bias_labels_num))
    for bsm, i, n, l in np.ndindex(biases.shape):
        if n == 0:
            biases[bsm, i, n, l] = 0

    for step in range(sample_size):
        _, val, summary, out = sess.run([optimizer, cost, merged_summary_op, nn],
                                   feed_dict={x: x_vals, e: edges, b: biases})
        if step % 5 == 0:
            summary_writer.add_summary(summary, step)
            print("step: {}, loss: {}".format(step, val))
            print("time", time.time() - now)
            now = time.time()
