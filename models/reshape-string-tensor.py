import tensorflow as tf


def reshape(x, dim):
    y = tf.reshape(x, shape=dim)
    return y


f = tf.function(reshape)

g = f.get_concrete_function(
    tf.ragged.constant(
        [],
        dtype=tf.string,
    ),
    tf.constant([2,2]),
).graph

tf.io.write_graph(g, "./proto", "reshape-string-tensor.pb", as_text=False)