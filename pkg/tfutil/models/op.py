# https://www.tensorflow.org/guide/intro_to_graphs
import tensorflow as tf

f = tf.function(tf.transpose)

g = f.get_concrete_function(
    tf.constant(0, dtype=tf.float64),
).graph

print(g.as_graph_def())
