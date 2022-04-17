# https://www.tensorflow.org/guide/intro_to_graphs
import tensorflow as tf

f = tf.function(tf.repeat)

axis = 1
g = f.get_concrete_function(
    tf.constant([[0,0,0],[0,0,0],[0,0,0]], dtype=tf.float64),
    tf.constant([0,0,0], dtype=tf.int64),
    axis,
).graph

print(g.as_graph_def())

tf.io.write_graph(g, "./proto", "op.pb", as_text=False)
