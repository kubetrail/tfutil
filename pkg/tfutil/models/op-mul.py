import tensorflow as tf

f = tf.function(tf.multiply)

g = f.get_concrete_function(
    tf.constant(0, dtype=tf.float64),
    tf.constant(0, dtype=tf.float64),
).graph

print(g.as_graph_def())
tf.io.write_graph(g, "./proto", "op-mul.pb", as_text=False)
