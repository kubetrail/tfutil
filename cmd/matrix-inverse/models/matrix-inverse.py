import tensorflow as tf

f = tf.function(tf.linalg.inv)
g = f.get_concrete_function(tf.constant([[1.,2.],[3.,4.]], dtype=tf.double)).graph
print(g.as_graph_def())
tf.io.write_graph(g, "./", "graph.pb", as_text=False)
