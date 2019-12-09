import tensorflow as tf

with tf.Session() as sess:
    model_filename ='./jiho/saved_model.pb'
    with tf.io.gfile.GFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='./log'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)