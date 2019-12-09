import os
import sys
import tensorflow as tf

a = tf.constant(3.0, name='a')
b = tf.constant(5.0, name='b')
c = a * b

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log/", sess.graph)
    sess.run(c)
    writer.close()

export_path = './jiho'
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
# builder.add_meta_graph_and_variables(
#       sess, [tf.saved_model.tag_constants.SERVING],
#       signature_def_map={
#            'predict_images':
#                prediction_signature,
#            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                classification_signature,
#       },
#       main_op=tf.tables_initializer())
builder.save()