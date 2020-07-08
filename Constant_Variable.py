import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess=tf.compat.v1.Session()

a = tf.constant(1)
b = tf.constant(2)
c = tf.add(a, b)

Constant=sess.run(c)

aa = tf.Variable(10)
bb = tf.Variable(20)
cc = tf.multiply(aa, bb)
init = tf.compat.v1.global_variables_initializer()

sess.run(init)
Variable=sess.run(cc)

print(Constant)
print(Variable)
