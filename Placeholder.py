import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


sess = tf.compat.v1.Session()

xarr = [1, 2, 3, 4, 5]

# y = x + 10 구성
x = tf.compat.v1.placeholder(tf.float32)
y = x + 10

placeholder_1 = sess.run(y, feed_dict={x: xarr})

xxarr = [1, 2, 3, 4, 5]
yyarr = [10, 20, 30, 40, 50]

# z = x + y 구성
x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)
z = x+y

placeholder_2 = sess.run(z, feed_dict={x: xxarr, y: yyarr})


print(placeholder_1)
print(placeholder_2)
