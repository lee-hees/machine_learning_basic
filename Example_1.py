import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]

W = tf.Variable(tf.compat.v1.random.uniform([1], -100, 100))
b = tf.Variable(tf.compat.v1.random.uniform([1], -100, 100))

X = tf.compat.v1.placeholder(tf.float32)
Y = tf.compat.v1.placeholder(tf.float32)
H = W*X+b

cost = tf.reduce_mean(tf.square(H-Y))
a = tf.Variable(0.01)

optimizer = tf.compat.v1.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)
init = tf.compat.v1.global_variables_initializer()

sess = tf.compat.v1.Session()
sess.run(init)

for i in range(5000+1):
    sess.run(train, feed_dict={X: xData, Y: yData})
    if i % 500 == 0:
        print(i, sess.run(cost, feed_dict={ X: xData, Y: yData}), sess.run(W), sess.run(b))

print(sess.run(H, feed_dict={X: [8]}))













