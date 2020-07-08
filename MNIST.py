#%%

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#%%

# 신경망 구성

X = tf.compat.v1.placeholder(tf.float32, [None, 784])
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random.normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random.normal([256, 64], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random.normal([64, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(0.01).minimize(cost)


#%%

# 신경망 학습

init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(1):
    total_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={
                               X: batch_x, Y: batch_y})
        total_cost += cost_val
    print('반복:', '%04d' % (epoch+1), '평균 손실갑:',
          '{:.4f}'.format(total_cost/total_batch))
print('학습완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))


#%%

# 결과 확인

labels = sess.run(model,
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels})

fig=plt.figure()

fig.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=1,hspace=1)
for i in range(20):
    subplot=fig.add_subplot(4,5,i+1)
    subplot.set_xticks([])
    subplot.set_yticks([])

    subplot.set_title('%d'%np.argmax(labels[i]))
    subplot.imshow(mnist.test.images[i].reshape((28,28)),cmap=plt.cm.gray_r)
plt.show()


# %%
