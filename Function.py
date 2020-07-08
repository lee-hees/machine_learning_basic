import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sess = tf.compat.v1.Session()

a = tf.constant(17)
b = tf.constant(5)

# 더하기
c= tf.add(a, b)
C = sess.run(c)
print("더하기 :",C)

# 빼기
c = tf.subtract(a, b)
C = sess.run(c)
print("빼기 :",C)

# 곱하기
c = tf.multiply(a, b)
C = sess.run(c)
print("곱하기 :",C)

# 나누기
c = tf.truediv(a, b)
C = sess.run(c)
print("나누기 :",C)

# 나머지
c = tf.math.mod(a, b)
C = sess.run(c)
print("나머지 :",C)

# 절대값
c = tf.abs(-a)
C = sess.run(c)
print("절대값 :",C)

a = tf.constant(17.0)
b = tf.constant(5.0)

# 음수
c = tf.negative(a)
C = sess.run(c)
print("음수 :",C)

# 부호
c = tf.sign(a)
C = sess.run(c)
print("부호 :",C)

# 제곱
c = tf.square(a)
C = sess.run(c)
print("제곱 :",C)

# 제곱근
c = tf.sqrt(a)
C = sess.run(c)
print("제곱근 :",C)

# 거듭제곱
c = tf.pow(a,3)
C = sess.run(c)
print("거듭제곱 :",C)

# 더 큰 값
c = tf.maximum(a, b)
C = sess.run(c)
print("더 큰 값 :",C)

# 더 작은 값
c = tf.minimum(a, b)
C = sess.run(c)
print("더 작은 값 :",C)

# 지수 값
c = tf.exp(a)
C = sess.run(c)
print("지수 값 :",C)

# 로그 값
c = tf.math.log(a)
C = sess.run(c)
print("로그 값 :",C)

# 사인 값
c = tf.sin(a)
C = sess.run(c)
print("사인 값 :",C)

# 코사인 값
c = tf.cos(a)
C = sess.run(c)
print("코사인 값 :",C)
