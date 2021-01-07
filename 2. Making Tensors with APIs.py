# 여러가지 Tensor를 만드는 방법

import tensorflow as tf
import numpy as np

test_list = [1, 1, 1, 1, 1, 1]
t1 = tf.constant(test_list)
print(t1)

# 1이 많아지면 어쩌려고.. 뭐하러 저렇게 하냐... 할때 tf.ones(shape=())
# shape에서 x와 y의 역할 -> x: numpy.array의 갯수 y: numpy.array의 1의 갯수(차원)
t2 = tf.ones(shape=(100, ))
print(t2)
# ex) [1, 1, 1]을 100개 만들고 싶다?
t2 = tf.ones(shape=(100, 3))
print(t2)

#그와 반대로 0으로 채우는 것은?
t3 = tf.zeros(shape=(128, 128, 3)) # 128 x 128 이미지 RGB를 표현한 예시
print(t3)

PI = np.pi # numpy에서 pi값 가져오기. 이건 몰랐네?
t4 = 3*tf.ones(shape=(128, 128, 3)) # tf.ones에 스칼라 곱
print(t4)

# 근데 ones보다는 zeros를 쓰는게 좋다.

test_list = [[1, 2, 3], [4, 5, 6]]

t1 = tf.Variable(test_list)
print(t1)

t2 = tf.ones(shape=t1.shape) # t1과 똑같은 shape을 가진 ones를 만든다.
# 또는,
t2 = tf.ones_like(t1)   # t1과 똑같은 shape을 가진 ones를 만든다.

t3 = tf.zeros_like(t1)  # t1과 똑같은 shape을 가진 zeros를 만든다.

# 랜덤한 값 만들기
t1 = tf.random.normal(shape=(10, 10))
print(t1)

np.random.seed(0)
tf.random.set_seed(0)


# 그냥 tf.random.normal 만든 부분 그래프로 보여주는거.
import matplotlib.pyplot as plt
t2 = tf.random.normal(mean=3, stddev=1, shape=(1000, )) # 평균이 3, 표준편차가 1
#t2 = tf.random.uniform(shape=(1000, ), minval=-10, maxval=10)  # 다양한 분포도
#t2 = tf.random.poisson(shape=(1000, ), lam = 5)
print(t2)

fig, ax = plt.subplots(figsize=(15, 15))
ax.hist(t2.numpy(), bins=30)

ax.tick_params(labelsize=20)

# 정보를 뽑는 방법
t1 = tf.random.normal(shape=(128, 128, 3))

# shape 뽑기
print("t1.shape: ", t1.shape)

# data type 뽑기
print("t1.dtype: ", t1.dtype)

# 데이터 타입은 dtype=tf.float32로 해야한다.
t1 = tf.constant(tf.zeros(shape=(10, 2)), dtype=tf.float32)
print(t1)