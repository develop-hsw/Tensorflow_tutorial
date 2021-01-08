# 텐서를 이용한 연산
# 크게 numpy와 다르지 않음.

import tensorflow as tf
import numpy as np

# 텐서는 그냥 각 인덱스 요소끼리 더해준다.
t1 = tf.constant([1, 2, 3])
t2 = tf.constant([10, 20, 30])
print(t1+t2)

# 파이썬은 concat처럼 리스트 뒤에 append 된다.
python_list1 = [1, 2, 3]
python_list2 = [10, 20, 30]
print(python_list1+python_list2)


print(t1 + t2)
print(t1 - t2)
print(t1 * t2) # matmul
print(t1 / t2)
print(t1 % t2)
print(t1 // t2)

t1 = tf.random.normal(shape=(3, 4), mean=0, stddev=5) # normal이라서 integer 출력
t2 = tf.random.normal(shape=(3, 4), mean=0, stddev=5)

# 캐스팅 tf.cast
t1 = tf.cast(t1, dtype=tf.int16)
t2 = tf.cast(t2, dtype=tf.int16)

print(t1.numpy())
print(t2.numpy())

print(t1+t2)

t3 = t1 + t2

print(t1.numpy(), '\n')
print(t2.numpy(), '\n')
print(t3.numpy(), '\n')