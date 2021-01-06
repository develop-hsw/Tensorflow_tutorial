import tensorflow as tf
print(tf.__version__) #버전 확인

# dataset -> model -> loss값 확인 (model-Optimizer-loss 반복)
# dataset에서는 학습 x
# model내 weight, bias가 학습에 활용되는 것들. w, b는 mutable해야한다.(업데이트 가능)

# tf의 데이터 타입 2가지. 1. Variable    2. constant
t1 = tf.Variable([1, 2, 3]) # <- mutable. 따라서 model에서 사용(weight, bias), model에 들어가는 파라미터 역할.
t2 = tf.constant([1, 2, 3]) # <- immutable. 따라서 dataset에서 사용

print(t1)
print(t2)

print(type(t1))
print(type(t2))

import numpy as np

# constant
print("\n########### constant ###########")
test_list = [1, 2, 3]
test_np = np.array([1, 2, 3])

t1 = tf.constant(test_list)
t2 = tf.constant(test_np)

print(t1)
print(t2)

print(type(t1))
print(type(t2))


# Variable
print("\n########### Variable ###########")
test_list = [1, 2, 3]
test_np = np.array([1, 2, 3])

t1 = tf.Variable(test_list)
t2 = tf.Variable(test_np)

print(t1)
print(t2)

print(type(t1))
print(type(t2))


# Convert
print("\n########### Convert ###########")
t1 = tf.constant(test_list)
t2 = tf.Variable(test_list)


#t3 = tf.constant(t2) # Variable Tensor to Constant Tensor -> ERROR!
t4 = tf.Variable(t1) # constant Tensor to Variable Tensor -> OK!
print("t4 : ", t4)
#굳이 Variable -> Constant를 해야겠다면, tf.convert_to_tensor([])를 사용한다.
t1 = tf.convert_to_tensor(test_list) # -> 전부 다 Eager Tensor, 즉 constant로 바꾼다.
t5 = tf.constant(t1)

print("t1 : ", t1)
print(type(t1))
print("t5 : ", t5)
print(type(t5))

# 텐서의 attribute, method 보기. constant와 Variable의 dir 결과가 다르다.
print(dir(t1))



# 즉, 정리하자면,
# Constant == EagerTensor == immutable
# Variable == ResourceVariable == mutable
