import tensorflow as tf
tf.ones([1,2,3])
tf.zeros([4,5])
import numpy as np
np.zeros(2)
#converting tensor like value to tensor
A=np.arange(1,7,dtype=np.int32)
B=tf.constant(A,shape=(2,3))

#rank 4 tensor
t1=tf.zeros([2,3,4,5])

print("Datatype of every element:",t1.dtype)
print("rank of tensor:",t1.ndim)
print("Shape of tensor:",t1.shape)
print("Elements along the 0 axis:",t1.shape[0])
print("Elements along the last axis",t1.shape[-1])
print("size of the tensor:",tf.size(t1))
print("size of the tensor:",tf.size(t1).numpy())

#indexing
t1[:1,:3,:1,:2]

t2=tf.constant([[3,6],[4,2]])
#adding a new axis
t3=t2[...,tf.newaxis]
t3
#wxpanding
t4=tf.expand_dims(t2,axis=-1)
t4
