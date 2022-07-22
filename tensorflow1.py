#import TensorFlow
import tensorflow as tf
print(tf.__version__)
#first tensor tf.constant()
scalar=tf.constant(7)
scalar
scalar.ndim
#create a vector
vector=tf.constant([10,10])
vector
vector.ndim
#matrix
matrix=tf.constant([[10,7],[7,10]])
matrix
matrix.ndim
a_matrix=tf.constant([[10.,7.],[3.,4.],[5.,6.]],dtype=tf.float16)
a_matrix
a_matrix.ndim
#tensor
tensor=tf.constant([[[1,2,3],[3,4,5]],[[6,7,8],[2,4,5]],[[12,7,8],[4,7,23]]])
tensor
#variable creation
c=tf.Variable([10,7])
v=tf.constant([10,7])
c,v
#c[0]=7  error
c[0].assign(7)
c[1].assign(8)
c
v[0]
#v[0].assign(18) error cause its constant br0
#creating random tensors
r=tf.random.Generator.from_seed(42)
r=r.normal(shape=(3,2))
r
r2=tf.random.Generator.from_seed(42)
r2=tf.random.normal(shape=(3,2))
r2
#condition
r,r2, r==r2
#uniform
r3=tf.random.Generator.from_seed(29)
r3=tf.random.uniform(shape=(2,2))
r3
#shuffle tensor
ns=tf.constant([[10,7],[3,4],[2,5]])
ns.ndim
ns
tf.random.shuffle(ns,seed=42)

ar=tf.Variable([[[1,2,3],[1,4,5],[4,3,2]],[[3,4,5],[5,6,7],[4,7,5]]])
ar=tf.random.shuffle(ar,seed=42)
ar
