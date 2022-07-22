import tensorflow as tf
a=tf.constant([-10.5,-12.4])
a
#changing the datatype from 32 bit precision to 16bit precision
b=tf.cast(a,dtype=tf.float16)
b
#finding absolute tensor
tf.abs(a)
r=tf.random.Generator.from_seed(32)
r=tf.random.normal(shape=(2,3))
r
import numpy as np
ra=tf.constant(np.random.randint(0,100,size=50))
ra
ra.ndim,ra.shape,tf.size(ra)
#aggregate functions
#minimum
tf.reduce_min(ra)
#maximum
tf.reduce_max(ra)
#mean
tf.reduce_mean(ra)
#np variance
tf.constant(np.var(ra))
#np standard deviation
tf.constant(np.std(ra))
#tf standard deviation
tf.math.reduce_std(tf.cast(ra,dtype=tf.float32))
#tf variance
tf.math.reduce_variance(tf.cast(ra,dtype=tf.float32))
#variance using tfp
import tensorflow_probability as tfp
tfp.stats.variance(ra)
#positional maximum and minimum
tf.random.Generator.from_seed(42)
p=tf.random.uniform(shape=[50])
#position of min
tf.argmin(p)
#min val
p[tf.argmin(p)]
#position of max
tf.argmax(p)
#max val
p[tf.argmax(p)]
#you can also get the value using np.argmax or np.argmin and convert it to tensor
