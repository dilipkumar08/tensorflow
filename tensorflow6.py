import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
A=tf.range(-100,100,4)
A
B=A+10
B
#visualize
plt.scatter(A,B)
A_training=A[:40]
B_training=B[:40]
A_test=A[40:]
B_test=B[40:]
len(A_training)
plt.figure(figsize=(15,10))
plt.scatter(A_training,B_training,c='b',label="Training data")
plt.scatter(A_test,B_test,c='r',label="Testing data")
plt.legend()
model=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=[1],name="input_layer"),tf.keras.layers.Dense(1,name="output_layer")],name="Model1")
model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['mae'])
model.fit(A_training,B_training,epochs=100,verbose=1)
model.summary()
B_predict=model.predict([A_test])
from tensorflow.keras.utils import plot_model
plot_model(model=model,show_shapes=True)
B_predict , B_test
def plot_predict(train_data,train_labels,test_data,test_labels,predict):
  plt.figure(figsize=(10,5))
  plt.scatter(train_data,train_labels,c="r",label="Training_data")
  plt.scatter(test_data,test_labels,c='g',label="Testing_data")
  plt.scatter(test_data,predict,c='y',label="predictions")
  plt.legend()
plot_predict(train_data=A_training,train_labels=B_training,test_data=A_test,test_labels=B_test,predict=B_predict)
