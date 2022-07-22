import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
A=tf.range(-100,100,4)
A
B=A+10
B
A_training=A[:40]
B_training=B[:40]
A_test=A[40:]
B_test=B[40:]
len(A_training)
plt.figure(figsize=(15,10))
plt.scatter(A_training,B_training,c='b',label="Training data")
plt.scatter(A_test,B_test,c='r',label="Testing data")
plt.legend()
def plot_predict(train_data,train_labels,test_data,test_labels,predict):
  plt.figure(figsize=(10,5))
  plt.scatter(train_data,train_labels,c="r",label="Training_data")
  plt.scatter(test_data,test_labels,c='g',label="Testing_data")
  plt.scatter(test_data,predict,c='y',label="predictions")
  plt.legend()
model.evaluate(A_test,B_test)
def mae(true,predict):
  return tf.metrics.mean_absolute_error(tf.squeeze(true),tf.squeeze(predict))
def mse(true,predict):
  return tf.metrics.mean_squared_error(tf.squeeze(true),tf.squeeze(predict))
#model1
model1=tf.keras.Sequential([tf.keras.layers.Dense(1,input_shape=[1])])
model1.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=["mae"])
model1.fit(A_training,B_training,epochs=100,verbose=1)
a1=model1.predict(A_test)
a1 , B_test
plot_predict(A_training,B_training,A_test,B_test,a1)
mae_1=mae(B_test,a1)
mse_1=mse(B_test,a1)
print(mae_1,mse_1)
#model2
model2=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=[1],name="input_layer"),tf.keras.layers.Dense(1,name="output_layer")])
model2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['mae'])
model2.fit(A_training,B_training,epochs=100,verbose=1)
a2=model2.predict(A_test)
a2,B_test
mae_2=mae(B_test,a2)
mse_2=mse(B_test,a2)
print(mae_2,mse_2)
plot_predict(A_training,B_training,A_test,B_test,a2)
#model3
model3=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=[1]),tf.keras.layers.Dense(1)])
model3.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(lr=0.01),metrics=['mae'])
model3.fit(A_training,B_training,epochs=500)
a3=model3.predict(A_test)
a3,B_test
plot_predict(A_training,B_training,A_test,B_test,a3)
mae_3=mae(B_test,a3)
mse_3=mse(B_test,a3)
print(mae_3,mse_3)
import pandas
model_results=[['model_1',mae_1.numpy(),mse_1.numpy()],['model_2',mae_2.numpy(),mse_2.numpy()],['model_3',mae_3.numpy(),mse_3.numpy()]]
all_results=pandas.DataFrame(model_results,columns=['model','mae','mse'])
all_results
