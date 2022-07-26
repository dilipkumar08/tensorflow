import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
data
data_onehot=pd.get_dummies(data)
x=data_onehot.drop("charges",axis=1)
x
y=data_onehot["charges"]
y   
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
tf.random.set_seed(42)
#model1
insurance_model=tf.keras.Sequential([tf.keras.layers.Dense(10),tf.keras.layers.Dense(1)])
insurance_model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.SGD(),metrics=['mae'])
insurance_model.fit(x_train,y_train,epochs=100)
insurance_model.evaluate(x_test,y_test)
#model2
insurance_model2=tf.keras.Sequential([tf.keras.layers.Dense(100),tf.keras.layers.Dense(100),tf.keras.layers.Dense(10),tf.keras.layers.Dense(1)])
insurance_model2.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(),metrics=['MAE'])
history=insurance_model2.fit(x_train,y_train,epochs=200)

insurance_model2.evaluate(x_test,y_test)

#loss curve
pd.DataFrame(history.history).plot()
plt.ylabel('loss')
plt.xlabel('epochs')
