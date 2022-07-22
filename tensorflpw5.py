import tensorflow as tf
X=tf.constant([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0,19.0])
Y=tf.constant([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0,29.0])
model=tf.keras.Sequential([tf.keras.layers.Dense(100,activation=None),
                          tf.keras.layers.Dense(1)])
model.compile(loss=tf.keras.losses.mae,optimizer=tf.keras.optimizers.Adam(lr=0.06),metrics=['mae'])
model.fit(tf.expand_dims(X,axis=1),Y,epochs=300)
model.predict([30])
