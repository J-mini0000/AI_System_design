import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


digit = datasets.load_digits()
x_train, x_test, y_train, y_test = train_test_split(digit.data, digit.target, train_size=0.6)
x_train = x_train.reshape(1078,64)
x_test = x_test.reshape(719,64)
x_train = x_train/16
x_test = x_test/16
y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)
    
n_input = 64
n_hidden1 = 2048




n_output = 10
    
mlp = Sequential()
mlp.add(Dense(units = n_hidden1, activation='relu', input_shape=(n_input,),
                         kernel_initializer='random_uniform', bias_initializer='zeros'))




    
    
mlp.add(Dense(units = n_output, activation='tanh',kernel_initializer='random_uniform',
                  bias_initializer='zeros'))
    
mlp.compile(loss='mse', optimizer = Adam(learning_rate=0.0001), metrics=['accuracy'])
hist=mlp.fit(x_train, y_train,batch_size=64, epochs=100, validation_data=(x_test,y_test),verbose=2)#32 500 96.8
    
res= mlp.evaluate(x_test,y_test,verbose=0)
    
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc ='lower right')
plt.grid()
plt.show()
    
res= mlp.evaluate(x_test,y_test,verbose=0)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'], loc ='upper right')
plt.grid()
plt.show()
    
print("Accuracy is",res[1]*100)
    
    
