
# %%

# Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dfply
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from xgboost import XGBRegressor
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# %%

# Data Wrangling
music = pd.read_csv('song_data.csv')
music = music.drop(columns=['song_name'])

# %%
X = music.drop(columns=['song_popularity'])
y = music['song_popularity']

X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.3, random_state=42)

# fit scaler on training data
norm = MinMaxScaler().fit(X_train)

# transform training data
X_train = norm.transform(X_train)

# transform testing dataabs
X_test = norm.transform(X_test)


# %%
# build a neural network using a sequential model
# input layer node size is the same as the training columns
# hidden layers: 128 node layer, 256 node layer, 64 node layer
# a single node output layer
model = Sequential()
model.add(Dense(128, input_dim=len(X_train[0]), activation='sigmoid'))
model.add(Dropout(.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))

model.add(Dense(1, activation='relu'))

model.summary()


# %%

# opt = keras.optimizers.Adam(learning_rate=0.0001)
opt = keras.optimizers.Adam()
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mse'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_mse', patience=30)

history = model.fit(X_train, y_train, epochs=2000, validation_split=.35, batch_size=20, callbacks=[early_stop],shuffle=False)
# history = model.fit(train_features, train_labels, epochs=2000, verbose=0, validation_split = .2, batch_size=tester2,
#                     callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)

hist

# h = hist
hist = hist.reset_index()
# h

def plot_history():
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error ')
    plt.plot(hist['index'], hist['mse'], label='Train Error')
    plt.plot(hist['index'], hist['val_mse'], label = 'Val Error')
    plt.legend()
    # plt.ylim([0,50])

plot_history()

predictions = np.round(model.predict(X_test),1)

predictions

result = mean_squared_error(y_test, predictions, squared=False)
result

r2 = r2_score(y_test,predictions)
r2

pred = pd.DataFrame(predictions,columns=['predictions'])
pred
pred['actual'] = y_test.tolist()
pred

pred['difference'] = pred['actual']-pred['predictions']
pred

import seaborn as sns
xlims =(0,55)
ax = sns.scatterplot(data=pred,x='actual',y='predictions')
ax.plot(xlims,xlims, color='r')
plt.show()









# %%

music.describe()


# %%
