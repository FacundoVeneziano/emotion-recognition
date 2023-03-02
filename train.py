import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('fer2013.csv')

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(data['pixels'], data['emotion'], test_size=0.2, random_state=42)
X_train = np.array([np.fromstring(x, dtype=int, sep=' ') for x in X_train.values])
X_test = np.array([np.fromstring(x, dtype=int, sep=' ') for x in X_test.values])
y_train = np_utils.to_categorical(y_train, 7)
y_test = np_utils.to_categorical(y_test, 7)
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32')
X_train /= 255
X_test /= 255

# Build model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# Save model
model.save('model.h5')
