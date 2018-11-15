"""
LSTM
"""

import pandas as pd
import numpy as np

# whatever you do to this model, it will do it in sequence
# build the model in the sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM


# import the csv - put header so you don't get the name of the column
dataset = pd.read_csv('Classification.csv', header=0)

# get columns
columns = ['JobSatisfaction', 'Age', 'DistanceFromHome', 'YearsInCurrentRole']

# only get the columns we want
df = dataset[columns]
# insert 0 for non numbers
df.fillna(0, inplace=True)

# make an x-axis with a numy rray
X = np.array(df.iloc[:, 1:])
# y = job satisfaction
y = np.array(df['JobSatisfaction'])

# have 1470 rows and 3 columns - age, distance from home and years of experience
print(X.shape)


# LSTM Requires time step
# reshape the X row and take the 3 columns but the first
X = X.reshape(X.shape[0], 1, X.shape[1])

# do everything in sequence now
model = Sequential()
# LSTM units = neurons/units; batch_size= the amt you want to throw at it
model.add(LSTM(50, batch_size=1, input_shape=(X.shape[1], X.shape[2])))
# add a dense layer
model.add(Dense(1))

# Compile; we can get the loss function from here
# Error - absolute error, mean squared error...etc
model.compile(loss='mae', optimizer='adam')

# have a model that compile but hasn't fit
# epoch = how many time it should run through the model
# take longer so run faster with CPU
# verbose = show you what's going on in your models
model.fit(X, y, epochs=1, batch_size=1, verbose=1)

# we now have a trained model - need to test now for validation
# make numpy out of Multidimensional arrays
X_test = np.array([[41, 1, 4]])
# reshape
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# now we can get our predicted values

predicted = model.predict(X_test)

print(predicted[0][0])

# Crappy data => not enough amount of data
