import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("oil_production.csv")
df.drop("Unnamed: 0", axis = 1, inplace = True)
df.drop("DATEPRD",axis = 1, inplace = True)

X = df.iloc[:,0:6]
Y = df.iloc[:,6]

from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, Normalizer, MaxAbsScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

test_fraction = 0.15
val_fraction = 0.15
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = test_fraction,
 shuffle = True, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size 
                                                  = val_fraction/(1-test_fraction),
                                                   shuffle = True, random_state = 1000)
scaler_x = MinMaxScaler()
x_train = scaler_x.fit_transform(x_train.as_matrix())
x_val = scaler_x.transform(x_val.as_matrix())
x_test = scaler_x.transform(x_test.as_matrix())

scaler_y = MinMaxScaler()
y_train = scaler_y.fit_transform(y_train.reshape(-1,1))
y_val = scaler_y.transform(y_val.reshape(-1,1))
y_test = scaler_y.transform(y_test.reshape(-1,1))




model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
#model.add(Dense(2, activation='relu'))

model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size = 64, epochs = 200, verbose = 1,
 validation_data = (x_val, y_val))

prediction = model.predict(x_test)

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2
print(r2(y_val, model.predict(x_val)))
print(r2(y_test, model.predict(x_test)))

print(mse(y_val, model.predict(x_val)))
print(mse(y_test, prediction))

print(prediction.min())
#plt.plot(prediction,'r')
#plt.plot(y_test,'b')
plt.scatter(prediction, y_test, color = 'r')
plt.xlabel("Predicted oil production")
plt.ylabel("Actual oil production")
plt.title("Deep neural network with 3 hidden layer")
plt.show()




