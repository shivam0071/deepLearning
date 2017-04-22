import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

data = pd.read_fwf('brainBody.txt')
x_values = data[['Brain']]
y_values = data[['Body']]

train = linear_model.LinearRegression()
train.fit(x_values,y_values)

plt.scatter(x_values,y_values)
plt.plot(x_values,train.predict(x_values))
plt.show()