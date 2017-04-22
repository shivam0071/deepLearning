import pandas as pd #to read our datasets
from sklearn import linear_model  #ML library
import matplotlib.pyplot as plt  #for ploting

#read_data
dataframe = pd.read_fwf('ex1data1.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg =linear_model.LinearRegression()
body_reg.fit(x_values,y_values)

#visulaize the results
plt.scatter(x_values,y_values)
plt.plot(x_values,body_reg.predict(x_values))
plt.show()
print(body_reg.predict(x_values))