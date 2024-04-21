# Example 1-1 in "Hands-on Machine Learning with Scikit-Learn, Keras, and Tensor Flow (Ed.3) by Aurélien Geron"

# The book author has a nice Jupyter Notebook in the githb, however, I decide to code the example again for my learning purpose

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# get the data
# data_rootpath = "https://github.com/ageron/data/raw/main" # this does not work for me, from my computer
data_localrootpath = "data\\lifesat.csv"
lifesat = pd.read_csv(data_localrootpath)
print(lifesat)

# extract feature (X) and label (y)
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# visualize the data to get some intiution
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# now let us do the ML!
model = LinearRegression()
model.fit(X, y)
X_test = [[82_807.6]]
print("Based on LinearRegression, the predicted Life Satisfaction for Singapore is", model.predict(X_test)) # ~9.36249...

from sklearn.neighbors import KNeighborsRegressor
KNmodel = KNeighborsRegressor(n_neighbors=3)
KNmodel.fit(X, y)
print("Based on KNeighborsRegressor, the predicted Life Satisfaction for Singapore is", KNmodel.predict(X_test)) # 7.3