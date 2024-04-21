import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# let me load the csv as a dataframe
housing_df = pd.read_csv("data\\housing.csv")
# print(housing_df.head())
# print(housing_df.tail())
print(housing_df.info())

# oh! pandas would have loaded all columns as float... convert value of columns that are supposed to be int
housing_df[["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]] = \
                    housing_df[["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]].fillna(0.0).astype(int)
# oho! convert default 'object' type to string
# housing_df["ocean_proximity"] = housing_df["ocean_proximity"].astype("string")

enc = OneHotEncoder(handle_unknown="ignore")
encoded = enc.fit_transform(housing_df[["ocean_proximity"]])
encoded_df = pd.DataFrame(encoded.toarray(), columns=enc.get_feature_names_out(), dtype="int")
# print(encoded_df.head())
housing_df = pd.concat([housing_df, encoded_df], axis=1).drop(["ocean_proximity"], axis=1)
# print(housing_df.head())

X = housing_df[housing_df.columns.difference(["median_house_value"])]
Y = housing_df[["median_house_value"]]
print(X.info())
print(Y.head())

# plt.scatter(X["housing_median_age"], Y)
plt.plot(X["housing_median_age"], Y, )
plt.show()