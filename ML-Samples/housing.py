import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# let me load the csv as a dataframe
housing_df = pd.read_csv("data\\housing.csv")
print(housing_df.head())
print(housing_df.tail())
print(housing_df.info())

# oh! pandas would have loaded all columns as float... convert value of columns that are supposed to be int
housing_df[["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]] = \
                    housing_df[["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]].fillna(0.0).astype(int)
# oho! convert default 'object' type to string
housing_df["ocean_proximity"] = housing_df["ocean_proximity"].astype("string")

print(housing_df.head())
print(housing_df.tail())
print(housing_df.info())

# hmm... convert ocean_proximity from string to categorical... let me try OneHotEncoder
# ocean_proximity_catmap = [["<1H OCEAN", 1],["INLAND", 2],["ISLAND", 3],["NEAR BAY", 4], ["NEAR OCEAN", 5]]

y = pd.get_dummies(housing_df.ocean_proximity, prefix="ocean_prox")
print(y.info())

enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(housing_df["ocean_proximity"].unique().reshape(-1,1))

# column_encoded = pd.DataFrame(enc.transform(housing_df["ocean_proximity"].to_numpy().reshape(-1, 1)))
# print(column_encoded)