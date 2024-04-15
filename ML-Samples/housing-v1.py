from pathlib import Path
import pandas as pd

housing = pd.read_csv("data\\housing.csv")

print(housing.head())
print(housing.describe())
print(housing["ocean_proximity"].value_counts()) # find how many categories are there and how many samples in each category

import matplotlib.pylab as plt
housing.hist(bins=50, figsize=(12, 8))
plt.show()

# randomly one can get test data, but will not be unique each time...
# another option can be to use hashcode of each row's unique identifier if available (like longitude and latitude)...
# better option is just use from sklearn.model_selection, train_test_split
import numpy as np

def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    print(shuffled_indices)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    print(test_indices)
    train_indices = shuffled_indices[test_set_size:]
    print(train_indices)

shuffle_and_split_data(housing, 0.2)

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# in oder to get test data that properly represents different income categories, instead of random split to get some random test data, let us do the following
# based on histogram of median_income, we can see median_income can be split into some ranges and name each one as one category (like 1, 2, 3...)
# we can infact make it as a temporary column and use it select test data from each of the income category, so we have proportional test data from each income category
housing["income_category"] = pd.cut(housing["median_income"],
                                    bins=[0., 1.5, 3.0, 4.5, 6, np.inf],
                                    labels=[1, 2, 3, 4, 5])
housing["income_category"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Number of districts")
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit

# option 1 - kind of manually, split test based on strata (income category)
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing, housing["income_category"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]
# or

# option 2
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_category"], random_state=42)

# let us see how well stratification worked
print(strat_test_set["income_category"].value_counts() / len(strat_test_set))

# now we dont need income_category anymore, let us drop it
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_category", axis=1, inplace=True)

housing = strat_train_set.copy() # now housing has only the train data (without test data)

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))
plt.show()

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# median_income seems to have interesting correlation with median_house_value
housing.plot(kind="scatter", x="median_income", y = "median_house_value", alpha=0.2, grid=True)
plt.show()

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# let us get ready for machine learning
housing = strat_test_set.drop("median_house_value", axis=1) # X
housing_labels = strat_test_set["median_house_value"].copy() # y

# fix some of the total_bedrooms having null values
from sklearn.impute import SimpleImputer # other imputers include KNNImputer, IterativeImputer
imputer = SimpleImputer(strategy="median") # other strategies are ("mean"), ("most_frequent"), ("constant", fill_value=...)

housing_num = housing.select_dtypes(include=[np.number]) # select only numeric columns
imputer.fit(housing_num)
print(imputer.statistics_) # only to understand, statistics_ has median for every column
X = imputer.transform(housing_num)
print(X)

# let us learn about handling categorical columns
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

# one way to handle categorical column is OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(ordinal_encoder.categories_)

# ordinal encoding might mislead ML algorithm by paying more attention than needed to the ordinal values
# better option would be OneHotEncoder (definitely preferred than pandas' get_dummies())
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Feature Scaling (mix-max scaling (normalization) and standardization)
# min-max scaler => scaled_value = (value - min) / (max - min)
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# feature scaling has more to learn like handling heavy tail then replacing them with its logarithm
# let us do that later...

# sometimes even the target values need to be transformed (scaled) like using one's logarithm values
# then predicted values too would be logarithm values... inverse_tranform() would be useful to determine the actual/intended predicted value
# option 1
from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)

some_new_data = housing[["median_income"]].iloc[:5]
scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
print("Explicit Inverse Transform", predictions)

# option 2 (precise)
from sklearn.compose import TransformedTargetRegressor
model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)
print("TransformedTargetRegressor", predictions)

# pipeline
# sample
# from sklearn.pipeline import Pipeline
# num_pipeline = Pipeline([
#     ("impute", SimpleImputer(strategy="median")),
#     ("standardize", StandardScaler)])

# Transformation Pipelines

# from sklearn.pipeline import Pipeline

# num_pipeline = Pipeline([
#     ("impute", SimpleImputer(strategy="median")),
#     ("standardize", StandardScaler())
# ])

from sklearn.pipeline import make_pipeline
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

housing_num_prepared = num_pipeline.fit_transform(housing_num)

# get a dataframe from the pipeline
df_housing_num_prepared = pd.DataFrame(
                                    housing_num_prepared,
                                    columns=num_pipeline.get_feature_names_out(),
                                    index=housing_num.index)
print(df_housing_num_prepared.head(2))

# column level transformer to apply for both numeric and category columns
from sklearn.compose import ColumnTransformer
num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

# it is not convenient to mention all the column titles like that... make_column_transformer can automatically pick the columns
from sklearn.compose import make_column_selector, make_column_transformer
preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline, make_column_selector(dtype_include=object)),
)

housing_prepared = preprocessing.fit_transform(housing)

# need to see if following is needed or not
# start - pipeline
from sklearn.preprocessing import FunctionTransformer

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler())

from sklearn.cluster import KMeans

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

cluster_simil = ClusterSimilarity()