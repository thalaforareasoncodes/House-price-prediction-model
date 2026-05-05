# Step 1: Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(df.head())

mi=int(input('Enter median income-:         '))
hma=int(input('Enter housing_median_age-:   '))
tr=int(input('Enter total_rooms-:           '))
tb=int(input('Enter total_bedrooms-:        '))
p=int(input('Enter population-:             '))
h=int(input('Enter households-:             '))
mhv=int(input('Enter median_house_value-:   '))


df = df[[
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_house_value"
]]
df = df.dropna()
X = df[[
    "median_income",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households"
]]
y = df["median_house_value"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
print("Train MAE:", mean_absolute_error(y_train, train_pred))
print("Test MAE:", mean_absolute_error(y_test, test_pred))

print("R2 Score:", r2_score(y_test, test_pred))
print("\nFeature Importance:")
for name, score in zip(X.columns, model.feature_importances_):
    print(name, ":", score)
new_house = pd.DataFrame({
    "median_income": [mi],
    "housing_median_age": [hma],
    "total_rooms": [tr],
    "total_bedrooms": [tb],
    "population": [p],
    "households": [h]
})
prediction = model.predict(new_house)

print("\nPredicted Price:", prediction[0])