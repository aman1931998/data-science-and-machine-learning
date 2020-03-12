import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import statsmodels.formula.api as sm


dataset = pd.read_csv('kc_house_data.csv')


#nan check
missing = dataset.isna().sum() #No missing values

#Price for dependent variable Y
price = dataset['price'].values.reshape(dataset.shape[0], 1)
#lb_price = StandardScaler()
#price = lb_price.fit_transform(price)
dataset.pop('price')

#bedrooms and #bathrooms and Feature Enginnering -> Bedrooms with attached bathrooms
bedrooms = dataset['bedrooms'].values.reshape(dataset.shape[0], 1)
bathrooms = dataset['bathrooms'].values.reshape(dataset.shape[0], 1)

attached_bedrooms = np.subtract(bedrooms, bathrooms)
for i in range(dataset.shape[0]):
    if attached_bedrooms[i][0]<=0: attached_bedrooms[i][0]=0
attached_bedrooms = StandardScaler().fit_transform(attached_bedrooms)

bedrooms = StandardScaler().fit_transform(bedrooms)
bathrooms = StandardScaler().fit_transform(bathrooms)

dataset.pop('bedrooms')
dataset.pop('bathrooms')
del i

#sqft_living
sqft_living = dataset['sqft_living'].values.reshape(dataset.shape[0], 1)
sqft_living = StandardScaler().fit_transform(sqft_living)
dataset.pop('sqft_living')

#sqft_lot
sqft_lot = dataset['sqft_lot'].values.reshape(dataset.shape[0], 1)
sqft_lot = StandardScaler().fit_transform(sqft_lot)
dataset.pop('sqft_lot')

#floors
floors = dataset['floors'].values.reshape(dataset.shape[0], 1)
floors = StandardScaler().fit_transform(floors)
dataset.pop('floors')

#waterfront
waterfront = dataset['waterfront'].values.reshape(dataset.shape[0], 1)
waterfront = StandardScaler().fit_transform(waterfront)
dataset.pop('waterfront')

#view
view = dataset['view'].values.reshape(dataset.shape[0], 1)
view = StandardScaler().fit_transform(view)
dataset.pop('view')

#condition
condition = dataset['condition'].values.reshape(dataset.shape[0], 1)
condition = StandardScaler().fit_transform(condition)
dataset.pop('condition')

#grade
grade = dataset['grade'].values.reshape(dataset.shape[0], 1)
grade = StandardScaler().fit_transform(grade)
dataset.pop('grade')

#sqft_above
sqft_above = dataset['sqft_above'].values.reshape(dataset.shape[0], 1)
sqft_above = StandardScaler().fit_transform(sqft_above)
dataset.pop('sqft_above')

#sqft_basement
sqft_basement = dataset['sqft_basement'].values.reshape(dataset.shape[0], 1)
sqft_basement = StandardScaler().fit_transform(sqft_basement)
dataset.pop('sqft_basement')

#sqft_living15
sqft_living15 = dataset['sqft_living15'].values.reshape(dataset.shape[0], 1)
sqft_living15 = StandardScaler().fit_transform(sqft_living15)
dataset.pop('sqft_living15')

#sqft_lot15
sqft_lot15 = dataset['sqft_lot15'].values.reshape(dataset.shape[0], 1)
sqft_lot15 = StandardScaler().fit_transform(sqft_lot15)
dataset.pop('sqft_lot15')


#yr_built and #yr_renovated
yr_built = dataset['yr_built']
yr_renovated = dataset['yr_renovated']
#checking counts
yr_built.value_counts()
yr_renovated.value_counts()
yr_built = yr_built.values.reshape(dataset.shape[0], 1)
yr_renovated = yr_renovated.values.reshape(dataset.shape[0], 1)

#Method 1? Directly use both values?
yr_built = LabelEncoder().fit_transform(yr_built).reshape(dataset.shape[0], 1)
yr_renovated = LabelEncoder().fit_transform(yr_renovated).reshape(dataset.shape[0], 1)
#Method 2? Replace yr_built values with yr_renovated values if !=0
for i in range(dataset.shape[0]):
    if yr_renovated[i][0]!=0: yr_built[i][0] = yr_renovated[i][0]
yr_built = LabelEncoder().fit_transform(yr_built).reshape(dataset.shape[0], 1)
del yr_renovated

dataset.pop('yr_built')
dataset.pop('yr_renovated')
del i

#zipcode
zipcode = dataset['zipcode']
#checking counts
zipcode.value_counts()
zipcode = zipcode.values.reshape(dataset.shape[0], 1)
zipcode = OneHotEncoder(categorical_features = [0]).fit_transform(zipcode).toarray()[:, 1:]
dataset.pop('zipcode')


#lat
lat = dataset['lat']
#checking counts
lat.value_counts()
lat = lat.values.reshape(dataset.shape[0], 1)
lat = np.array(list(map(lambda x:int(float("%.1f"%x)*10), lat))).reshape(dataset.shape[0], 1)
lat = OneHotEncoder(categorical_features = [0]).fit_transform(lat).toarray()[:, 1:]
dataset.pop('lat')


#long
long = dataset['long']
#checking counts
long.value_counts()
long = long.values.reshape(dataset.shape[0], 1)
long = np.array(list(map(lambda x:-int(float("%.1f"%x)*10), long))).reshape(dataset.shape[0], 1)
long = OneHotEncoder(categorical_features = [0]).fit_transform(long).toarray()[:, 1:]
dataset.pop('long')


#date
date = dataset['date']
base_date = datetime.datetime(2014, 1, 1)
l = []
for i in date: l.append(datetime.datetime.strptime(i, '%Y%m%dT000000'))
date = []
for i in l: date.append((i-base_date).days)
date = np.array(date).reshape(dataset.shape[0], 1)
date = StandardScaler().fit_transform(date)
dataset.pop('date')

del base_date, l, i

#Creating X and Y
X = np.concatenate((zipcode, yr_built, waterfront, view, attached_bedrooms
                    sqft_lot15, sqft_lot, sqft_living, sqft_living15, 
                    sqft_basement, sqft_above, long, lat, grade, 
                    floors, date, condition, bedrooms, bathrooms), axis = 1)

del zipcode, yr_built, waterfront, view, sqft_lot15, sqft_lot, sqft_living, sqft_living15, sqft_basement, sqft_above, long, lat, grade, floors, date, condition, bedrooms, bathrooms
del missing

Y = np.array(price)
del price

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

del X, Y

np.save('npy_files/X.npy', X)
np.save('npy_files/Y.npy', Y)
np.save('npy_files/X_test.npy', X_test)
np.save('npy_files/X_train.npy', X_train)
np.save('npy_files/Y_test.npy', Y_test)
np.save('npy_files/Y_train.npy', Y_train)

X = np.save('npy_files/X.npy')
Y = np.save('npy_files/Y.npy')
X_test = np.load('npy_files/X_test.npy')
X_train = np.load('npy_files/X_train.npy')
Y_test = np.load('npy_files/Y_test.npy')
Y_train = np.load('npy_files/Y_train.npy')

#Model

xgb = XGBRegressor(max_depth = 6, n_estimators = 2200, learning_rate = 0.025, n_jobs = 7)
xgb.fit(X_train, Y_train)

Y_pred_train = xgb.predict(X_train).reshape((Y_train.shape[0], 1))
Y_pred_test = xgb.predict(X_test).reshape((Y_test.shape[0], 1))

loss_train, loss_test = 0, 0

sctrain = StandardScaler()
Y_train = sctrain.fit_transform(Y_train)
Y_pred_train = sctrain.transform(Y_pred_train)

sctest = StandardScaler()
Y_test = sctest.fit_transform(Y_test)
Y_pred_test = sctest.transform(Y_pred_test)


for i in range(len(Y_pred_train)):
    loss_train+=abs(Y_train[i][0] - Y_pred_train[i])

for i in range(len(Y_pred_test)):
    loss_test+=abs(Y_test[i][0] - Y_pred_test[i])

loss_train, loss_test = loss_train/len(Y_train), loss_test/len(Y_test)

















################################ ADDITIONAL TESTING  ############################

#FILTERING USING Statsmodels.formula.api

#X_train_sm = np.append(arr = np.ones((17290, 1)), values = X_train, axis = 1)
#X_test_sm = np.append(arr = np.ones((4323, 1)), values = X_test, axis = 1)

X_sm = np.append(arr = np.ones((21613, 1)), values = X, axis = 1)

regressor_ols = sm.OLS(endog = Y, exog = X_sm).fit()

regressor_ols.summary()

summary = regressor_ols.summary()

summary = summary.as_csv()

file = open('summary.csv', 'w')

file.write(summary)

file.close()


X_fixed = list(range(104))
for i in [12, 18, 19, 20, 26, 27, 40, 68, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97]:
    X_fixed.remove(i)

X_fixed = X_sm[:, X_fixed]



#Model
X_train, X_test, Y_train, Y_test = train_test_split(X_fixed, Y, test_size = 0.2)

xgb = XGBRegressor(max_depth = 7, n_estimators = 2500, learning_rate = 0.025, n_jobs = 6)
xgb.fit(X_train, Y_train)

Y_pred_train = xgb.predict(X_train)
Y_pred_test = xgb.predict(X_test)

loss_train, loss_test = 0, 0

for i in range(len(Y_pred_train)):
    loss_train+=abs(Y_train[i][0] - Y_pred_train[i])

for i in range(len(Y_pred_test)):
    loss_test+=abs(Y_test[i][0] - Y_pred_test[i])

loss_train, loss_test = loss_train/len(Y_train), loss_test/len(Y_test)

print("Train Error: ", 1-loss_train[0])
print("Test Error: ", 1-loss_test[0])
