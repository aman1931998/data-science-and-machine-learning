import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
import datetime
from xgboost import XGBRegressor

dataset = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')

#missing values check
missing = dataset.isna().sum()

#name -> Later
name = dataset.name.apply(lambda x: x.split('_')[0].lower())

name_values = name.value_counts()

dataset.pop('name')

#seller
seller = dataset.seller
seller.value_counts()
seller = seller.values
seller = OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(seller).reshape((dataset.shape[0], 1))).toarray()[:, 1:]

#np.save('npy_files/seller.npy', seller)
#seller = np.load('npy_files/seller.npy')

dataset.pop('seller')

#offerType
offerType = dataset.offerType
offerType.value_counts()
offerType = offerType.values
offerType = OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(offerType).reshape((dataset.shape[0], 1))).toarray()[:, :-1]

#np.save('npy_files/offerType.npy', offerType)
#offerType = np.load('npy_files/offerType.npy')

dataset.pop('offerType')

#price
price = dataset.price.values

#np.save('npy_files/price.npy', price)
#price = np.load('npy_files/price.npy')

dataset.pop('price')

#abtest
abtest = dataset.abtest
abtest.value_counts()
abtest = abtest.values
abtest = OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(abtest).reshape((dataset.shape[0], 1))).toarray()[:, :-1]

#np.save('npy_files/abtest.npy', abtest)
#abtest = np.load('npy_files/abtest.npy')

dataset.pop('abtest')

#vehicleType -> too much missing values -> ignore
dataset.pop('vehicleType')

#yearOfRegistration              #what if we convert to year-sets of 5? -> reduces dummy columns by 5
yearOfRegistration = dataset.yearOfRegistration
yearOfRegistration.value_counts()
yearOfRegistration = yearOfRegistration.values
yearOfRegistration = LabelEncoder().fit_transform(yearOfRegistration).reshape((dataset.shape[0], 1))
#yearOfRegistration = OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(yearOfRegistration).reshape((dataset.shape[0], 1))).toarray()[:, :-1]

#np.save('npy_files/yearOfRegistration.npy', yearOfRegistration)
#yearOfRegistration = np.load('npy_files/yearOfRegistration.npy')

dataset.pop('yearOfRegistration')

#gearbox -> too much missing values -> ignore
dataset.pop('gearbox')

#powerPS
powerPS = dataset.powerPS.values.reshape((dataset.shape[0], 1))
powerPS = StandardScaler().fit_transform(powerPS)

#np.save('npy_files/powerPS.npy', powerPS)
#powerPS = np.load('npy_files/powerPS.npy')

dataset.pop('powerPS')

#model -> too much missing values -> ignore
dataset.pop('model')

#kilometer
kilometer = dataset.kilometer.values.reshape((dataset.shape[0], 1))
kilometer = StandardScaler().fit_transform(kilometer)

#np.save('npy_files/kilometer.npy', kilometer)
#kilometer = np.load('npy_files/kilometer.npy')

dataset.pop('kilometer')

#monthOfRegistration
monthOfRegistration = dataset.monthOfRegistration.values.reshape((dataset.shape[0], 1))

#np.save('npy_files/monthOfRegistration.npy', monthOfRegistration)
#monthOfRegistration = np.load('npy_files/monthOfRegistration.npy')

dataset.pop('monthOfRegistration')

#fuelType -> too much missing values -> ignore
dataset.pop('fuelType')

#brand
brand = dataset.brand
brand.value_counts()
brand = brand.values
brand = OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(brand).reshape((dataset.shape[0], 1))).toarray()[:, :-1]

#np.save('npy_files/brand.npy', brand)
#brand = np.load('npy_files/brand.npy')

dataset.pop('brand')

#notRepairedDamage -> too too too much missing values -> definately a no no!
dataset.pop('notRepairedDamage')

#dateCreated and dateCrawled and lastSeen-> Later!!!
dateCreated = dataset.dateCreated
dateCrawled = dataset.dateCrawled.values
lastSeen = dataset.lastSeen

base_date = ''
l = []
for i in dateCrawled:
    l.append(datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
dateCrawled = l

del l, i
dataset.pop('dateCreated')
dataset.pop('dateCrawled')
dataset.pop('lastSeen')

#nrOfPictures -> Useless data
dataset.pop('nrOfPictures')

#postalCode -> Later!!! try converting postal codes to states
postalCode = dataset.postalCode
postalCode.value_counts()

dataset.pop('postalCode')

###############################################################################

#Creating dataset and X, Y variables
X = np.concatenate((abtest, brand, kilometer, monthOfRegistration, 
                    offerType, powerPS, seller, yearOfRegistration), axis = 1)
Y = price

del abtest, brand, kilometer, monthOfRegistration, offerType, powerPS, seller, yearOfRegistration
del price

np.save('npy_files/X.npy', X)
np.save('npy_files/Y.npy', Y)

X = np.load('npy_files/X.npy')
Y = np.load('npy_files/Y.npy')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

del X, Y, missing

#sc_y = StandardScaler()
#Y_train = sc_y.fit_transform(Y_train.reshape((Y_train.shape[0], 1))).ravel()
#Y_test = sc_y.transform(Y_test.reshape((Y_test.shape[0], 1))).ravel()

###############################################################################
model = XGBRegressor(n_estimators = 2000, 
                     max_depth = 5, 
                     n_jobs = 4, 
                     learning_rate = 0.00005)

model.fit(X_train, Y_train)

Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

train_error = np.sum(np.abs(np.subtract(Y_train, Y_train_pred)))/297222
test_error = np.sum(np.abs(np.subtract(Y_test, Y_test_pred)))/74306

print("Train Error:", train_error)
print("Test Error:", test_error)


