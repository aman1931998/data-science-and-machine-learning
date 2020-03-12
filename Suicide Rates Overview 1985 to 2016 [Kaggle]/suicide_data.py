import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

dataset = pd.read_csv("master.csv")

###############     country
country = dataset['country']

#cheking for values/items
country_counts = country.value_counts()

country = country.values

lb_country = LabelEncoder()
country = lb_country.fit_transform(country).reshape((dataset.shape[0], 1))
ohe_country = OneHotEncoder(categorical_features = [0])
country = ohe_country.fit_transform(country).toarray()[:, :-1]

#np.save(r'npy_files\country.npy', country)

dataset.pop('country')

del country_counts

###############     sex
sex = dataset['sex'].values

lb_sex = LabelEncoder()
sex = lb_sex.fit_transform(sex).reshape((dataset.shape[0], 1))

#np.save(r'npy_files\sex.npy', sex)

dataset.pop('sex')

###############     suicides_no
suicides_no = dataset['suicides_no'].values.reshape((dataset.shape[0], 1))

#np.save(r'npy_files\suicides_no.npy', suicides_no)

dataset.pop('suicides_no')

###############     population
population = dataset['population'].values.reshape((dataset.shape[0], 1))

sc_population = StandardScaler()
population = sc_population.fit_transform(population).reshape((dataset.shape[0], 1))
#np.save(r'npy_files\population.npy', population)

dataset.pop('population')

###############     age
age = dataset['age'].values

#Approach 1
l = ['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years', ]
d = dict(zip(l, list(range(6))))
l = []
for i in age:
    l.append(d[i])
age = np.array(l).reshape((dataset.shape[0], 1))
del i, l, d

#Approach 2
#Use OneHotEncoder (along-with LabelEncoder) ?????
#lb_age = LabelEncoder()
#age = lb_age.fit_transform(age).reshape((dataset.shape[0], 1))
#ohe_age = OneHotEncoder(categorical_features = [0])
#age = ohe_age.fit_transform(age).toarray()[:, :-1]

#np.save(r'npy_files\age.npy', age)

dataset.pop('age')

###############     suicides/100k
s100k = dataset['suicides/100k pop'].values.reshape((dataset.shape[0], 1))
sc_s100k = StandardScaler()
s100k = sc_s100k.fit_transform(s100k).reshape((dataset.shape[0], 1))
#np.save(r'npy_files\s100k.npy', s100k)

dataset.pop('suicides/100k pop')

###############     year
##Approach 1
#year = dataset['year'].values.reshape((dataset.shape[0], 1))
#ohe_year = OneHotEncoder(categorical_features = [0])
#year = ohe_year.fit_transform(year).toarray()[:, :-1]

#Approach 2
year = dataset['year'].values
lb_year = LabelEncoder()
year = lb_year.fit_transform(year).reshape((dataset.shape[0], 1))
sc_year = StandardScaler()
year = sc_year.fit_transform(year).reshape((dataset.shape[0], 1))
#np.save(r'npy_files\year.npy', year)

dataset.pop('year')

###############     gdp_per_capita
gdp_per_capita = dataset['gdp_per_capita ($)'].values.reshape((dataset.shape[0], 1))
sc_gdp_per_capita = StandardScaler()
gdp_per_capita = sc_gdp_per_capita.fit_transform(gdp_per_capita).reshape((dataset.shape[0], 1))
#np.save(r'npy_files\gdp_per_capita.npy', gdp_per_capita)

dataset.pop('gdp_per_capita ($)')

###############     gdp_for_year
gdp_for_year = dataset[' gdp_for_year ($) '].values.reshape((dataset.shape[0], 1))
l = []
for i in gdp_for_year:
    x_ = ""
    for i in i[0]:
        if i!= ',': x_ += i
    l.append(int(x_))
gdp_for_year = np.array(l).reshape((dataset.shape[0], 1))
sc_gdp_for_year = StandardScaler()
gdp_for_year = sc_gdp_for_year.fit_transform(gdp_for_year).reshape((dataset.shape[0], 1))
#np.save(r'npy_files\gdp_for_year.npy', gdp_for_year)

del i, l, x_
dataset.pop(' gdp_for_year ($) ')

###############     generation
generation = dataset['generation'].values.reshape((dataset.shape[0], 1))
lb_generation = LabelEncoder()
generation = lb_generation.fit_transform(generation).reshape((dataset.shape[0], 1))
ohe_generation = OneHotEncoder(categorical_features = [0])
generation = ohe_generation.fit_transform(generation).toarray()[:, :-1]

#np.save(r'npy_files\generation.npy', generation)

dataset.pop('generation')

###############     HDI for year
dataset.pop('HDI for year')

###############     country-year
dataset.pop('country-year')

###############     Combining all files
X = np.concatenate((age, country, gdp_for_year, 
                          gdp_per_capita, generation, 
                          population, s100k, sex, 
                          year), axis = 1)

Y = suicides_no

np.save(r'npy_files\X.npy', X)
np.save(r'npy_files\Y.npy', Y)

del generation
del gdp_for_year
del s100k
del age
del population
del suicides_no
del sex
del country
del year
del gdp_per_capita


###############     train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15, random_state = 44)

np.save(r'npy_files\X_train.npy', X_train)
np.save(r'npy_files\Y_train.npy', Y_train)
np.save(r'npy_files\X_test.npy', X_test)
np.save(r'npy_files\Y_test.npy', Y_test)

del X, Y


###############     XGB_regressor

results = []

from xgboost import XGBRegressor

i, j = 6, 700
model = XGBRegressor(max_depth = i, 
                     learning_rate = 0.1, 
                     n_estimators = j, 
                     silent = False, 
                     n_jobs = 6)
model.fit(X_train, Y_train)
Y_pred_train = model.predict(X_train).reshape(Y_train.shape[0], 1)
Y_pred_test = model.predict(X_test).reshape(Y_test.shape[0], 1)

#Observation

cm_train= np.subtract(Y_train, Y_pred_train)
cm_train= np.abs(cm_train)
cm_train= np.sum(cm_train)
cm_train/=Y_train.shape[0]

cm_test = np.subtract(Y_test, Y_pred_test)
cm_test = np.abs(cm_test)
cm_test = np.sum(cm_test)
cm_test/=Y_test.shape[0]


print("\n\n\n")
print("max_depth, n_estimators = %d, %d" %(i, j))
print("Train: %.8f" %(cm_train))
print("Test: %.8f" %(cm_test))

