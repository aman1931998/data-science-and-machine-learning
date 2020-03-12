from xgboost import XGBRegressor
import statsmodels.formula.api as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


train = pd.read_excel('Data_Train.xlsx')
test = pd.read_excel('Data_Test.xlsx')

#Y_train
Y_train = train.pop('Price').values.reshape((train.shape[0], 1))
#dataset
dataset = pd.concat((train, test), axis = 0)
del train, test

#missing
missing = dataset.isna().sum()

########################################Mileage 

#1
#nan_mileage = list(dataset['Mileage'].isna())
#name_mileage = list(dataset['Name'])
#nan_mileage_ = []
#for i in range(dataset.shape[0]):
#    if nan_mileage[i] == True: nan_mileage_.append(name_mileage[i])
#
#mileage = dataset['Mileage']
#mileage = mileage.replace(to_replace = 'Mahindra E Verito D4', value = '110.0 km/full charge')
#mileage = mileage.replace(to_replace = 'Toyota Prius 2009-2016 Z4', value = '100.0 km/full charge')

#2 Removing 2 missing mileage values (Electric cars -> Useless)
dataset = dataset[pd.notnull(dataset['Mileage'])] #removed 2 nan (Electric cars)
mileage = np.array(list(map(lambda x:float(x.split()[0]), dataset['Mileage'].values))).reshape(dataset.shape[0], 1)
dataset.pop('Mileage')
mileage = StandardScaler().fit_transform(mileage).reshape((mileage.shape))

#Same for Y_train
Y_train = np.concatenate((Y_train[:4446], Y_train[4447:4904], Y_train[4905:]), axis = 0)

np.save('npy/Y_train.npy', Y_train)
np.save('npy/mileage.npy', mileage)

########################################Location

location = dataset['Location']
location.value_counts()
location = location.values.reshape((dataset.shape[0], 1))
location = OneHotEncoder(categories = 'auto').fit_transform(location).toarray()[:, 1:]
dataset.pop('Location')

np.save('npy/location.npy', location)

########################################Year

year = dataset['Year'].values
year = LabelEncoder().fit_transform(year).reshape((dataset.shape[0], 1))
year = StandardScaler().fit_transform(year)
dataset.pop('Year')

np.save('npy/year.npy', year)

########################################Kilometers_Driven

kilometers_driven = dataset['Kilometers_Driven'].values.reshape((dataset.shape[0], 1))
kilometers_driven = StandardScaler().fit_transform(kilometers_driven)
dataset.pop('Kilometers_Driven')

np.save('npy/kilometers_driven.npy', kilometers_driven)

########################################Fuel_Type

fuel_type = dataset['Fuel_Type']
fuel_type.value_counts()
fuel_type = fuel_type.values.reshape((dataset.shape[0], 1))
fuel_type = OneHotEncoder(categories = 'auto').fit_transform(fuel_type).toarray()[:, :-1]
#fuel_type = OneHotEncoder(categories = 'auto').fit_transform(fuel_type).toarray()[:, 1:]
dataset.pop('Fuel_Type')

np.save('npy/fuel_type.npy', fuel_type)

########################################Transmission

transmission = dataset['Transmission']
transmission.value_counts()
transmission = transmission.values
transmission = LabelEncoder().fit_transform(transmission).reshape((dataset.shape[0], 1))
#fuel_type = OneHotEncoder(categories = 'auto').fit_transform(fuel_type).toarray()[:, 1:]
dataset.pop('Transmission')

np.save('npy/transmission.npy', transmission)

########################################Owner_Type

owner_type = dataset['Owner_Type']
owner_type.value_counts()
owner_type = owner_type.replace(to_replace = 'First', value = 1)
owner_type = owner_type.replace(to_replace = 'Second', value = 2)
owner_type = owner_type.replace(to_replace = 'Third', value = 3)
owner_type = owner_type.replace(to_replace = 'Fourth & Above', value = 4)
owner_type = owner_type.values.reshape((dataset.shape[0], 1))
dataset.pop('Owner_Type')

np.save('npy/owner_type.npy', owner_type)

########################################New Price

dataset.pop('New_Price')

del missing

########################################Engine Power & Seats

#engine
engine = list(dataset['Engine'].isna())
name_engine = list(dataset['Name'])
nan_engine = []
for i in range(dataset.shape[0]):
    if engine[i] == True: nan_engine.append(name_engine[i])
nan_engine = list(set(nan_engine))
del engine, i, name_engine

#power
power = dataset['Power']
power = power.replace(to_replace = 'null bhp', value = np.nan)
dataset['Power'] = power
power = list(dataset['Power'].isna())
name_power = list(dataset['Name'])
nan_power = []
for i in range(dataset.shape[0]):
    if power[i] == True: nan_power.append(name_power[i])
nan_power = list(set(nan_power))
del i, name_power, power

#seats
seats = list(dataset['Seats'].isna())
name_seats = list(dataset['Name'])
nan_seats = []
for i in range(dataset.shape[0]):
    if seats[i] == True: nan_seats.append(name_seats[i])
nan_seats = list(set(nan_seats))
del i, name_seats, seats

#Sorting
nan_engine = sorted(nan_engine)
nan_power = sorted(nan_power)
nan_seats = sorted(nan_seats)

#manual filling
values_all = []
values_power_extra = []
values_seats_extra = []
values_seats_power_extra = []

############### template
#values_power_extra.append([nan_power[], ''])
#values.append([nan_engine[], ])
#values_seats_extra.append([nan_seats[], ])
#del nan_engine[0]
#del nan_power[0]
#del nan_seats[0]
###################


values_power_extra.append([nan_power[0], '265 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1995 CC', '188 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '104 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '100 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1172 CC', '67 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1248 CC', '75 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1368 CC', '90 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '70 bhp'])
del nan_power[0]

values_seats_power_extra.append([nan_seats[0], '156 bhp', 7])
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '143 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '67 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '68 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '101 bhp'])
del nan_power[0]

values_seats_extra.append([nan_seats[0], 5])
del nan_seats[0]

values_power_extra.append([nan_power[0], '74.87 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1997 CC', '141.1 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '185 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1497 CC', '116.386 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1493 CC', '100 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_seats_extra.append([nan_seats[0], 5])
del nan_seats[0]

values_seats_extra.append([nan_seats[0], 5])
del nan_seats[0]

values_all.append([nan_engine[0], '1198 CC', '88.8 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '68 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '999 CC', '63 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '999 CC', '62 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1086 CC', '47 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1396 CC', '99 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '2993 CC', '255 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '2993 CC', '241.4 bhp', 6])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '2197 CC', '118 bhp', 9])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '60 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '94 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '83.1 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '85 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '85 bhp'])
del nan_power[0]

values_seats_power_extra.append([nan_power[0], '67.1 bhp', 5])
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1197 CC', '84 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1197 CC', '84 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1197 CC', '84 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_all.append([nan_engine[0], '1197 CC', '84 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '74 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '74 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1061 CC', '67 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '141 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '63 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '170 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '265 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '265 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1798 CC', '158 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '160 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '60 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '62 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '67 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1364 CC', '68 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '80 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '67 bhp'])
del nan_power[0]

values_all.append([nan_engine[0], '1197 CC', '80 bhp', 5])
del nan_engine[0]
del nan_power[0]
del nan_seats[0]

values_power_extra.append([nan_power[0], '73.974 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '73.974 bhp'])
del nan_power[0]

values_power_extra.append([nan_power[0], '105 bhp'])
del nan_power[0]

del nan_engine, nan_seats, nan_power

##################################BACKUP
np.save('npy/values_all.npy', np.array(values_all))
np.save('npy/values_power_extra.npy', np.array(values_power_extra))
np.save('npy/values_seats_extra.npy', np.array(values_seats_extra))
np.save('npy/values_seats_power_extra.npy', np.array(values_seats_power_extra))
##################################

##### Creating Engine, Seats, Power dataset
name = [i for i in dataset['Name'].values]
engine = [i for i in dataset['Engine'].values]
power = [i for i in dataset['Power'].values]
seats = [i for i in dataset['Seats'].values]

#Taking values from values_all
for i in values_all:
    for j in range(len(engine)):
        if i[0] == name[j]:
            engine[j] = i[1]
            power[j] = i[2]
            seats[j] = i[3]
del values_all

#Taking values from values_power_extra
for i in values_power_extra:
    for j in range(len(power)):
        if i[0] == name[j]:
            power[j] = i[1]
del values_power_extra

#Taking values from values_seats_extra
for i in values_seats_extra:
    for j in range(len(seats)):
        if i[0] == name[j]:
            seats[j] = i[1]
del values_seats_extra

#Taking values from values_seats_power_extra
for i in values_seats_power_extra:
    for j in range(len(power)):
        if i[0] == name[j]:
            power[j] = i[1]
            seats[j] = i[2]
del values_seats_power_extra
del i, j

#Checking for all data, if it contains missing values or not
missing_seats = []
temp = np.isnan(np.array(seats))
for i in range(len(temp)):
    if temp[i]:
        missing_seats.append(i)
del i, missing_seats, temp
#Found 1
seats[1917] = 5

#Engine Ready
engine = [float(i.split()[0]) for i in engine]
engine = np.array(engine).reshape((dataset.shape[0], 1))
engine = StandardScaler().fit_transform(engine).reshape((engine.shape))
dataset.pop('Engine')

np.save('npy/engine.npy', engine)

#Seats Ready
seats = np.array(seats).reshape((dataset.shape[0], 1))
seats = StandardScaler().fit_transform(seats).reshape((seats.shape))
dataset.pop('Seats')

np.save('npy/seats.npy', seats)

#Power Ready

power = [float(i.split()[0]) for i in power]
power = np.array(power).reshape((dataset.shape[0], 1))
power = StandardScaler().fit_transform(power).reshape((power.shape))
dataset.pop('Power')

np.save('npy/power.npy', power)

del name

########################################Name

name = [i.split()[0] for i in dataset['Name'].values]
name = np.array(name).reshape((dataset.shape[0], 1))
name = OneHotEncoder(categories = 'auto').fit_transform(name).toarray()[:, 1:]
dataset.pop('Name')

np.save('npy/name.npy', name)

###############################################################################

#Loading dataset
name = np.load('npy/name.npy')
location = np.load('npy/location.npy')
year = np.load('npy/year.npy')
kilometers_driven = np.load('npy/kilometers_driven.npy')
fuel_type = np.load('npy/fuel_type.npy')
transmission = np.load('npy/transmission.npy')
owner_type = np.load('npy/owner_type.npy')
mileage = np.load('npy/mileage.npy')
seats = np.load('npy/seats.npy')
engine = np.load('npy/engine.npy')
power = np.load('npy/power.npy')


#DATASET READY
dataset = np.concatenate((name,                         #32
                          location,                     #10
                          year,                         #1
                          kilometers_driven,            #1
                          fuel_type,                    #3
                          transmission,                 #1
                          owner_type,                   #1
                          mileage,                      #1
                          seats,                        #1
                          engine,                       #1
                          power                         #1
                          ), axis = -1)

del  name, location, year, kilometers_driven, fuel_type
del transmission, owner_type, mileage, seats, engine, power

X_train, X_test = dataset[:6017], dataset[6017:]

np.save('npy/dataset.npy', dataset)
np.save('npy/X_train.npy', X_train)
np.save('npy/X_test.npy', X_test)

#Creating CV set for error checking

#from sklearn.model_selection import train_test_split
#X_train, X_cv, Y_train, Y_cv = train_test_split(X_train, Y_train, test_size = 0.05, random_state = 33)

###############################################################################################################
#statsmodels.formula.api performance checking
X_sm = np.append(arr = np.ones((X_train.shape[0], 1)), values = X_train, axis = 1)

x_opt = X_sm[:, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 
                             29, 30, 31, 32, 33, 34, 35, 36, 37, 
                             38, 40, 41, 42, 43, 44, 45, 46, 
                             47, 49, 50, 52, 53]]

regressor_ols = sm.OLS(endog = Y_train, exog = X_sm).fit()
regressor_ols.summary()

#print('[ ', end = '')
#for i in list(range(54)):
#    print(i, end = ', ')
#print(']')






##################################XGB MODEL


ii, jj = 3500, 4

model = XGBRegressor(n_estimators = ii, n_jobs = 6, max_depth = jj, learning_rate = 0.05)
model.fit(X_train, Y_train)

pred_train = model.predict(X_train).reshape((X_train.shape[0], 1))
pred_test = model.predict(X_test)

output = {'Price': pred_test
  }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(ii)+'_'+str(jj)+'_final.xlsx', index = None, header = True)

#####################################

ii, jj = 3000, 4

model = XGBRegressor(n_estimators = ii, n_jobs = 6, max_depth = jj, learning_rate = 0.05)
model.fit(np.append(arr = np.ones((X_train.shape[0], 1)), values = X_train, axis = 1), Y_train)

#pred_train = model.predict(X_train).reshape((X_train.shape[0], 1))
pred_test = model.predict(np.append(arr = np.ones((X_test.shape[0], 1)), values = X_test, axis = 1))

output = {'Price': pred_test
  }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(ii)+'_'+str(jj)+'_sm.xlsx', index = None, header = True)


#Improved

X_sm = np.append(arr = np.ones((X_train.shape[0], 1)), values = X_train, axis = 1)

x_opt = X_sm[:, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 
                             29, 30, 31, 32, 33, 34, 35, 36, 37, 
                             38, 40, 41, 42, 43, 44, 45, 46, 
                             47, 49, 50, 52, 53]]

X_test_sm = np.append(arr = np.ones((X_test.shape[0], 1)),
                      values = X_test, axis = 1)[:, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 
                      11, 12, 13, 14, 15, 16, 17, 18, 19,
                      21, 22, 23, 24, 25, 26, 27, 28, 
                      29, 30, 31, 32, 33, 34, 35, 36, 37, 
                      38, 40, 41, 42, 43, 44, 45, 46, 
                      47, 49, 50, 52, 53]]

ii, jj = 2500, 4

model = XGBRegressor(n_estimators = ii, n_jobs = 6, max_depth = jj, learning_rate = 0.05)
model.fit(x_opt, Y_train)

#pred_train = model.predict(X_train).reshape((X_train.shape[0], 1))
pred_test = model.predict(X_test_sm)

output = {'Price': pred_test
  }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(ii)+'_'+str(jj)+'_optimized.xlsx', index = None, header = True)


#Improved 2
dataset = np.concatenate((name,                                           #32
                          location,                                       #10
                          year,                                           #1
                          year*year,                                      #1
                          kilometers_driven,                              #1
                          kilometers_driven*kilometers_driven,            #1
                          fuel_type,                                      #3
                          transmission,                                   #1
                          owner_type,                                     #1
                          mileage,                                        #1
                          mileage*mileage,                                 #1
                          seats,                                          #1
                          engine,                                         #1
                          engine*engine,                                  #1
                          power,                                          #1
                          power*power                                     #1
                          ), axis = -1)


X_train, X_test = dataset[:6017], dataset[6017:]


ii, jj = 2750, 4

model = XGBRegressor(n_estimators = ii, n_jobs = 6, max_depth = jj, learning_rate = 0.05)
model.fit(X_train, Y_train)

pred_train = model.predict(X_train).reshape((X_train.shape[0], 1))
pred_test = model.predict(X_test)

output = {'Price': pred_test
  }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(ii)+'_'+str(jj)+'_improved2_poly.xlsx', index = None, header = True)


#Improved 3

dataset = np.concatenate((name,                                           #32
                          location,                                       #10
                          year,                                           #1
                          year*year,                                      #1
                          kilometers_driven,                              #1
                          kilometers_driven*kilometers_driven,            #1
                          fuel_type,                                      #3
                          transmission,                                   #1
                          owner_type,                                     #1
                          mileage,                                        #1
                          mileage*mileage                                 #1
                          seats,                                          #1
                          engine,                                         #1
                          engine*engine,                                  #1
                          power,                                          #1
                          power*power                                     #1
                          ), axis = -1)

X_train, X_test = dataset[:6017], dataset[6017:]

X_sm = np.append(arr = np.ones((X_train.shape[0], 1)), values = X_train, axis = 1)

x_opt = X_sm[:, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15, 16, 17, 18, 19,
                             21, 22, 23, 24, 25, 26, 27, 28, 
                             29, 30, 31, 32, 33, 34, 35, 36, 37, 
                             38, 40, 41, 42, 43, 44, 45, 46, 
                             47, 49, 50, 52, 53]]

X_test_sm = np.append(arr = np.ones((X_test.shape[0], 1)),
                      values = X_test, axis = 1)[:, [ 0, 3, 4, 5, 6, 7, 8, 9, 10, 
                      11, 12, 13, 14, 15, 16, 17, 18, 19,
                      21, 22, 23, 24, 25, 26, 27, 28, 
                      29, 30, 31, 32, 33, 34, 35, 36, 37, 
                      38, 40, 41, 42, 43, 44, 45, 46, 
                      47, 49, 50, 52, 53]]

ii, jj = 2500, 4

model = XGBRegressor(n_estimators = ii, n_jobs = 6, max_depth = jj, learning_rate = 0.05)
model.fit(x_opt, Y_train)

#pred_train = model.predict(X_train).reshape((X_train.shape[0], 1))
pred_test = model.predict(X_test_sm)

output = {'Price': pred_test
  }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(ii)+'_'+str(jj)+'_optimized.xlsx', index = None, header = True)


