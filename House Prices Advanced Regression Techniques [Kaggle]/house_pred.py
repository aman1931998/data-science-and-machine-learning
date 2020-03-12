import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Dropping Id
train = train.drop('Id', axis = 1)
test = test.drop('Id', axis = 1)

#adding fake SalePrice
test = test.assign(SalePrice = 0)

#Concatenating both DS
dataset = pd.concat([train, test], axis = 0)

#Finding missing data
missing = dataset.isna().sum()

#LotFrontage
#lotfrontage = dataset['LotFrontage']

dataset.pop('LotFrontage')

#Alley
dataset.pop('Alley')

#GarageType, GarageFinish, GarageQual, GarageCond, GarageYrBlt, FireplaceQu
dataset['GarageType'] = dataset['GarageType'].fillna('0') #based on observation
dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(0) #based on observation
dataset['GarageFinish'] = dataset['GarageFinish'].fillna('0') #based on observation
dataset['GarageQual'] = dataset['GarageQual'].fillna('0') #based on observation
dataset['GarageCond'] = dataset['GarageCond'].fillna('0') #based on observation
dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna(0) #based on observation
dataset['Functional'] = dataset['Functional'].fillna('Typ') #based on observation



#Utilities #consider karu??
dataset['Utilities'].value_counts()  #checking for values
dataset['Utilities'] = dataset['Utilities'].fillna('AllPub') #based on observation
dataset['Utilities'].isna().sum()

#HouseStyle
HouseStyle = dataset['HouseStyle']

HouseStyle.value_counts()  #checking for values

HouseStyle = HouseStyle.values
#Method 1 -> Manual Mapping/Encoding
d = {'1Story': [1, 0, 0], 
     '2Story': [0, 1, 0], 
     '1.5Fin': [3, 0, 0], 
     'SLvl'  : [0, 0, 1], 
     'SFoyer': [0, 0, 2], 
     '2.5Unf': [0, 2, 0], 
     '1.5Unf': [2, 0, 0], 
     '2.5Fin': [0, 3, 0], 
}
HouseStyle = [d[i] for i in HouseStyle]
HouseStyle = np.array(HouseStyle)

dataset.pop('HouseStyle')
del d

#Feature Engg -> YearGap
YearGap = np.subtract(dataset['YearRemodAdd'].values, 
                      dataset['YearBuilt'].values).reshape((dataset.shape[0], 1))

########################Filling missing values with mode
missing_mode = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                'BsmtFinType2', 'Exterior1st', 'Exterior2nd', 'MSZoning', 
                'Electrical', 'SaleType', 'KitchenQual']

for i in missing_mode:
#    dataset[i].value_counts()  #checking for values
    dataset[i].fillna(dataset[i].mode()[0], inplace = True) #replacing values with 'mode'
#    dataset[i].isna().sum()
del missing_mode
########################Filliing missing values with mean
missing_mean = ['MasVnrArea', ]
for i in missing_mean:
#    dataset[i].value_counts()  #checking for values
    dataset[i].fillna(dataset[i].mean(), inplace = True) #replacing values with 'mode'
#    dataset[i].isna().sum()
del missing_mean
########################Filliing missing values with median
missing_median = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                  'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea', ]
for i in missing_median:
#    dataset[i].value_counts()  #checking for values
    dataset[i].fillna(dataset[i].median(), inplace = True) #replacing values with 'mode'
#    dataset[i].isna().sum()
del missing_median

#MasVnrType
dataset['MasVnrType'].value_counts()  #checking for values
dataset['MasVnrType'].fillna('None', inplace = True) #replacing values with 'mode'
dataset['MasVnrType'].isna().sum()


#ExterQual and ExterCond and others
d = {'Ex': 5, 
     'Gd': 4, 
     'TA': 3, 
     'Fa': 2, 
     'Po': 1, 
     0: 0
}
FireplaceQu = dataset['FireplaceQu']
ExterQual = dataset['ExterQual']
ExterCond = dataset['ExterCond']
HeatingQC = dataset['HeatingQC']
KitchenQual = dataset['KitchenQual']

FireplaceQu = FireplaceQu.values
ExterQual = ExterQual.values
ExterCond = ExterCond.values
KitchenQual = KitchenQual.values
HeatingQC = HeatingQC.values

#Method 1 -> Manual Mapping/Encoding
FireplaceQu = [d[i] for i in FireplaceQu]
FireplaceQu = np.array(FireplaceQu).reshape((dataset.shape[0], 1))

ExterQual = [d[i] for i in ExterQual]
ExterQual = np.array(ExterQual).reshape((dataset.shape[0], 1))

ExterCond = [d[i] for i in ExterCond]
ExterCond = np.array(ExterCond).reshape((dataset.shape[0], 1))

KitchenQual = [d[i] for i in KitchenQual]
KitchenQual = np.array(KitchenQual).reshape((dataset.shape[0], 1))

HeatingQC = [d[i] for i in HeatingQC]
HeatingQC = np.array(HeatingQC).reshape((dataset.shape[0], 1))

dataset.pop('ExterQual')
dataset.pop('ExterCond')
dataset.pop('KitchenQual')
dataset.pop('HeatingQC')
dataset.pop('FireplaceQu')
del d

########################OneHotEncoder
ohe_list = [ 'MSSubClass', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'MasVnrType', 
       'RoofStyle', 'Utilities', 'Exterior1st', 'Exterior2nd', 
       'RoofMatl', 'Foundation', 'MSZoning', 
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
       'Heating', 'CentralAir', 'PavedDrive', 'SaleCondition', 'Electrical', 'SaleType', 
       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', ]
ohe_op = []
for i in ohe_list:
    ohe_op.append(OneHotEncoder(categorical_features = [0]).fit_transform(LabelEncoder().fit_transform(dataset[i].values).reshape((dataset.shape[0], 1))).toarray()[:, 1:])
    dataset.pop(i)
final_ohe_op = ohe_op[0]
for i in ohe_op[1:]:
    final_ohe_op = np.concatenate((final_ohe_op, i), axis = 1)

del i, ohe_op, ohe_list


########################StandardScaler
sc_list = ['LotArea', 'MasVnrArea', 
           'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
           '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 
           'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF',  'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 
           ]
sc_op = []
for i in sc_list:
    sc_op.append(StandardScaler().fit_transform(dataset[i].values.reshape((dataset.shape[0], 1))).ravel())
    dataset.pop(i)
sc_op = np.array(sc_op).T
del i, sc_list, 

########################LabelEncoder
le_list = ['Street', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
           'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'MoSold', 'YrSold', 'GarageCars', 
           'GarageYrBlt', 
           ]
le_op = []
for i in le_list:
    le_op.append(LabelEncoder().fit_transform(dataset[i].values.reshape((dataset.shape[0], 1))).ravel())
    dataset.pop(i)
le_op = np.array(le_op).T
del le_list, i


#############################

d = {
'Typ'  :7,
'Min1' :6,
'Min2' :5,
'Mod'  :4,
'Maj1' :3,
'Maj2' :2,
'Sev'  :1,
'Sal'  :0 }
Functional = dataset['Functional'].values

Functional = [d[i] for i in Functional]
Functional = np.array(Functional).reshape((dataset.shape[0], 1))
dataset.pop('Functional')
del d

###############################################################################

#Creating Final Dataset

dataset = np.concatenate((ExterCond, ExterQual, FireplaceQu, Functional,
                          HeatingQC, HouseStyle, KitchenQual, YearGap,
                          final_ohe_op, le_op, sc_op), axis = 1)

del ExterCond, ExterQual, FireplaceQu, Functional, HeatingQC, HouseStyle, KitchenQual, YearGap, final_ohe_op, le_op, sc_op

#np.save('npy_files/dataset.npy', dataset)
#dataset = np.load('npy_files/dataset.npy')

X_train, X_test = dataset[:1460, :], dataset[1460:, :]
Y_train = pd.read_csv('train.csv').SalePrice.values

###############################################################################

xgbregressor = XGBRegressor(max_depth = 5,
                            learning_rate = 0.05,
                            n_estimators = 750,
                            n_jobs = 6,
                            silent = False)

xgbregressor.fit(X_train, Y_train)

#predicting on training data
Y_train_pred = xgbregressor.predict(X_train)

#Predictng
Y_pred = xgbregressor.predict(X_test)
        
#Saving test output
output = {'Id': pd.read_csv('test.csv')['Id'].values,
          'SalePrice': Y_pred
          }
df = pd.DataFrame(output, columns = ['Id', 'SalePrice'])
df.to_csv(r'Output_xgb.csv', index = None, header = True)


###############################################################################
linmodel = LinearRegression(normalize = False, n_jobs = 6)
#Fitting
linmodel.fit(X_train, Y_train)

#predicting on training data
Y_train_pred = linmodel.predict(X_train)

#Predictng
Y_pred = linmodel.predict(X_test)
        
#Saving test output
output = {'Id': pd.read_csv('test.csv')['Id'].values,
          'SalePrice': Y_pred
          }
df = pd.DataFrame(output, columns = ['Id', 'SalePrice'])
df.to_csv(r'Output_lin.csv', index = None, header = True)

###############################################################################
brmodel = BayesianRidge(n_iter = 900, 
                        compute_score = True, 
                        alpha_1 = 0.00001, 
                        alpha_2 = 0.00005, 
                        lambda_1 = 0.00002, 
                        lambda_2 = 0.000001)
#Fitting
brmodel.fit(X_train, Y_train)

#predicting on training data
Y_train_pred = brmodel.predict(X_train)

#Predictng
Y_pred = brmodel.predict(X_test)
        
#Saving test output
output = {'Id': pd.read_csv('test.csv')['Id'].values,
          'SalePrice': Y_pred
          }
df = pd.DataFrame(output, columns = ['Id', 'SalePrice'])
df.to_csv(r'Output_br.csv', index = None, header = True)

