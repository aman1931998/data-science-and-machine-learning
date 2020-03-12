import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

#Questions ->
#    Apply Scaling???
#    MinMax or Standard

#Loading the dataset
data = pd.read_csv("heart.csv")

missing = data.isna().sum()  #0 missing

#age
age = MinMaxScaler().fit_transform(data.pop('age').values.reshape((data.shape[0], 1))).reshape((data.shape[0], 1))

#sex
data['sex'][data['sex'] == 0] = 'female'
data['sex'][data['sex'] == 1] = 'male'

#cp
data['cp'][data['cp']==0] = 'typical angina'
data['cp'][data['cp']==1] = 'atypical angina'
data['cp'][data['cp']==2] = 'non-typical angina'
data['cp'][data['cp']==3] = 'asymptomatic'

#trestbps
trestbps = MinMaxScaler().fit_transform(data.pop('trestbps').values.reshape((data.shape[0], 1))).reshape((data.shape[0], 1))

#chol
chol = MinMaxScaler().fit_transform(data.pop('chol').values.reshape((data.shape[0], 1))).reshape((data.shape[0], 1))

#fbs
data['fbs'][data['fbs'] == 0] = 'below 120 mg/dl'
data['fbs'][data['fbs'] == 1] = 'above 120 mg/dl'

#restecg
data['restecg'][data['restecg'] == 0] = 'normal'
data['restecg'][data['restecg'] == 1] = 'ST-T wave abnormality'
data['restecg'][data['restecg'] == 2] = 'left ventricular hypertrophy'

#thalach
thalach = MinMaxScaler().fit_transform(data.pop('thalach').values.reshape((data.shape[0], 1))).reshape((data.shape[0], 1))

#exang
#No changes needed

#oldpeak
oldpeak = MinMaxScaler().fit_transform(data.pop('oldpeak').values.reshape((data.shape[0], 1))).reshape((data.shape[0], 1))

#slope
data['slope'][data['slope'] == 0] = 'upsloping'
data['slope'][data['slope'] == 1] = 'flat'
data['slope'][data['slope'] == 2] = 'downsloping'

#ca
#No changes needed

#thal
data['thal'][data['thal'] == 0] = 'Unknown'
data['thal'][data['thal'] == 1] = 'normal'
data['thal'][data['thal'] == 2] = 'fixed defect'
data['thal'][data['thal'] == 3] = 'reversable defect'

###############################################################################
#Converting data to OneHotDummies
data = pd.get_dummies(data, drop_first = True)

#Creating X and Y
Y = data.pop('target').values

X = np.concatenate((data.values, age, chol, oldpeak, thalach, trestbps), axis = 1)

np.save('npy_files/X.npy', X)
np.save('npy_files/Y.npy', Y)

del data, age, chol, oldpeak, thalach, trestbps
del missing

#Test train Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,  test_size = 0.2, random_state = 32)

np.save('npy_files/X_train.npy', X_train)
np.save('npy_files/X_test.npy', X_test)
np.save('npy_files/Y_train.npy', Y_train)
np.save('npy_files/Y_test.npy', Y_test)

del X, Y

#Model
from xgboost import XGBClassifier

xbg = XGBClassifier(max_depth = 3, learning_rate = 0.00025, n_estimators = 450, silent = False, n_jobs = 6)
xbg.fit(X_train, Y_train)

Y_pred = xbg.predict(X_test)
Y_train_pred = xbg.predict(X_train)

count_train, count_test = 0, 0

for i in range(242):
    if(Y_train[i] == Y_train_pred[i]): count_train+=1

for i in range(61):
    if(Y_test[i] == Y_pred[i]): count_test+=1

print("Train Percentage: %.3f\nTest Percentage: %.3f"%(count_train/242, count_test/61))


########################### OUTPUT  ##########################################
Train Percentage: 0.860
Test Percentage: 0.836
