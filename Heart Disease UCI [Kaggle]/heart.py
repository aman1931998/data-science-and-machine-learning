import tensorflow as tf
import numpy as np
import pandas as pd


#Loading the dataset
data = pd.read_csv("heart.csv")
dataset = data.values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#cp
onehot1 = OneHotEncoder(categorical_features = [2])
dataset = onehot1.fit_transform(dataset).toarray()[:,1:] #null_trap 

#restecg
onehot2 = OneHotEncoder(categorical_features = [8])
dataset = onehot2.fit_transform(dataset).toarray()[:,1:] #null_trap 

#slope
onehot3 = OneHotEncoder(categorical_features = [13])
dataset = onehot3.fit_transform(dataset).toarray()[:,1:] #null_trap 

#thal
onehot4 = OneHotEncoder(categorical_features = [16])
dataset = onehot4.fit_transform(dataset).toarray()[:,1:] #null_trap 



#Test train Split?~
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size = 0.2, random_state = 42)
#Y_train = np.reshape(Y_train, (242, 1))
#Y_test = np.reshape(Y_test, (61, 1))



#2, 0.075, 200
#Model
from xgboost import XGBClassifier
xbg = XGBClassifier(max_depth = 2, learning_rate = 0.075, n_estimators = 200, silent = False, n_jobs = 2)
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
#Train Percentage: 0.946
#Test Percentage: 0.869
