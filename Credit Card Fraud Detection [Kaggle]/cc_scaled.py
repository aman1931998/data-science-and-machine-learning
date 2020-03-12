import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


#Loading the dataset
dataset = pd.read_csv("creditcard.csv")

dataset = dataset.sample(frac = 1)

fraud_data = dataset.loc[dataset['Class'] == 1]
non_fraud_data = dataset.loc[dataset['Class'] == 0][:1000]

dataset = np.concatenate((fraud_data, non_fraud_data), axis = 0)

#Test train Split
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, :-1], dataset[:, -1], test_size = 0.15)

xbg = XGBClassifier(
                    max_depth = 3,
                    learning_rate = 0.1,
                    n_estimators = 200,
                    n_jobs = 6,
                    )
xbg.fit(X_train, Y_train)

Y_test_pred = xbg.predict(X_test)
Y_train_pred = xbg.predict(X_train)


count_train, count_test = 0, 0
for i in range(836):
    if(Y_train[i] == Y_train_pred[i]): count_train+=1

for i in range(148):
    if(Y_test[i] == Y_test_pred[i]): count_test+=1

print("Train Percentage: %.7f\nTest Percentage: %.7f"%(count_train/836, count_test/148))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_train = confusion_matrix(Y_train, Y_train_pred)
cm_test = confusion_matrix(Y_test, Y_test_pred)

print("Train CM:", cm_train, "Test CM:", cm_test, sep = '\n')

