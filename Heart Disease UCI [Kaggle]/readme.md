Heart Disease UCI

FINAL Score: 0.869 or (87%)

url: https://www.kaggle.com/ronitf/heart-disease-uci

Hackathon Info: This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.

Methodology used: 2 Implementations of same Approach, 1 using numpy arrays and another using pandas dataFrame

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, and xgboost.

#############################  IMPLEMENTATION 1  ######################################
Column: age, 
        trestbps, 
        chol, 
        thalach, 
        oldpeak -> applied standard Scaler

Column: sex, 
        cp, 
        fbs, 
        restecg, 
        slope, 
        thal -> OneHotEncoded using pandas.get_dummies

Applied XGBRegressor with max_depth = 3, learning_rate = 0.00025, n_estimators = 450

Train Percentage: 0.860
Test Percentage: 0.836

#############################  IMPLEMENTATION 2  ######################################
Necessary columns are converted to OneHot data.

Applied XGBRegressor with max_depth = 2 and n_estimators = 200 and learning_rate = 0.075

Train Percentage: 0.946
Test Percentage: 0.869
