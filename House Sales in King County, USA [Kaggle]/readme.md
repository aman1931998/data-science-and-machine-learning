House Sales in King County, USA

Train Error:  0.89758
Test Error:  0.82106

url: https://www.kaggle.com/harlfoxem/housesalesprediction

Hackathon Info: This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, datetime and xgboost.

#########################################################################################
Column: price -> independent variable

Column: bedrooms, 
        bathrooms -> applied StandardScaler

Column: attached_bedrooms -> Feature Engineering from bedrooms and bathrooms -> max((bedrooms-bathrooms), 0)

Column: sqft_living, 
        sqft_lot, 
        sqft_above, 
        sqft_basement, 
        sqft_living15, 
        sqft_lot15, 
        grade, 
        condition, 
        view, 
        waterfront, 
        floors -> applied LabelEncoder

Column: yr_built,
        yr_renovated -> replaced values of yr_renovated to yr_built if yr_renovated != 0

Column: zipcode -> applied LabelEncoder and OneHotEncoder

Column: latitude, 
        longitude -> precision scaling upto 2 floating points and then applied LabelEncoder and OneHotEncoder

Column: date -> converted date to integer value of days w.r.t. to 1st January 2014, applied StandardScaler

#########################################################################################

Applied XGBRegressor with max_depth = 6, n_estimators = 2200 and learning_rate = 0.025

Train Error:  0.89758
Test Error:  0.82106
