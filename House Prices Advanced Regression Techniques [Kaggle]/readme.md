House Prices: Advanced Regression Techniques

Final Score: 1-0.11323 = 0.88677 or (88.67%)

url: https://www.kaggle.com/c/house-prices-advanced-regression-techniques

Hackathon Info: Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn and xgboost.

#########################################################################################
Column: -> applied LabelEncoder and OneHotEncoder        

Column: BsmtQual, 
        'BsmtCond, 
        BsmtExposure, 
        BsmtFinType1, 
        BsmtFinType2, 
        Exterior1st, 
        Exterior2nd,
        MSZoning, 
        Electrical, 
        SaleType,
        KitchenQual -> filled missing values with mode and applied LabelEncoder and OneHotEncoder


Column: MasVnrArea -> filled missing values with mean and applied StandardScaler

Column: BsmtFinSF1, 
        BsmtFinSF2, 
        BsmtUnfSF, 
        TotalBsmtSF, 
        BsmtFullBath, 
        'BsmtHalfBath, 
        GarageCars, 
        GarageArea -> filled missing values with median and appliied respective Scaling/Encoding

Column: MasVnrType -> Manual value filling with 'None' and applied Respective Scaling/Encoding

Column: FireplaceQu -> Filled missing values with 0 and applied custom Encoding according to ratings

Column: ExterQual, 
        ExterCond, 
        KitchenQual, 
        HeatingQC -> Applied custom Encoding accorting to ratings

Column: HouseStyle -> applied manual coding to 3x1 according to data_dictionary.txt

Column: Functional -> applied manual coding according to data_dictionary.txt

Column: GarageType,
        GarageFinish,
        GarageQual, 
        Utilities, 
        GarageCond,
        GarageYrBlt, 
         -> Manual value filling with '0' and applied Respective Scaling/Encoding

Column: YearGap -> Feature Engineering using (YearRemodAdd-YearBuilt), further applied LabelEncoder

Column(s): A lot of other columns have been transformed(encoded/scaled) according to the requirement and data_dictionary.txt

#########################################################################################

Applied XGBRegressor with max_depth = 5 and n_estimators = 750
Final Score: 1-0.11323 = 0.88677 or (88.67%)

Applied Linear Regressor
Final Score: 1- 0.18540 = 0.8145 or (81.45%)

Applied Bayesian Ridge Regressor with n_iter = 900, compute_score = True, 
                alpha_1 = 0.00001, alpha_2 = 0.00005, 
                lambda_1 = 0.00002, lambda_2 = 0.000001

Final Score: 1- 0.1576 = 0.8424 or (84.24%)

