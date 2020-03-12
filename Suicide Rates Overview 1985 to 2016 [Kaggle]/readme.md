Suicide Rates Overview 1985 to 2016

Train Error: 1.9764
Test Error: 8.5398

url: https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016

Hackathon Info: This compiled dataset pulled from four other datasets linked by time and place, and was built to find signals correlated to increased suicide rates among different cohorts globally, across the socio-economic spectrum.

Methodology used: XGBoost with data-preprocessing

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, datetime and xgboost.

#########################################################################################
Column: country, 
        generation -> applied LabelEncoder and OneHotEncoder

Column: sex -> applied LabelEncoder

Column: population, 
        suicides/100k, 
        gdp_per_capita -> applied StandardScaler

Column: age -> applied LabelEncoder manually but looking at the 6 unique items.

Column: sex -> applied LabelEncoder and StandardScaler

Column: gdp_for_year -> converted data to integers manually and applied StandardScaler

Column: suicides_no -> This will be the dependent_variable

#########################################################################################

Applied XGBRegressor max_depth =6 and n_estimators =  700

Train Error: 1.9764
Test Error: 8.5398

