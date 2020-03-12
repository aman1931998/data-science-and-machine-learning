Predicting The Costs Of Used Cars - Hackathon By Imarticus Learning

FINAL Score: 0.9388 (or 93.88%)

url: https://www.machinehack.com/course/predicting-the-costs-of-used-cars-hackathon-by-imarticus/

Hackathon Info: Driverless cars are getting closer to reality and at a faster pace than ever. But it is still a bit far fetched dream to have one in your garage. For the time being, there are still a lot of combustion and hybrid cars that roar around the road, for some it chills. Though the overall data on sales of automobiles shows a huge drop in sales in the last couple of years, cars are still a big attraction for many. Cars are more than just a utility for many. They are often the pride and status of the family. We all have different tastes when it comes to owning a car or at least when thinking of owning one.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, statsmodels.formula.api and xgboost.

#########################################################################################
Column: Mileage -> removed 2 missing values and applied StandardScaler. Similarly for Y_train

Column: Location, 
        Fuel_Type, -> applied LabelEncoder and OneHotEncoder

Column: Year,
        Kilometers_Driven, 
        Transmission, 
        Name, -> applied LabelEncoder

Column: Owner_Type -> manual LabelEncoder

Column: Engine, Power and Seats -> Manual Web Scraping off cardekho.com and carwale.com and some other sites to find missing values. Further applied StandardScaler.


#########################################################################################


Use: statsmodel.formula.api for finding the effective dependent variables of the dataset and filter out useless data.
Observation: Eliminated some columns and applied XGBRegressor

Applied XGBRegressor with max_depth = 4 and n_estimators = 2750

FINAL Score: 0.9352810 (or 93.52%)

