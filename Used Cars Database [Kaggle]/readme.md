Kaggle - Used cars database

Train Error: 0.5065
Test Error: 0.4754

url: https://www.kaggle.com/orgesleka/used-cars-database

Hackathon Info: Over 370000 used cars scraped with Scrapy from Ebay-Kleinanzeigen. The content of the data is in german, so one has to translate it first if one can not speak german.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, datetime, bs4, requests, itertools and xgboost.

#########################################################################################
Data is encoded with ISO-8859-1, so while importing dataset, we use parameter -> encoding = 'ISO-8859-1'

Column: seller,
        offerType, 
        abtest, 
        brand -> applied LabelEncoder and OneHotEncoder

Column: yearOfRegistration -> applied LabelEncoder

Column: powerPS, 
        kilometer -> applied StandardScaler

Column: monthOfRegistration -> no change

Column: price -> Target Variable

Column: dateCreated, 
        dateCrawled, 
        lastSeen -> skipped for now

Column: postalCode -> skipped for now -> try converting postal codes to states

Column: name, 
        notRepairedDamage, 
        nrOfPictures -> ignored

Column: vehicleType, 
        gearbox, 
        model, 
        fuelType -> skipped for now -> try filling missing values

#########################################################################################
Applied XGBRegressor with max_depth = 5 and n_estimators = 1000 and learning_rate = 0.00005

Train Error: 0.5065
Test Error: 0.4754

