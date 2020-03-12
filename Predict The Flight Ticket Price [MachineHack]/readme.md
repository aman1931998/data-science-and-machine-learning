Predict The Flight Ticket Price Hackathon

FINAL Score: 0.9352810 (or 93.52%)

url: https://www.machinehack.com/course/predict-the-flight-ticket-price-hackathon/

Hackathon Info: Flight ticket prices can be something hard to guess, today we might see a price, check out the price of the same flight tomorrow, it will be a different story. We might have often heard travellers saying that flight ticket prices are so unpredictable. Huh! Here we take on the challenge! As data scientists, we are gonna prove that given the right data anything can be predicted. Here you will be provided with prices of flight tickets for various airlines between the months of March and June of 2019 and between various cities.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, datetime, bs4, requests, itertools and xgboost.

#########################################################################################
Column: Airline -> applied LabelEncoder and OneHotEncoder

Column: Duration -> converted time to minutes (or hours and minutes separately)

Column: Total_Stops -> applied LabelEncoder and OneHotEncoder

Column: Date_of_Journey -> applied datetime module and converted to number of days w.r.t. JAN 01, 2019.

Column: Delay_in_Journey -> used date_of_journey, departure_time and arrival_time to calculate the delay in scheduled flight and actual flight.

Column: Path -> used Route column to determine the path of source to destination, then I found the distance between each 2 stations in the route using web-scraping off website https://www.prokerala.com. (refer to code to understand my concept)
Column: Additional_Info -> Reduced the number of classes_ and applied LabelEncoder and OneHotEncoder.
#########################################################################################

Applied StandardScaler to final dataset (or individual if necessary)

Applied XGBRegressor with max_depth = 5 and n_estimators = 960

FINAL Score: 0.9352810 (or 93.52%)

