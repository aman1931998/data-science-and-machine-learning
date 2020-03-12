Credit Card Fraud Detection

FINAL Score: 0.98 or (98%)

url: https://www.kaggle.com/mlg-ulb/creditcardfraud

Hackathon Info: It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

Methodology used: 

Implemented using python programming language, modules used are pandas, numpy, scikit-learn and xgboost.

################################  Implementation 1  ################################
Gather Sense of Our Data:
Except for the transaction and amount we dont know what the other columns are (due to privacy reasons). The only thing we know, is that those columns that are unknown have been scaled already.

The transaction amount is relatively small. Mean -> USD 88.
No "Null" values
Non-Fraud (99.83%)
Fraud (0.17%)

Applied XGBRegressor with max_depth = 5, learning_rate = 0.1 and n_estimators = 1500

Train Percentage: 1.0000000
Test Percentage: 0.9995691

Train CM:
[ [249569      0]
  [     0    431] ]
Test CM:
[ [34744     2]
  [   13    48] ]



################################  Implementation 2  ################################
Since we have 492 +ve and 250k -ve txns, we need to scale it.
We can choose a random of another 492 or 884 txns and create a new dataset

Applying XGBRegressor with max_depth = 3, n_estimators = 200

Train Percentage: 1.0000000
Test Percentage: 0.9729730
Train CM:
[[857   0]
 [  0 411]]
Test CM:
[[143   0]
 [  9  72]]