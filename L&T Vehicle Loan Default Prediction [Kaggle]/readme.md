L&T Vehicle Loan Default Prediction

Final Score on CV data:  0.8946 (or 89.46%)
Final Score on Train data:  0.8963 (or 89.63%)

url: https://www.kaggle.com/gauravdesurkar/lt-vehicle-loan-default-prediction

Hackathon Info: Financial institutions incur significant losses due to the default of vehicle loans. This has led to the tightening up of vehicle loan underwriting and increased vehicle loan rejection rates. The need for a better credit risk scoring model is also raised by these institutions. This warrants a study to estimate the determinants of vehicle loan default. A financial institution has hired you to accurately predict the probability of loanee/borrower defaulting on a vehicle loan in the first EMI (Equated Monthly Instalments) on the due date. Following Information regarding the loan and loanee are provided in the datasets: Loanee Information (Demographic data like age, Identity proof etc.) Loan Information (Disbursal details, loan to value ratio etc.) Bureau data & history (Bureau score, number of active accounts, the status of other loans, credit history etc.) Doing so will ensure that clients capable of repayment are not rejected and important determinants can be identified which can be further used for minimising the default rates.

Methodology used:

Implemented using python programming language, modules used are pandas, numpy, scikit-learn, datetime and xgboost.

#########################################################################################
Column: disbursed_amount, 
        asset_cost, 
        ltv, 
        Perform_CNS_SCORE, 
        -> applied StandardScaler

Column: branch_id, 
        supplier_id, 
        Current_pincode_ID, 
        Employee_code_ID-> applied OneHotEncoder (#Use only if Memory is sufficient enough)

Column: State_ID, manufacturer_id -> applied OneHotEncoder

Column: Date_of_Birth -> convert date to integer values (days) w.r.t. 1st January 1945 and applied StandardScaler

Column: DisbursalDate-> convert date to integer values (days) w.r.t. 1st January 2018 and applied StandardScaler

Column: Employment_Type -> Filled missing values with 'MODE' and applied LabelEncoder and OneHotEncoder

Column: CustomerData_Flags -> Certain Flags with binary data

Column: Primary and Secondary (12 columns), 
        Installment (2 columns), 
        Accs (2 columns), 
        number_of_inquires, 
        loan_tenure, 
        -> No change 
Column: Loan_tenure, 
        Credit_History_Length -> Convert to months and applied StandardScaler

#########################################################################################

Applied XGBRegressor with max_depth = 6 and n_estimators = 4500

Final Score on CV data:  0.8946 (or 89.46%)
Final Score on Train data:  0.8963 (or 89.63%)
