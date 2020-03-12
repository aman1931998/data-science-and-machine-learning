from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import datetime
from sklearn.preprocessing import StandardScaler

train = pd.read_csv("train.csv")
test = pd.read_csv("test_bqCt9Pv.csv")

######################    Checking for nan values    ##########################
missing_train = train.isna().sum()    #found Employment.Type with nan
missing_test = test.isna().sum()     #found Employment.Type with nan

##############################    Y_train    ##################################
Y_train = train.pop('loan_default').values

##############################    dataset    ##################################
dataset = train.append(test)
del test, train

#Throwaway data
dataset.pop('PERFORM_CNS.SCORE.DESCRIPTION')


#########################    disbursed_amount    ##############################
disbursed_amount = dataset['disbursed_amount'].values.reshape((dataset.shape[0], 1))
dataset.pop('disbursed_amount')

sc_disbursed_amount = StandardScaler()
disbursed_amount = sc_disbursed_amount.fit_transform(disbursed_amount)

###########################    asset_cost    ##################################
asset_cost = dataset['asset_cost'].values.reshape((dataset.shape[0], 1))
dataset.pop('asset_cost')

sc_asset_cost = StandardScaler()
asset_cost = sc_asset_cost.fit_transform(asset_cost)

#Multiply by 100?


#############################    ltv    #######################################
ltv = dataset['ltv'].values.reshape((dataset.shape[0], 1))
dataset.pop('ltv')

sc_ltv = StandardScaler()
ltv = sc_ltv.fit_transform(ltv)

##########################    branch_id    ####################################
branch_id = dataset['branch_id'].values.reshape((dataset.shape[0], 1))
dataset.pop('branch_id')

##checking number of classes/branches
#x = []
#for i in branch_id: x.append(i[0])
#x = list(set(x))
#del i, x

ohe_branch_id = OneHotEncoder(categorical_features = [0])
branch_id = ohe_branch_id.fit_transform(branch_id).toarray()[:,:-1]

###########################    supplier_id    #################################
#supplier_id = dataset['supplier_id'].values.reshape((dataset.shape[0], 1))
#dataset.pop('supplier_id')
#
###checking number of classes/branches
##x = []
##for i in supplier_id: x.append(i[0])
##x = list(set(x))
##del i, x
#
#ohe_supplier_id = OneHotEncoder(categorical_features = [0])
#supplier_id = ohe_supplier_id.fit_transform(supplier_id).toarray()[:,:-1]
#supplier_id = np.array(supplier_id, dtype = np.int32)
#
##3088 columns? wtf

##########################    manufacturer_id    ##############################
manufacturer_id = dataset['manufacturer_id'].values.reshape((dataset.shape[0], 1))
dataset.pop('manufacturer_id')

##checking number of classes/branches
#x = []
#for i in manufacturer_id: x.append(i[0])
#x = list(set(x))
#del i, x

#Use only if Memory is sufficient enough
ohe_manufacturer_id = OneHotEncoder(categorical_features = [0])
manufacturer_id = ohe_manufacturer_id.fit_transform(manufacturer_id).toarray()[:,:-1]


#########################    Current_pincode_ID    #############################
#Current_pincode_ID = dataset['Current_pincode_ID'].values.reshape((dataset.shape[0], 1))
#dataset.pop('Current_pincode_ID')
##
###checking number of classes/branches
##x = []
##for i in Current_pincode_ID: x.append(i[0])
##x = list(set(x))
##del i, x         #No chance
#
##Use only if Memory is sufficient enough
##ohe_Current_pincode_ID = OneHotEncoder(categorical_features = [0])
##Current_pincode_ID = ohe_Current_pincode_ID.fit_transform(Current_pincode_ID).toarray()[:,:-1]
#
#del Current_pincode_ID
#
##7095 columns wtf?
#
##########################    Date.of.Birth    ################################
Date_of_Birth = dataset['Date.of.Birth'].values.reshape((dataset.shape[0], 1))
dataset.pop('Date.of.Birth')

##Temp
#x = 99
#for i in Date_of_Birth:
#    if(int(i[0][-2:])<30): continue
#    if(int(i[0][-2:])<x):
#        x = int(i[0][-2:])
#del x, i
#
##checking number of classes
#d = {}
#for i in Date_of_Birth:
#    if(i[0][-2:] in d.keys()): d[(i[0][-2:])]+=1
#    else: d[(i[0][-2:])] = 1
#del d, i

#Refining Data with base date 1-Jan-1945
l = []
base_date = datetime.datetime(1945, 1, 1)
for i in Date_of_Birth:
    if(int(i[0][-2:]) == 0):
        x = i[0][:-2] + '20' + i[0][-2:]
        l.append(datetime.datetime.strptime(x, '%d-%m-%Y'))
    else:
        x = i[0][:-2] + '19' + i[0][-2:]
        l.append(datetime.datetime.strptime(x, '%d-%m-%Y'))
Date_of_Birth = []
for i in l:
    Date_of_Birth.append((i - base_date).days) #//30?

Date_of_Birth = np.array(Date_of_Birth, dtype = np.float64).reshape((dataset.shape[0], 1))

sc_Date_of_Birth = StandardScaler()
Date_of_Birth = sc_Date_of_Birth.fit_transform(Date_of_Birth)


del i, base_date, l, x

#########################    Employment.Type    ###############################
#This column has a lot of missing values. Consider??
Employment_Type = dataset['Employment.Type']

#check for missing data
Employment_Type.isna().sum() #11104 values

#Fill with 'MODE'
Employment_Type.fillna(Employment_Type.mode()[0], inplace = True)
Employment_Type.isna().sum() #0 values

Employment_Type = Employment_Type.values.reshape((dataset.shape[0], 1))
dataset.pop('Employment.Type')

lb_Employment_Type = LabelEncoder()
Employment_Type = lb_Employment_Type.fit_transform(Employment_Type).reshape((dataset.shape[0], 1))
ohe_Employment_Type = OneHotEncoder()
Employment_Type = ohe_Employment_Type.fit_transform(Employment_Type).toarray()[:, 1:]

#del Employment_Type


##########################    DisbursalDate    ################################
DisbursalDate = dataset['DisbursalDate'].values.reshape((dataset.shape[0], 1))
dataset.pop('DisbursalDate')
#
##Temp
#x = 99
#for i in DisbursalDate:
#    if(int(i[0][-2:])<x):
#        x = int(i[0][-2:])
#del x, i
#
##checking number of classes
#d = {}
#for i in Date_of_Birth:
#    if(i[0][-2:] in d.keys()): d[(i[0][-2:])]+=1
#    else: d[(i[0][-2:])] = 1
#del d, i

#Refining Data with base date 1-Jan-1945
l = []
base_date = datetime.datetime(2018, 1, 1)
for i in DisbursalDate:
    x = i[0][:-2] + '20' + i[0][-2:]
    l.append(datetime.datetime.strptime(x, '%d-%m-%Y'))
DisbursalDate = []
for i in l:
    DisbursalDate.append((i - base_date).days) #//30?

DisbursalDate = np.array(DisbursalDate, dtype = np.float64).reshape((dataset.shape[0], 1))

sc_DisbursalDate = StandardScaler()
DisbursalDate = sc_DisbursalDate.fit_transform(DisbursalDate)

del i, base_date, l, x

#############################    State_ID    ##################################
State_ID = dataset['State_ID'].values.reshape((dataset.shape[0], 1))
dataset.pop('State_ID')

##checking number of classes
#d = {}
#for i in State_ID:
#    if(i[0] in d.keys()): d[(i[0])]+=1
#    else: d[(i[0])] = 1
#del d, i


ohe_State_ID = OneHotEncoder(categorical_features = [0])
State_ID = ohe_State_ID.fit_transform(State_ID).toarray()[:,:-1]


#########################    Employee_code_ID    ###############################
#Employee_code_ID = dataset['Employee_code_ID'].values.reshape((dataset.shape[0], 1))
#dataset.pop('Employee_code_ID')
#
###checking number of classes
##d = {}
##for i in Employee_code_ID:
##    if(i[0] in d.keys()): d[(i[0])]+=1
##    else: d[(i[0])] = 1
##del d, i
#
#lb_Employee_code_ID = LabelEncoder()
#Employee_code_ID= lb_Employee_code_ID.fit_transform(Employee_code_ID).reshape((dataset.shape[0], 1))
#
##ohe_Employee_code_ID = OneHotEncoder(categorical_features = [0])
##Employee_code_ID= ohe_Employee_code_ID.fit_transform(Employee_code_ID).toarray()[:,:-1]
##Employee_code_ID = np.array(Employee_code_ID, dtype = np.int32)
#
##del Employee_code_ID
#
##3397?
#
#######################    CustomerData_Flags    ###############################
Flags = dataset.iloc[:, :6].values
dataset.pop('MobileNo_Avl_Flag')
dataset.pop('Aadhar_flag')
dataset.pop('PAN_flag')
dataset.pop('VoterID_flag')
dataset.pop('Driving_flag')
dataset.pop('Passport_flag')

#del Flags

######################    PERFORM_CNS.SCORE    ################################
Perform_CNS_SCORE = dataset['PERFORM_CNS.SCORE'].values.reshape((dataset.shape[0], 1))
dataset.pop('PERFORM_CNS.SCORE')

sc_Perform_CNS_SCORE = StandardScaler()
Perform_CNS_SCORE = sc_Perform_CNS_SCORE.fit_transform(Perform_CNS_SCORE)

#del Perform_CNS_SCORE


########################     #Primary and Secondary     #######################
PnS = dataset.iloc[:, :12].values
dataset.pop( 'PRI.NO.OF.ACCTS')
dataset.pop( 'PRI.ACTIVE.ACCTS')
dataset.pop( 'PRI.OVERDUE.ACCTS')
dataset.pop( 'PRI.CURRENT.BALANCE')
dataset.pop( 'PRI.SANCTIONED.AMOUNT')
dataset.pop( 'PRI.DISBURSED.AMOUNT')
dataset.pop( 'SEC.NO.OF.ACCTS')
dataset.pop( 'SEC.ACTIVE.ACCTS')
dataset.pop( 'SEC.OVERDUE.ACCTS')
dataset.pop( 'SEC.CURRENT.BALANCE')
dataset.pop( 'SEC.SANCTIONED.AMOUNT')
dataset.pop( 'SEC.DISBURSED.AMOUNT')

#del PnS

###########################    Installment    #################################
Installment = dataset.iloc[:, :2].values
dataset.pop('PRIMARY.INSTAL.AMT')
dataset.pop('SEC.INSTAL.AMT')

Installment = StandardScaler().fit_transform(Installment)

#del Installment

#############################    Accs     #####################################
Accs = dataset.iloc[:, :2].values
dataset.pop('NEW.ACCTS.IN.LAST.SIX.MONTHS')
dataset.pop('DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS')

#del Accs

#########################    number_of_inquires    ############################
inquires = dataset['NO.OF_INQUIRIES'].values.reshape((dataset.shape[0], 1))
dataset.pop('NO.OF_INQUIRIES')

#del inquires

###########################    Loan tenure     ################################
loan_tenure = dataset['AVERAGE.ACCT.AGE'].values.reshape((dataset.shape[0], 1))
dataset.pop('AVERAGE.ACCT.AGE')

l = []
for i in loan_tenure:
    l.append(i[0].split())

loan_tenure = []
for i in l:
    loan_tenure.append(int(i[0][:-3]) * 12 + int(i[1][:-3]))
del i, l

loan_tenure = np.array(loan_tenure).reshape((dataset.shape[0], 1))
loan_tenure = StandardScaler().fit_transform(loan_tenure)
#del loan_tenure

#######################    CREDIT.HISTORY.LENGTH    ###########################
credit = dataset['CREDIT.HISTORY.LENGTH'].values.reshape((dataset.shape[0], 1))
dataset.pop('CREDIT.HISTORY.LENGTH')

l = []
for i in credit:
    l.append(i[0].split())

credit = []
for i in l:
    credit.append(int(i[0][:-3]) * 12 + int(i[1][:-3]))
del i, l

credit = np.array(credit).reshape((dataset.shape[0], 1))
credit = StandardScaler().fit_transform(credit)

#del credit

############################    MERGE    ######################################
dataset = np.concatenate((Accs, Date_of_Birth, DisbursalDate, Flags, Installment, Employment_Type, 
                          Perform_CNS_SCORE, 
                          PnS, State_ID, asset_cost, branch_id,
                          credit, disbursed_amount, inquires,
                          loan_tenure, ltv, manufacturer_id), axis = 1)
del Accs, Date_of_Birth, DisbursalDate, Flags, Installment
del Perform_CNS_SCORE, Employment_Type
del PnS, State_ID, asset_cost, branch_id
del credit, disbursed_amount, inquires
del loan_tenure, ltv, manufacturer_id

#dataset = np.concatenate((dataset, supplier_id, Employee_code_ID), axis = 1) ????

#np.save('npy_files/dataset.npy', dataset)
#dataset = np.load("npy_files/dataset.npy")

############################# MODEL  ##########################################

X_train, X_cv, X_test = dataset[:215154, :], dataset[215154:233154, :], dataset[233154:, :]
Y_train, Y_cv = Y_train[:215154], Y_train[215154:]
del dataset

l = []

#for i in range(5000, 6051, 150):
#    for j in range(4, 7, 1):
#
#temp = [i, j]

xgb = XGBClassifier(
        max_depth = 6, 
        n_estimators = 4500, 
        n_jobs = 6
        )

xgb.fit(X_train, Y_train)
Y_pred = xgb.predict(X_test)
Y_train_pred = xgb.predict(X_train)
Y_cv_pred = xgb.predict(X_cv)


cm_train = 0
for i in range(215154):
    if(Y_train[i] == Y_train_pred[i]):
        cm_train+=1

cm_cv = 0
for i in range(18000):
    if(Y_train[i] == Y_train_pred[i]):
        cm_cv+=1

print('Performance on CV data: ', cm_cv/18000)
print('Performance on Train data: ', cm_train/215154)



Y_pred = Y_pred.reshape((Y_pred.shape[0], 1))

###############################################################################
train = pd.read_csv("train.csv")
test = pd.read_csv("test_bqCt9Pv.csv")
Y_train = train.pop('loan_default').values
dataset = train.append(test)
del test, train
        
output = []

temp = dataset.pop('UniqueID').values.reshape((dataset.shape[0], 1))
for i in range(Y_pred.shape[0]):
    output.append([int(temp[i][0]), int(Y_pred[i][0])])
        
df = pd.DataFrame(output, columns = ['UniqueID', 'loan_default'])
df.to_csv(r'Output.csv', index = None, header = True)
