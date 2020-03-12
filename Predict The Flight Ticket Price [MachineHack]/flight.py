from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import datetime
from bs4 import BeautifulSoup
import requests
from itertools import combinations
from sklearn.preprocessing import StandardScaler

train = pd.read_excel("Data_Train.xlsx")
test = pd.read_excel("Test_set.xlsx")
len_train = len(train)
len_test = len(test)

#################################Airline#######################################
airline = np.concatenate((train['Airline'].values, test['Airline'].values))
le = LabelEncoder()
airline = le.fit_transform(airline).reshape((len_train + len_test, 1))
ohe = OneHotEncoder(categorical_features = [0])
airline = ohe.fit_transform(airline).toarray()
airline = np.array(airline, dtype = np.float32)
airline_train, airline_test = airline[:len_train], airline[len_train:]
del airline
###############################################################################

################################Duration#######################################
def split(x): return x.split()

duration_train = train['Duration'].values
duration_test = test['Duration'].values

dtrain = list(map(split, duration_train))
duration_train = []
for i in dtrain:
    length = len(i)
    val = int(i[0][:-1]) * 60
    if(length == 2):
        val+= int(i[1][:-1])
    duration_train.append(val)
del dtrain

dtest = list(map(split, duration_test))
duration_test = []
for i in dtest:
    length = len(i)
    val = int(i[0][:-1]) * 60
    if(length == 2):
        val+= int(i[1][:-1])
    duration_test.append(val)
del dtest
del i, length, val
duration_train = np.array(duration_train, dtype = np.float32).reshape((len_train, 1))
duration_test = np.array(duration_test, dtype = np.float32).reshape((len_test, 1))
###############################################################################

###############################Total_Stops#####################################
train['Total_Stops'] = train['Total_Stops'].fillna('non-stop') #replacing 1 nan with 'non-stop' acc. to data

total_stops = np.concatenate((train['Total_Stops'].values, test['Total_Stops'].values))
le = LabelEncoder()
total_stops = le.fit_transform(total_stops).reshape((len_train + len_test, 1))
total_stops_train, total_stops_test = np.array(total_stops[:len_train], dtype = np.float32), np.array(total_stops[len_train:], dtype = np.float32)
del total_stops
###############################################################################

############################Date_of_Journey####################################
date_of_journey = np.concatenate((train['Date_of_Journey'].values, test['Date_of_Journey'].values))
l = []
base_date = datetime.datetime(2019, 1, 1)
for i in date_of_journey:
    l.append(datetime.datetime.strptime(i, '%d/%m/%Y'))
date_of_journey = []
for i in l: date_of_journey.append((i - base_date).days)
date_of_journey = np.array(date_of_journey, dtype = np.float32).reshape((len_train + len_test, 1))
date_of_journey_train, date_of_journey_test = date_of_journey[:len_train], date_of_journey[len_train:]

del i, base_date, date_of_journey, l
###############################################################################

###########################Delay_in_Journey####################################
date_of_journey = np.concatenate((train['Date_of_Journey'].values, test['Date_of_Journey'].values))
dep_time = np.concatenate((train['Dep_Time'].values, test['Dep_Time'].values))
arrival_time = np.concatenate((train['Arrival_Time'].values, test['Arrival_Time'].values))

dep_time = list(map(lambda x, y: y + " " + x, date_of_journey, dep_time))
arrival_time_ = list(map(lambda x: x + " 2019", arrival_time))

arrival_time = []
for i in range(len_train + len_test):
    if(len(arrival_time_[i].split()) == 4):
        arrival_time.append(datetime.datetime.strptime(arrival_time_[i], '%H:%M %d %b %Y'))
    else:
        arrival_time.append(datetime.datetime.strptime(arrival_time_[i][:6]+
                                                       dep_time[i][-10:-5].strip()+
                                                       arrival_time_[i][-5:],
                                                       '%H:%M %d/%m %Y'))
dep_time = list(map(lambda x:datetime.datetime.strptime(x, '%H:%M %d/%m/%Y'), dep_time))

real_time_of_journey = list(map(lambda x, y:(abs(x-y).days*24*60 + abs(x-y).seconds/60), dep_time, arrival_time))
real_time_of_journey = np.array(real_time_of_journey, dtype = np.float32).reshape((len_train + len_test, 1))
real_time_of_journey_train, real_time_of_journey_test = real_time_of_journey[:len_train], real_time_of_journey[len_train:]

#NEEDED TO CALL DURATION OF FLIGHTS

delay_train = np.subtract(real_time_of_journey_train, duration_train)
delay_test = np.subtract(real_time_of_journey_test, duration_test)

del date_of_journey, i, arrival_time_, arrival_time, dep_time, real_time_of_journey, duration_test, duration_train, real_time_of_journey_test, real_time_of_journey_train
###############################################################################

####################################Path#######################################
train['Route'] = train['Route'].fillna('DEL → BOM → COK') #replacing 1 nan with 'DEL → BOM → COK' acc to data

route = np.concatenate((train['Route'].values, test['Route'].values))
route = [(list(map(lambda x:x.strip(), i.split('→')))) for i in route]
x = set()
for i in route:
    for j in i:
        x.add(j)
airports = list(x)
airports_combinations = []
for i in combinations(airports, 2):
    airports_combinations.append(i)
del x, airports

c, distances = 0, []
for i, j in airports_combinations:
    url = 'https://www.prokerala.com/travel/airports/distance/from-' + i.lower() + '/to-' + j.lower() + '/'
    yelp_r = requests.get(url)
    yelp_soup = BeautifulSoup(yelp_r.text, 'html.parser')
    distances.append(str(yelp_soup.findAll(name = 'h2')[0]).split()[3])
    c+=1
    print(c)

distance, c = {}, 0
for i, j in airports_combinations:
    distance[i+j] = float(distances[c])
    distance[j+i] = float(distances[c])
    c+=1

distance_a2a = []
for i in route:
    
    length, dist = len(i)-1, 0
    for j in range(length):
        dist += distance[i[j] + i[j+1]]
    distance_a2a.append(dist)

distances = np.array(distance_a2a, dtype = np.float32).reshape((len_train + len_test, 1))
distances_train, distances_test = distances[:len_train], distances[len_train:]

del airports_combinations, c, dist, distance, distance_a2a, distances, i, j, length, route, url, yelp_r, yelp_soup
###############################################################################

############################Additional_Info####################################
train['Additional_Info'] = train['Additional_Info'].replace('No Info', 'No info')
test['Additional_Info'] = test['Additional_Info'].replace('No Info', 'No info')

train['Additional_Info'] = train['Additional_Info'].replace('1 Short layover', '1 Long layover')
test['Additional_Info'] = test['Additional_Info'].replace('1 Short layover', '1 Long layover')
train['Additional_Info'] = train['Additional_Info'].replace('2 Long layover', '1 Long layover')
test['Additional_Info'] = test['Additional_Info'].replace('2 Long layover', '1 Long layover')

train['Additional_Info'] = train['Additional_Info'].replace('Change airports', 'Others')
test['Additional_Info'] = test['Additional_Info'].replace('Change airports', 'Others')
train['Additional_Info'] = train['Additional_Info'].replace('Red-eye flight', 'Others')
test['Additional_Info'] = test['Additional_Info'].replace('Red-eye flight', 'Others')

additional_info = np.concatenate((train['Additional_Info'].values, test['Additional_Info'].values))

le = LabelEncoder()
additional_info = le.fit_transform(additional_info).reshape((len_train + len_test, 1))
ohe = OneHotEncoder(categorical_features = [0])
additional_info = ohe.fit_transform(additional_info).toarray()
additional_info = np.array(additional_info, dtype = np.float32)
additional_info_train, additional_info_test = additional_info[:len_train], additional_info[len_train:]

del additional_info
###############################################################################

########################SAVE DATA TO NPY FILES#################################
np.save("additional_info_test.npy", additional_info_test)
np.save("additional_info_train.npy", additional_info_train)
np.save("airline_test.npy", airline_test)
np.save("airline_train.npy", airline_train)
np.save("date_of_journey_test.npy", date_of_journey_test)
np.save("date_of_journey_train.npy", date_of_journey_train)
np.save("delay_test.npy", delay_test)
np.save("delay_train.npy", delay_train)
np.save("distances_test.npy", distances_test)
np.save("distances_train.npy", distances_train)
np.save("total_stops_test.npy", total_stops_test)
np.save("total_stops_train.npy", total_stops_train)
###############################################################################

#######################LOAD DATA FROM NPY FILES################################
additional_info_test = np.load("additional_info_test.npy")
additional_info_train = np.load("additional_info_train.npy")
airline_test = np.load("airline_test.npy")
airline_train = np.load("airline_train.npy")
date_of_journey_test = np.load("date_of_journey_test.npy")
date_of_journey_train = np.load("date_of_journey_train.npy")
delay_test = np.load("delay_test.npy")
delay_train = np.load("delay_train.npy")
distances_test = np.load("distances_test.npy")
distances_train = np.load("distances_train.npy")
total_stops_test = np.load("total_stops_test.npy")
total_stops_train = np.load("total_stops_train.npy")
###############################################################################

############################JOINING_DATA#######################################
X_train = np.concatenate((
                            additional_info_train, 
                            airline_train, 
                            date_of_journey_train, 
                            delay_train, 
                            distances_train, 
                            total_stops_train
                        ), axis = -1)

Y_train = np.array(train['Price'].values.reshape((10683, 1)), dtype = np.float32)

X_test = np.concatenate((
                            additional_info_test, 
                            airline_test, 
                            date_of_journey_test, 
                            delay_test, 
                            distances_test, 
                            total_stops_test
                        ), axis = -1)
###############################################################################

##########################Standard_Scaler######################################
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
###############################################################################

#########################SAVING_FINAL_DATA#####################################
np.save("X_train.npy", X_train)
np.save("Y_train.npy", Y_train)
np.save("X_test.npy", X_test)
###############################################################################

########################LOADING_FINAL_DATA#####################################
X_train = np.load("X_train.npy")
Y_train = np.load("Y_train.npy")
X_test = np.load("X_test.npy")
###############################################################################

#############################____MODEL____#####################################
v1 = 5
v2 = 800

regressor = XGBRegressor(
        max_depth = v1, 
        n_estimators = v2, 
        n_jobs = 4
        )
regressor.fit(X_train, Y_train)

Y_test = regressor.predict(X_test)
###############################################################################

############################SAVING THE OUTPUT##################################
output = {'Price': Y_test
          }
df = pd.DataFrame(output, columns = ['Price'])
df.to_excel(r'Output_'+str(v1)+'_'+str(v2)+'_all.xlsx', index = None, header = True)
###############################################################################
