import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier



# set seaborn style because it prettier
sns.set()
# %% read and plot
data = pd.read_csv("app\static\src\original.csv")

data.head(5)

# Create two dataframes with fraud and non-fraud data 
df_fraud = data.loc[data.fraud == 1] 
df_non_fraud = data.loc[data.fraud == 0]


sns.countplot(x="fraud",data=data)
plt.title("Count of Fraudulent Payments")
plt.legend()
plt.show()
print("Number of normal examples: ",df_non_fraud.fraud.count())
print("Number of fradulent examples: ",df_fraud.fraud.count())
print(data.fraud.value_counts()) # does the same thing above

# print("Mean feature values per category",data.groupby('category')['amount','fraud'].mean())

print("Columns: ", data.columns)



# Plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)
plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)
plt.title("Histogram for fraud and nonfraud payments")
plt.ylim(0,10000)
plt.xlim(0,1000)
plt.legend()
plt.show()

# %% Preprocessing
print(data.zipcodeOri.nunique())
print(data.zipMerchant.nunique())

# dropping zipcodeori and zipMerchant since they have only one unique value
data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)

data_reduced.columns

# turning object columns type to categorical for later purposes
col_categorical = data_reduced.select_dtypes(include= ['object']).columns
for col in col_categorical:
    data_reduced[col] = data_reduced[col].astype('category')

# it's usually better to turn the categorical values (customer, merchant, and category variables  )
# into dummies because they have no relation in size(i.e. 5>4) but since they are too many (over 500k) the features will grow too many and 
# it will take forever to train but here is the code below for turning categorical features into dummies
data_reduced.loc[:,['customer','merchant','category']].astype('category')
data_dum = pd.get_dummies(data_reduced.loc[:,['customer','merchant','category','gender']],drop_first=True) # dummies
print(data_dum.info())

# categorical values ==> numeric values
data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

# define X and y
X = data_reduced.drop(['fraud'],axis=1)
y = data['fraud']


# I won't do cross validation since we have a lot of instances
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True,stratify=y)

# %% Function for plotting ROC_AUC curve

def plot_roc_auc(y_test, preds):
    '''
    Takes actual and predicted(probabilities) as input and plots the Receiver
    Operating Characteristic (ROC) curve
    '''
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

# The base score should be better than predicting always non-fraduelent
print("Base score we must beat is: ", 
      df_non_fraud.fraud.count()/ np.add(df_non_fraud.fraud.count(),df_fraud.fraud.count()) * 100)


# %% K-ello Neigbors

knn = KNeighborsClassifier(n_neighbors=5,p=1)

knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

# High precision on fraudulent examples almost perfect score on non-fraudulent examples
print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, knn.predict_proba(X_test)[:,1])

# %% Random Forest Classifier

rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,
                                verbose=1,class_weight="balanced")

rf_clf.fit(X_train,y_train)
y_pred = rf_clf.predict(X_test)

# 98 % recall on fraudulent examples but low 24 % precision.
print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))
plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:,1])

from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print("Model Score")
print(model.score(xtest, ytest))

# array(["'es_transportation'", "'es_health'", "'es_otherservices'",
    #    "'es_food'", "'es_hotelservices'", "'es_barsandrestaurants'",
    #    "'es_tech'", "'es_sportsandtoys'", "'es_wellnessandbeauty'",
    #    "'es_hyper'", "'es_fashion'", "'es_home'", "'es_contents'",
    #    "'es_travel'", "'es_leisure'"], dtype=object)

# array([12,  4,  9,  3,  6,  0, 11, 10, 14,  7,  2,  5,  1, 13,  8],
#       dtype=int8)

# max feature values per category                          
# category                  amount  fraud  age                                
# 'es_barsandrestaurants'   695.63      1  '6'
# 'es_fashion'              773.61      1  '6'
# 'es_health'              1972.81      1  'U'
# 'es_home'                1540.23      1  'U'
# 'es_hotelservices'       1429.04      1  '6'
# 'es_hyper'                488.02      1  'U'
# 'es_leisure'              592.03      1  'U'
# 'es_otherservices'        964.30      1  'U'
# 'es_sportsandtoys'       1258.33      1  '6'
# 'es_tech'                1305.35      1  '6'
# 'es_travel'              8329.96      1  '6'
# 'es_wellnessandbeauty'    750.51      1  'U'

# min feature values per category  
# category                 amount  fraud  age                                   
# 'es_barsandrestaurants'    2.37      1  '0'
# 'es_fashion'               4.35      1  '0'
# 'es_health'                0.03      1  '0'
# 'es_home'                  0.19      1  '0'
# 'es_hotelservices'         0.41      1  '0'
# 'es_hyper'                 1.13      1  '1'
# 'es_leisure'              45.46      1  '0'
# 'es_otherservices'         7.66      1  '1'
# 'es_sportsandtoys'         0.42      1  '0'
# 'es_tech'                  9.71      1  '1'
# 'es_travel'                7.40      1  '0'
# 'es_wellnessandbeauty'     0.04      1  '0'

# min feature values per category                          
# category                 amount  fraud  age                         
# 'es_barsandrestaurants'    0.01      0
# 'es_contents'              0.01      0
# 'es_fashion'               0.01      0
# 'es_food'                  0.00      0
# 'es_health'                0.07      0
# 'es_home'                  0.13      0
# 'es_hotelservices'         0.02      0
# 'es_hyper'                 0.02      0
# 'es_leisure'              38.74      0
# 'es_otherservices'         0.06      0
# 'es_sportsandtoys'         0.11      0
# 'es_tech'                  0.09      0
# 'es_transportation'        0.00      0
# 'es_travel'                0.47      0
# 'es_wellnessandbeauty'     0.02      0


# max feature values per category                           
#category                   amount  fraud  age                                   
# 'es_barsandrestaurants'   166.81      0  'U'
# 'es_contents'             185.13      0  'U'
# 'es_fashion'              269.39      0  'U'
# 'es_food'                 154.91      0  'U'
# 'es_health'               468.81      0  'U'
# 'es_home'                 523.11      0  'U'
# 'es_hotelservices'        345.87      0  'U'
# 'es_hyper'                168.94      0  'U'
# 'es_leisure'              120.92      0  '6'
# 'es_otherservices'        298.81      0  'U'
# 'es_sportsandtoys'        374.46      0  'U'
# 'es_tech'                 454.83      0  'U'
# 'es_transportation'       118.07      0  'U'
# 'es_travel'              2144.86      0  '6'
# 'es_wellnessandbeauty'    260.12      0  'U'

#  This feature represents the day from the start of simulation. It has 180 steps so simulation ran for virtually 6 months
step = input("Enter the day transaction Processed")

# This feature represents the customer id
cust = input("Enter the customer ID as a src")

# Merchant: The merchant's id
merc= input("Enter the Merchant ID as a dest")

# Categorized age
# 0: <= 18,
# 1: 19-25,
# 2: 26-35,
# 3: 36-45,
# 4: 46:55,
# 5: 56:65,
# 6: > 65
age = input("Enter the Categorized age")

# Gender for customer
# E : Enterprise,
# F: Female,
# M: Male,
# U: Unknown
gen = input("Enter Your Gender as 1:M 2:Female")

# categoried data:
# 12:es_transportation
# 4:es_health
# 9:es_otherservices
# 3:es_food
# 6:es_hotelservices
# 0:es_barsandrestaurants
# 11:es_tech
# 10:es_sportsandtoys
# 14:es_wellnessandbeauty
# 7:es_hyper
# 2:es_fashion
# 5:es_home
# 1:es_contents
# 13:es_travel
# 8:es_leisure
catg = input("Enter the Categorized the data values")

# Amount of the purchase
amt = input("Enter Amount of the purchase")

# print("pridicted value is ")
# pred = model.predict([[step,cust,age,gen,merc,catg,amt]])
# if(pred[0] == 1):
#     print("Fraud")
# elif(pred[0]== 0):
#     print("No Fraud")

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


models = [LogisticRegression(), XGBClassifier(),
        #   SVC(kernel='rbf', probability=True),
          RandomForestClassifier(n_estimators=7,
                                 criterion='entropy',
                                 random_state=7)]

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.10,random_state=42)

# for i in range(len(models)):
#     models[i].fit(x_train, y_train)
#     print(f'{models[i]} : ')

#     train_preds = models[i].predict_proba(x_train)[:, 1]
#     print('Training Accuracy : ', ras(y_train, train_preds))

#     y_preds = models[i].predict_proba(x_test)[:, 1]
#     print('Validation Accuracy : ', ras(y_test, y_preds))
#     print()

dic = {}
for i in range(len(models)):
        models[i].fit(x_train, y_train)
        print(f'{models[i]} : ')
        
        train_preds = models[i].predict_proba(x_train)[:, 1]
        print('Training Accuracy : ', ras(y_train, train_preds))

        y_preds = models[i].predict_proba(x_test)[:, 1]
        print('Validation Accuracy : ', ras(y_test, y_preds))
        list = [ras(y_train, train_preds),ras(y_test, y_preds)]
        str = f'{models[i]}'
        dic[str] = list
        print()
print(dic)

  


# %% XG-Boost
# XGBoost_CLF = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=400, 
#                                 objective="binary:hinge", booster='gbtree', 
#                                 n_jobs=-1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, 
#                                 subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, 
#                                 scale_pos_weight=1, base_score=0.5, random_state=42, verbosity=True)

# XGBoost_CLF.fit(X_train,y_train)

# y_pred = XGBoost_CLF.predict(X_test)

# # reatively high precision and recall for fraudulent class
# print("Classification Report for XGBoost: \n", classification_report(y_test, y_pred)) # Accuracy for XGBoost:  0.9963059088641371
# print("Confusion Matrix of XGBoost: \n", confusion_matrix(y_test,y_pred))
# plot_roc_auc(y_test, XGBoost_CLF.predict_proba(X_test)[:,1])

# # %% Ensemble 

# estimators = [("KNN",knn),("rf",rf_clf),("xgb",XGBoost_CLF)]
# ens = VotingClassifier(estimators=estimators, voting="soft",weights=[1,4,1])

# ens.fit(X_train,y_train)
# y_pred = ens.predict(X_test)


# # Combined Random Forest model's recall and other models' precision thus this model
# # ensures a higher recall with less false alarms (false positives)
# print("Classification Report for Ensembled Models: \n", classification_report(y_test, y_pred)) # Accuracy for XGBoost:  0.9963059088641371
# print("Confusion Matrix of Ensembled Models: \n", confusion_matrix(y_test,y_pred))
# plot_roc_auc(y_test, ens.predict_proba(X_test)[:,1])
