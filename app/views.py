from django.shortcuts import render,redirect
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
import os


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score as ras
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from .models import *
from .forms import *
from django.views import View
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.contrib.auth import logout

@login_required
def dataframe(request):
    path = Path.objects.filter(user = request.user)
    inst = path[len(path)-1]
    data = train(str(inst.path))
    # Create two dataframes with fraud and non-fraud data 
    df_fraud = data.loc[data.fraud == 1] 
    df_non_fraud = data.loc[data.fraud == 0]


    sns.countplot(x="fraud",data=data)
    plt.title("Count of Fraudulent Payments")
    plt.legend()
    plt.savefig(os.path.join("app\static\img", "dataframe.png"))    

    non_fraud = df_non_fraud.fraud.count()
    _fraud = df_fraud.fraud.count()

    # print("Number of normal examples: ",df_non_fraud.fraud.count())
    # print("Number of fradulent examples: ",df_fraud.fraud.count())
    print(data.fraud.value_counts()) # does the same thing above
    
    img = "app\static\img\dataframe.png"
    return render(request,"df.html",{"non_fraud":non_fraud,"fraud":_fraud,"img":img})

@login_required
def histogram(request):
    path = Path.objects.filter(user = request.user)
    inst = path[len(path)-1]
    data = train(str(inst.path))
    # Create two dataframes with fraud and non-fraud data 
    df_fraud = data.loc[data.fraud == 1] 
    df_non_fraud = data.loc[data.fraud == 0]
    data_col =  data.columns
    best_score = df_non_fraud.fraud.count()/ np.add(df_non_fraud.fraud.count(),df_fraud.fraud.count()) * 100
    # print("Base score we must beat is: ", 
    #     df_non_fraud.fraud.count()/ np.add(df_non_fraud.fraud.count(),df_fraud.fraud.count()) * 100)
    # print("Columns: ", data.columns)
    # Plot histograms of the amounts in fraud and non-fraud data 
    plt.hist(df_fraud.amount, alpha=0.5, label='fraud',bins=100)
    plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud',bins=100)
    plt.title("Histogram for fraud and nonfraud payments")
    plt.ylim(0,10000)
    plt.xlim(0,1000)
    plt.legend()
    plt.savefig(os.path.join("app\static\img", "histogram.png"))  
    img =  "app\static\img\histogram.png" 
    return render(request,"his.html",{"data_col":data_col,"base_score":best_score,"img":img})

def plot_roc_auc(y_test, preds,name):
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
        plt.savefig(os.path.join("app\static\img", name))     

def xy(data):
    data = train(data)
     # %% Preprocessing
    # print(data.zipcodeOri.nunique())
    # print(data.zipMerchant.nunique())

    # dropping zipcodeori and zipMerchant since they have only one unique value
    data_reduced = data.drop(['zipcodeOri','zipMerchant'],axis=1)

    data_reduced.columns

    # turning object columns type to categorical for later purposes
    col_categorical = data_reduced.select_dtypes(include= ['object']).columns
    for col in col_categorical:
        data_reduced[col] = data_reduced[col].astype('category')

    data_reduced.loc[:,['customer','merchant','category']].astype('category')
    data_dum = pd.get_dummies(data_reduced.loc[:,['customer','merchant','category','gender']],drop_first=True) # dummies
    print(data_dum.info())

    # categorical values ==> numeric values
    data_reduced[col_categorical] = data_reduced[col_categorical].apply(lambda x: x.cat.codes)

    # define X and y
    X = data_reduced.drop(['fraud'],axis=1)
    y = data['fraud']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,shuffle=True,stratify=y)
    list = [X,y,X_train,X_test,y_train,y_test]
    return list

@login_required
def Knn(request):
    path = Path.objects.filter(user = request.user)
    inst = path[len(path)-1]
    data = str(inst.path)
    lis = xy(data)
    X_train = lis[2]
    y_train = lis[4]
    X_test = lis[3]
    y_test = lis[5]
    knn = KNeighborsClassifier(n_neighbors=5,p=1)

    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)

    # High precision on fraudulent examples almost perfect score on non-fraudulent examples
    # print("Classification Report for K-Nearest Neighbours: \n", classification_report(y_test, y_pred))
    # print("Confusion Matrix of K-Nearest Neigbours: \n", confusion_matrix(y_test,y_pred))

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test,y_pred)

    plot_roc_auc(y_test, knn.predict_proba(X_test)[:,1],"knn.png")
    img = "app\static\img\knn.png"
    return render(request,"knn.html",{"report":report,"matrix":matrix,"img":img})

@login_required
def randomForest(request):
     
    rf_clf = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42,
                            verbose=1,class_weight="balanced")
    path = Path.objects.filter(user = request.user)
    inst = path[len(path)-1]
    data = str(inst.path)
   
    lis = xy(data)
    X_train = lis[2]
    y_train = lis[4]
    X_test = lis[3]
    y_test = lis[5]  

    rf_clf.fit(X_train,y_train)
    y_pred = rf_clf.predict(X_test)

    # 98 % recall on fraudulent examples but low 24 % precision.
    # print("Classification Report for Random Forest Classifier: \n", classification_report(y_test, y_pred))
    # print("Confusion Matrix of Random Forest Classifier: \n", confusion_matrix(y_test,y_pred))

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test,y_pred)
    plot_roc_auc(y_test, rf_clf.predict_proba(X_test)[:,1],"RF.png")
    img = "app\static\img\RF.png"
    return render(request,"rf.html",{"report":report,"matrix":matrix,"img":img})

@login_required
def predict(request):
    if(request.method == "POST"):
        path = Path.objects.filter(user = request.user)
        inst = path[len(path)-1]
        data = str(inst.path)
        lis = xy(data)
        X = lis[0]
        y = lis[1]
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.10, random_state=42)
        model = DecisionTreeClassifier()
        model.fit(xtrain, ytrain)
        step = request.POST.get('steps')
        cust = request.POST.get('accno')
        age = request.POST.get('age')
        gen = request.POST.get('gender')
        merc = request.POST.get('desacno')
        catg = request.POST.get('cat')
        amt = request.POST.get('amt')
        # print("Model Score")
        # print(model.score(xtest, ytest))
        model_score = model.score(xtest, ytest) * 100
        print("step,cust,age,gen,merc,catg,amt",step,cust,age,gen,merc,catg,amt)
        pred = model.predict([[step,cust,age,gen,merc,catg,amt]])
        # pred = model.predict([[0,123,1,1,321,2,1200]])
        isfraud = ""
        action = ""
        if(pred[0] == 1):
            isfraud = "Fraud"
            history = diffalgoAccu(data)
            action = "danger" 
        elif(pred[0]== 0):
            action = "success"
            isfraud = "No Fraud"
        history = diffalgoAccu(data)
        History(user = request.user,lr=history[0],rf = history[2],xgb=history[1],mdsc=model_score,status=isfraud).save()
        return render(request,"alert.html",{"isfraud":isfraud,"model_score":model_score,"action":action})
    
    if(request.method == "GET"):
        return render(request,"pred.html")
        

def diffalgoAccu(data):
    lis = xy(data)
    X = lis[0]
    y = lis[1]
    models = [LogisticRegression(), XGBClassifier(),
            #   SVC(kernel='rbf', probability=True),
            RandomForestClassifier(n_estimators=7,
                                    criterion='entropy',
                                    random_state=7)]

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

    dic = []
    for i in range(len(models)):
        models[i].fit(x_train, y_train)
        print(f'{models[i]} : ')
        
        train_preds = models[i].predict_proba(x_train)[:, 1]
        # print('Training Accuracy : ', ras(y_train, train_preds))

        y_preds = models[i].predict_proba(x_test)[:, 1]
        # print('Validation Accuracy : ', ras(y_test, y_preds))
        dic.insert(i,ras(y_test, y_preds)*100)
        # list = [(ras(y_train, train_preds))*100,(ras(y_test, y_preds))*100]
        # str = f'{models[i]}'
        # dic[str] = list
        # print()

    return dic
   
def train(dat):
    # set seaborn style because it prettier
    sns.set()
    # %% read and plot
    data = pd.read_csv(dat)
    data.head(5)
    return data

@login_required
def home(request):
    if(request.method == "POST"):
        user = request.user
        file =request.POST.get('file')
        print(file)
        dat = str(file)
        data = f'app/static/src/{dat}'
        if(dat != None ):
            Path(user = user,path= data).save()
        print(data)
        path = Path.objects.filter(user = user)
        inst = path[len(path)-1]
        # print(inst.path)
        return render(request,'home.html')
    if(request.method == "GET"):
        return render(request,'home.html')

    
@login_required
def history(request):
    data = History.objects.filter(user = request.user)
    # print(data)
    return render(request,"history.html",{"data":data})

@login_required
def clrhistory(request):
    data = History.objects.filter(user = request.user)
    for i in range(len(data)-2):
        data[i].delete()
    return redirect('hist')

@login_required
def reset(request):
    data = Path.objects.filter(user = request.user)
    for i in range(len(data)):
        data[i].delete()
    return redirect('home')

@login_required
def about(request):
    return render(request,"about.html")

@login_required
def profile(request):
    data = History.objects.filter(user = request.user)
    dat = data[len(data)-1]    
    dat1 = data[len(data)-2]    
    return render(request,"profile.html",{"user":request.user,"email":request.user.email,"data":dat,"data1":dat1})

class customerregistration(View):
    def get(self, request):
        fm = CustomerRegistrationForm()
        return render(request, 'auth/customerregistration.html', {'form': fm})

    def post(self, request):
        fm = CustomerRegistrationForm(data=request.POST)
        if fm.is_valid():
            # messages.success(request,'Congratulations !! Register Successfully')
            fm.save()
        return render(request, 'auth/customerregistration.html', {'form': fm})
        # return redirect('login')

@login_required
def custom_logout(request):
    data = Path.objects.filter(user = request.user)
    for i in range(len(data)):
        data[i].delete()
    logout(request)
    return redirect("login")