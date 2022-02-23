

# Importing the libraries
import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sc = StandardScaler()

df = pd.read_csv('dataset.csv')
X = df.iloc[ : ,0:2]
y = df.iloc[:, -1]

def output(f11,parmeter,Classifier):
    df={}
    df["parmeter"] = np.array(parmeter)
    df["mean_f11"] = np.array(np.mean(f11 , axis=0))
    df["std_f11"] = np.array(np.std(f11 , axis=0))
    print(pd.DataFrame(df))
    print("the the best f1 for " + Classifier +" is : "+str(parmeter[np.mean(f11 , axis=0).argmax()]))
    return
def ypred(classifier,X_train, y_train,X_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

def KNeighbors(X ,y ):
    f11= np.zeros((1000,20))
    for s in range(0,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = s )
        X_train = sc.fit_transform(X_train)
        X_test= sc.transform(X_test)
        for n_neighbors in range(1,21):
            classifier = KNeighborsClassifier(n_neighbors = n_neighbors , metric = 'minkowski', p = 2)#it will represent Euclidean Distance
            f11[s,n_neighbors-1]=f1_score(y_test, ypred(classifier,X_train, y_train,X_test), average=None)[1]    
    output(f11,np.arange(1,21), "KNeighbors")
    return

def LogisticR(X ,y ):
    f11= np.zeros((1000))
    for s in range(0,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = s )
        X_train = sc.fit_transform(X_train)
        X_test= sc.transform(X_test)
        model = LogisticRegression()
        f11[s]=f1_score(y_test, ypred(model,X_train, y_train,X_test), average=None)[1]
    print ("the mean in Logistic Regression of f1 score is: " + str(np.mean(f11)))
    print ("the std in Logistic Regression of f1 score is: " + str(np.std(f11)))
    return

def linearSVC(X ,y ):
    f11= np.zeros((1000))
    for s in range(0,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = s )
        X_train = sc.fit_transform(X_train)
        X_test= sc.transform(X_test)
        svc_model = SVC(kernel ="linear")
        f11[s]=f1_score(y_test, ypred(svc_model,X_train, y_train,X_test), average=None)[1]
    print( "the mean in SVC of f1 score is:"+str(np.mean(f11)))
    print( "the std in SVC of f1 score is:"+str(np.std(f11)))
    return


def PolynomSvc(X ,y ):
    f11= np.zeros((1000,4))
    for s in range(0,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = s )
        X_train = sc.fit_transform(X_train)
        X_test= sc.transform(X_test)
        for degree in range(2,6):
            svc_model = SVC(kernel ="poly" , degree = degree)
            f11[s,degree-2]= f1_score(y_test, ypred(svc_model,X_train, y_train,X_test), average=None)[1]
    output(f11,np.arange(2,6), "polynom Svc")
    return


def Gaussian_Svc(X ,y ):
    f11= np.zeros((1000,5))
    param=(0.2 , 0.5 , 1.2 ,1.8 ,3)
    for s in range(0,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = s )
        X_train = sc.fit_transform(X_train)
        X_test= sc.transform(X_test)
        for c in param:
            svc_model = SVC(kernel="rbf", C=c)
            svc_model.fit(X_train, y_train)
            f11[s,param.index(c)]= f1_score(y_test, ypred(svc_model,X_train, y_train,X_test), average=None)[1]
    output(f11,param,"Gaussian Svc" )
    return 

KNeighbors(X,y)
LogisticR(X,y)
linearSVC(X,y)
PolynomSvc(X,y)
Gaussian_Svc(X,y)
