from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report , accuracy_score,roc_auc_score,precision_recall_curve,roc_curve
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
data=pd.read_csv("C:/Users/udayi/Downloads/WA_Fn-UseC_-HR-Employee-Attrition.csv")
data['Attrition']=data['Attrition'].map({'Yes':1,'No':0})
x=data.drop(['Attrition'],axis=1)
y=data["Attrition"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
numeric_feature=x.select_dtypes(include=['int64','float64']).columns
categorial_feature=x.select_dtypes(include=['object']).columns
numeric_pipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
categorial_pipeline=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
preprocessor=ColumnTransformer([("num",numeric_pipeline,numeric_feature),("cat",categorial_pipeline,categorial_feature)])
pipeline=Pipeline([("preprocess",preprocessor),("model",XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,sub_sample=0.8,colsample_bytee=0.8,eval_metric='logloss',random_state=42))])
pipeline.fit(x_train,y_train)
y_pred=pipeline.predict(x_test)
print("accuracy score:",accuracy_score(y_test,y_pred))
print("classification report\n",classification_report(y_test,y_pred))
xgb_model=pipeline.named_steps['model']
importances=xgb_model.feature_importances_
print(importances)
joblib.dump(pipeline,"employee_attrition.pkl")
joblib.dump(x_train.columns.tolist(),"columns.pkl")
print("pipeline saved succesfully")