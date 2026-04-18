from random import random
import pandas as pd
import numpy as np 
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.feature_selection import mutual_info_classif
df=pd.read_csv(r'ML_CA_dataset\selected\q19_water_potability\water_potability.csv')
print(df.isnull().sum())
df['ph'] = df['ph'].fillna(df['ph'].mean())
df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())            
df['Trihalomethanes'] = df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean())
print(df.isnull().sum())

X=df.drop('Potability',axis=1)
y=df['Potability']
MutualInfo=mutual_info_classif(X,y)
MutualInfo_df=pd.DataFrame({'features':X.columns,'MutualInfo':MutualInfo}).sort_values(by='MutualInfo',ascending=False)
print(MutualInfo_df)
top5_features=MutualInfo_df['features'].head(5).tolist()
print("Top 5 features:",top5_features)

X_selected=X[top5_features]
X_train,X_test,y_train,y_test=train_test_split(X_selected,y,test_size=0.2,stratify=y,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

model=RandomForestClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("Accuracy=",accuracy_score(y_test,y_pred))
print("Precision=",precision_score(y_test,y_pred))
print("Recall=",recall_score(y_test,y_pred))
print("F1=",f1_score(y_test,y_pred))
print("MCC=",matthews_corrcoef(y_test,y_pred))

X_full=df.drop('Potability',axis=1)
X_train_f,X_test_f,y_train_f,y_test_f=train_test_split(X_full,y,test_size=0.2,stratify=y,random_state=42)
X_train_f=scaler.fit_transform(X_train_f)
X_test_f=scaler.transform(X_test_f)
model_full=RandomForestClassifier(n_estimators=200,random_state=42)
model_full.fit(X_train_f,y_train_f)
y_pred_f=model_full.predict(X_test_f)
print("Full_Accuracy:",accuracy_score(y_test_f,y_pred_f))
print("Full_F1:",f1_score(y_test_f,y_pred_f))
print("Full_MCC:",matthews_corrcoef(y_test_f,y_pred_f))

importance=model_full.feature_importances_
feat_imp=pd.DataFrame({'Feature':X_full.columns,'Importance':importance}).sort_values(by='Importance',ascending=False)
print(feat_imp)
print("Top 5 RF features:", feat_imp['Feature'].head(5).tolist())

scores=cross_val_score(model_full,X_full,y,cv=10,scoring='accuracy')
print("Mean Cross-Validation Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())

n_iterations=500
scores=[]
n_size=len(X_test)
for i in range(n_iterations):
    index=np.random.choice(range(n_size),n_size,replace=True)
    X_sample=X_test[index]
    y_sample=y_test.iloc[index]
    y_pred_sample=model.predict(X_sample)
    score=accuracy_score(y_sample,y_pred_sample)
    scores.append(score)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
print("95% CI:", (lower, upper))
print("CI Width:", upper - lower)   


"""Theory Questions"""

"""1.Does the 95% bootstrap CI for accuracy include or exclude the 0.50 baseline?what does this conform about the model being better than random guessing?
=====>The 95% confidence interval is from about 0.55 to 0.64, so it does not include 0.50. This means the model is better than random guessing. So we can say that model is learning something useful from the data but it is not very strong.


2.Which chemical feature has both the highest ML score and the highest Random Forest importance? Is this consistent with known chemistry of potable water?
====>Hardness has the highest MI, but Sulfate has the highest importance in Random Forest, so they are different. This means different methods choose different important features, as water quality depends on many factors.

3.Write 3-4 sentences for AquaSafes field teams and partner communities explaining : Which chemical properties most strongly distinguish safe water from unsafe water ; whether the analysis suggest that certain chemical levels are reliable indicators of a water supply being unsafe; and what the collected water data reveles about the charactyeristics that most commonly apperar in non-potable samples.

======>Some chemical levels help us to tell if water is safe or not. Sulfate, pH, and hardness seem most important. If these levels are too high or too low, the water is usually unsafe. Unsafe samples also show similar patterns, so they are easier to identify. """