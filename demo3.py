import pandas as pd

df = pd.read_csv("water_potability.csv")

# Missing values count
print(df.isnull().sum())

# Imputation
df['ph'].fillna(df['ph'].median(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].median(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)

# Verify
print(df.isnull().sum())








from sklearn.feature_selection import mutual_info_classif

X = df.drop('Potability', axis=1)
y = df['Potability']

mi_scores = mutual_info_classif(X, y)

mi_df = pd.DataFrame({
    'Feature': X.columns,
    'MI Score': mi_scores
}).sort_values(by='MI Score', ascending=False)

print(mi_df)

top5_features = mi_df['Feature'].head(5).tolist()
print("Top 5 features:", top5_features)








from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_selected = X[top5_features]

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)







from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))











import numpy as np

n_iterations = 500
scores = []

n_size = len(X_test)

for i in range(n_iterations):
    indices = np.random.choice(range(n_size), n_size, replace=True)
    X_sample = X_test[indices]
    y_sample = y_test.iloc[indices]
    
    y_pred_sample = model.predict(X_sample)
    score = accuracy_score(y_sample, y_pred_sample)
    scores.append(score)

lower = np.percentile(scores, 2.5)
upper = np.percentile(scores, 97.5)

print("95% CI:", (lower, upper))
print("CI Width:", upper - lower)











X_full = df.drop('Potability', axis=1)

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_f = scaler.fit_transform(X_train_f)
X_test_f = scaler.transform(X_test_f)

model_full = RandomForestClassifier(n_estimators=200, random_state=42)
model_full.fit(X_train_f, y_train_f)

y_pred_f = model_full.predict(X_test_f)

print("Full Model Accuracy:", accuracy_score(y_test_f, y_pred_f))
print("Full Model F1:", f1_score(y_test_f, y_pred_f))
print("Full Model MCC:", matthews_corrcoef(y_test_f, y_pred_f))










importances = model_full.feature_importances_

feat_imp = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feat_imp)

print("Top 5 RF features:", feat_imp['Feature'].head(5).tolist())







from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model_full, X_full, y, cv=10, scoring='accuracy'
)

print("Mean CV Accuracy:", scores.mean())
print("Std Dev:", scores.std())