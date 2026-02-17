import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("train.csv")
df["Heart Disease"].replace({'Presence': 1, 'Absence': 0}, inplace=True)

features = df.columns.to_list()
features.remove("Heart Disease")
features.remove("id")

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

X_train = train_df[features]
y_train = train_df["Heart Disease"]
X_test = test_df[features]
y_test = test_df["Heart Disease"]

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_train.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print(f'Accuracy: {model.score(X_test_scaled, y_test)}')
print(f'Kappa: {cohen_kappa_score(model.predict(X_test_scaled), y_test)}')

feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

print("Feature Importances:")
print(feature_importance.to_string(index=False))

# this is a change