import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

X = train_data.drop(columns=['Target', 'Id'])
y = train_data['Target']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

y_val_pred = rf.predict(X_val)
balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
print(f'Balanced Accuracy: {balanced_acc}')

X_test = test_data.drop(columns=['Id'])
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

test_preds = rf.predict(X_test_scaled)

submission = pd.DataFrame({'Id': test_data['Id'], 'Target': test_preds})
submission.to_csv('submission.csv', index=False, header=True)
