import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree


file_path_full = r"C:\Users\dhrub\Downloads\prodigy ds dataset\TASK3\bank\bank-full.csv"
file_path = r"C:\Users\dhrub\Downloads\prodigy ds dataset\TASK3\bank\bank.csv"

data_full = pd.read_csv(file_path_full, delimiter=';')
data = pd.read_csv(file_path, delimiter=';')


print(data.head())


print(data.describe())


missing_values = data.isnull().sum()
print(missing_values)


le = LabelEncoder()
data_encoded = data.copy()
for col in data_encoded.columns:
    if data_encoded[col].dtype == 'object':
        data_encoded[col] = le.fit_transform(data_encoded[col])


plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=data)
plt.title('Distribution of Target Variable (y)')
plt.show()


plt.figure(figsize=(10, 8))
sns.heatmap(data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


X = data_encoded.drop('y', axis=1)
y = data_encoded['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
plt.show()
