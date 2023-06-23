import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Data Import
df = pd.read_csv('laptop_data.csv')


# EDA
#print(df.isnull().sum())
#print(df.dtypes)

#TypeValues = set(df['TypeName'].values)
#print(TypeValues)


# Feature Transformation
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

#print(df.dtypes)


# Train Test Split
features = df.drop('Price', axis=1)
target = df['Price']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.22, random_state=22)

#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# Model Training
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score as evs

models = [SVR(), RandomForestRegressor(), GradientBoostingRegressor(), DecisionTreeRegressor(), LinearRegression()]

for m in models:
    m.fit(X_train, Y_train)
    
    train_pred = m.predict(X_train)
    print(f'Training Accuracy of {m} is : {evs(Y_train, train_pred)}')
    
    test_pred = m.predict(X_test)
    print(f'Test Accuracy of {m} is : {evs(Y_test, test_pred)}')
    print('|')
    print('|')
    print('|')