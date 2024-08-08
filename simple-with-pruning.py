import argparse

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Script argument handler
parser = argparse.ArgumentParser(description="Help:",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--csv",  help="path to csv",  required=True)
args = parser.parse_args()

# Data Processing
transactions = pd.read_csv(args.csv, parse_dates=["Transaction Date"], date_format="%d/%m/%Y")
print(f"\nAll categories in data:", "\n")
print(transactions["Categories"].value_counts(), "\n")

# Prune categories
cats_of_interest = transactions["Categories"].value_counts()[transactions["Categories"].value_counts() >= 5 ].index
print(f"Categories of interest (Total {len(cats_of_interest)})")
transactions = transactions[transactions["Categories"].isin(cats_of_interest)]
print(transactions["Categories"].value_counts().head())

# Separate data and drop columns
transactions_x = transactions.drop(["Categories","Transaction Date", "Debit", "Credit", "Transaction Type"], axis=1)
transactions_y = transactions['Categories']

# Validatiion
if transactions_y.isnull().values.any():
    raise Exception("ERROR: There are NaN values in transactions_y")

# Tokenisation
transactions_x_encoded = pd.get_dummies(transactions_x, drop_first=True, dtype=int)

# Split test and train data
X_train, X_test, y_train, y_test = train_test_split(transactions_x_encoded, transactions_y, test_size=0.3)
print("Total size of data:", len(transactions_x_encoded))
print("Size of training data:", len(X_train))
print("Size of test data:", len(X_test))

# Classification
dtree = DecisionTreeClassifier(max_depth=50)
dtree.fit(X_train, y_train)
train_predictions = dtree.predict(X_train)
test_predictions = dtree.predict(X_test)
train_acc = accuracy_score(y_train, train_predictions)
test_acc = accuracy_score(y_test, test_predictions)
null_acc = y_test.value_counts().iloc[0]/sum(y_test.value_counts())

print(f"\n############\n  ACCURACY  \n############")
print('train acc', train_acc)
print('test acc', test_acc)
print('null accuracy', null_acc)

print(f"\n############\n\n IMPORTANCE \n############")
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(dtree.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances)

print(f"\n############\n")
