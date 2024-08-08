import argparse

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer
import numpy as np


# Script argument handler
parser = argparse.ArgumentParser(description="Help:",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--csv",  help="path to csv",  required=True)
args = parser.parse_args()

# Data processing
transactions = pd.read_csv(args.csv, parse_dates=["Transaction Date"], date_format="%d/%m/%Y")
print(f"\nAll categories in data:", "\n")
print(transactions["Categories"].value_counts(), "\n")

# Separate data and drop columns
transactions_x = transactions.drop(["Categories","Transaction Date", "Debit", "Credit", "Transaction Type"], axis=1)
transactions_y = transactions['Categories']

# Validation
if transactions_y.isnull().values.any():
    raise Exception("ERROR: There are NaN values in transactions_y")

#### Tokenisation

# this
transactions_x_encoded = pd.get_dummies(transactions_x, dtype=int)
print(transactions_x_encoded.head())

# # is the same as
# vectorizer = CountVectorizer(analyzer="word", max_features=100)
# # Tokenise the Narration column
# narration_matrix = vectorizer.fit_transform(transactions_x["Narration"])
# narration_vect_df = pd.DataFrame(narration_matrix.todense(), columns=vectorizer.get_feature_names_out())
# # Tokenise the Transaction Type column
# trans_type_matrix = vectorizer.fit_transform(transactions_x["Transaction Type"])
# trans_type_vect_df = pd.DataFrame(trans_type_matrix.todense(), columns=vectorizer.get_feature_names_out())
# # Add the tokenised columns
# transactions_x_encoded = pd.concat([transactions_x, narration_vect_df, trans_type_vect_df], axis=1)
# # Remove the old columns
# transactions_x_encoded = transactions_x_encoded.drop(["Narration", "Transaction Type"], axis=1)
# print(transactions_x_encoded.head())
####

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
