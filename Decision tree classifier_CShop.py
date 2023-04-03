# Decision tree classifier
# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier , plot_tree
import matplotlib.pyplot as plt
# Load the iris dataset
CShop = pd.read_csv("Decision trees/computer_shop.csv", sep=";")
CShop = CShop.drop(columns=['ID'])

#split dataset in features and target variable
feature_cols =['age', 'Income', 'student', 'credit_rating']
X = CShop[feature_cols] # Features
y = CShop.buys_computer # Target variable

#encode all the categorical columns
X_encoded = pd.get_dummies(X, drop_first=True)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=(2),criterion="entropy")

# Train Decision Tree Classifer
clf = clf.fit(X_encoded, y)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf, filled=True, feature_names=X_encoded.columns, class_names=y, ax=ax)
plt.show()  

