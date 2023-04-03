# Decision tree classifier
# Import libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier , plot_tree
import matplotlib.pyplot as plt
# Load the iris dataset
iris = load_iris()
# Define the decision tree classifier model
clf = DecisionTreeClassifier(max_depth = 2)

# Train the model on the iris dataset
clf.fit(iris.data, iris.target)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(12, 12))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, ax=ax)
plt.show()