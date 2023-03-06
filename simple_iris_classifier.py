from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Load the Iris dataset
iris = load_iris()

# Define the features (X) and target (y)
X = iris.data
y = iris.target

# Build a logistic regression model
model = LogisticRegression()

# Evaluate the model using cross-validation
scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy across all cross-validation folds
print("Average accuracy:", scores.mean())

