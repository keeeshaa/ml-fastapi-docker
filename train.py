import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# 1. Load the classic Iris dataset
X, y = load_iris(return_X_y=True)

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a simple classifier
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 4. Check accuracy
print("Test accuracy:", model.score(X_test, y_test))

# 5. Save the model
joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")
