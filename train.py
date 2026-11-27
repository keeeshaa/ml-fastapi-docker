import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Hyperparameters
C = 1.0
MAX_ITER = 200
TEST_SIZE = 0.2

# MLflow setup: store runs in a local SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-classification")

# 1. Load the classic Iris dataset
X, y = load_iris(return_X_y=True)

# 2. Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

with mlflow.start_run():
    # 3. Train a simple classifier
    model = LogisticRegression(C=C, max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print("Accuracy:", accuracy)
    print("F1-score:", f1)

    # 5. Log parameters and metrics to MLflow
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", MAX_ITER)
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # 6. Save the model file for the API and log it as an artifact
    joblib.dump(model, "model.joblib")
    mlflow.log_artifact("model.joblib")

    # 7. Log the model and register it in the Model Registry
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="iris-classifier",
    )

print("Model saved to model.joblib and registered as 'iris-classifier'")
