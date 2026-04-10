import mlflow
import mlflow.sklearn

with mlflow.start_run():
    model = train_model(X_train, y_train)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")