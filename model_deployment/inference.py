import mlflow.sklearn

model = mlflow.sklearn.load_model("models:/my_model/Production")
preds = model.predict(X_new)