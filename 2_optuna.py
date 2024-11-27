import os
import optuna
import mlflow
import pandas as pd
import mlflow.sklearn
import optuna.visualization as vis
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from mlflow.models.signature import infer_signature

data = pd.read_csv('medical_insurance.csv')
encoder = OneHotEncoder(drop='first')
encoder.fit(data[['sex', 'region', 'smoker']])
data[encoder.get_feature_names_out()] = encoder.transform(data[['sex', 'region', 'smoker']]).toarray()
data.drop(['sex', 'region', 'smoker'], axis=1, inplace=True)

scaler = StandardScaler()
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlflow.set_experiment("Optuna Medical Insurance Experiment")

def objective(trial):
    
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    
    run_name = f"Iteration {trial.number + 1}"
    with mlflow.start_run(nested=True, run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
    
    return r2

study = optuna.create_study(direction='maximize') 

with mlflow.start_run(run_name="optuna_medical_insurance"):
    study.optimize(objective, n_trials=10) 
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_r2", study.best_value)

    best_params = study.best_params
    final_model = RandomForestRegressor(
        **best_params,
        random_state=42
    )
    final_model.fit(X_train, y_train)
    y_pred_final = final_model.predict(X_test)

    final_r2 = r2_score(y_test, y_pred_final)
    final_mae = mean_absolute_error(y_test, y_pred_final)
    final_rmse = mean_squared_error(y_test, y_pred_final)**0.5
    
    mlflow.log_metric("final_r2", final_r2)
    mlflow.log_metric("final_mae", final_mae)
    mlflow.log_metric("final_rmse", final_rmse)
    
    input_example = pd.DataFrame(X_train[:1], columns=data.drop('charges', axis=1).columns)
    signature = infer_signature(X_train, final_model.predict(X_train))
    
    mlflow.sklearn.log_model(
        sk_model=final_model,
        artifact_path="model_medical_insurance",
        input_example=input_example,
        signature=signature
    )
    
    opt_history_path = "optimization_history.png"
    opt_slice_path = "slice_plot.png"
    
    vis.plot_optimization_history(study).write_image(opt_history_path)
    vis.plot_slice(study).write_image(opt_slice_path)
    
    mlflow.log_artifact(opt_history_path)
    mlflow.log_artifact(opt_slice_path)

    os.system(f"rm *.png")

print("Best params:")
print(study.best_params)
