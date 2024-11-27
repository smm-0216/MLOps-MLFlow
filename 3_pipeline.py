import os
import mlflow
import pandas as pd
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlflow.models.signature import infer_signature

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('medical_insurance.csv')
X = data.drop('charges', axis=1)
y = data['charges']
categorical_features = ['sex', 'region', 'smoker']
numerical_features = [col for col in X.columns if col not in categorical_features]
best_params = {'n_estimators': 369, 'max_depth': 12, 'min_samples_split': 3, 'min_samples_leaf': 1}

mlflow.set_experiment("Optuna Medical Insurance Experiment")

with mlflow.start_run(run_name="final_model_pipeline"):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )

    model = RandomForestRegressor(
        **best_params,
        random_state=42
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X, y)

    input_example = X[:1]
    signature = infer_signature(X, pipeline.predict(X))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="pipeline_medical_insurance",
        input_example=input_example,
        signature=signature
    )

print("Pipeline saved with MLflow.")
