# Proceso #1
# Esto es lo que se usaría en FastAPI(no cambia nada respecto lo que se tenía)

import joblib
import pandas as pd

model = joblib.load('pipeline_model/model.pkl')

data = pd.DataFrame([[23,'male',34.4,0,'no','southwest'],[35,'female',40,1,'no','northwest']], columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

pred = model.predict(data)
print(pred)

# Proceso #2
# Usar MLFlow para publicar el endpoint

# Levantar el endpoint
# mlflow models serve -m /home/mendez/mlflow_test/MLOps-MLFlow/mlruns/535902209982421411/ff6fed6f64bf4362bce105df581645c5/artifacts/pipeline_medical_insurance -p 2222 --no-conda
# Hacer petición
# curl -X POST -H "Content-Type: application/json" -d '{"inputs": [{"age": 29, "bmi": 27.9, "children": 0, "sex": "female", "region": "southwest", "smoker": "no"}]}' http://127.0.0.1:2222/invocations
