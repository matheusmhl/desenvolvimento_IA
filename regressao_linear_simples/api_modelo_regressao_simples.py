from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar umas instância do Fast API
app = FastAPI()

# Criar uma clase que terá os dados de request body para a API
class request_body(BaseModel):
    horas_estudo: float

# Carregar Modelo para realizar a predição
modelo_pontuacao = joblib.load('./modelo_regressao_simples.pkl')

@app.post('/predict')

def predict(data: request_body):
    # Preparar os dados para predição
    input_feature = [[data.horas_estudo]]

    # Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)
    return {'pontuacao_teste' : y_pred.tolist()}
