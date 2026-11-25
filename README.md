# ML Model Deployment with FastAPI and Docker

A simple Logistic Regression model trained on the Iris dataset and served as a REST API with FastAPI, packaged in Docker.

## Project structure

```
ml-fastapi-docker/
├── main.py           # FastAPI app
├── train.py          # trains and saves the model
├── model.joblib      # saved model
├── requirements.txt  # Python dependencies
├── Dockerfile        # container instructions
└── README.md
```

## 1. Train the model

```bash
pip install -r requirements.txt
python train.py
```

This creates `model.joblib`.

## 2. Run the API locally

```bash
uvicorn main:app --reload
```

- Root: http://127.0.0.1:8000/
- Swagger docs: http://127.0.0.1:8000/docs

## 3. Build and run with Docker

```bash
docker build -t ml-fastapi .
docker run -p 8000:8000 ml-fastapi
```

Then open http://localhost:8000/docs

## Endpoints

**GET /**

```json
{"message": "ML API is running"}
```

**POST /predict**

Request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```json
{"prediction": "setosa"}
```

Example with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
