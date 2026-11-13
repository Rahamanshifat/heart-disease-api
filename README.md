# Heart Disease Prediction API (FastAPI + Docker + Render)

A small FastAPI service that serves predictions from a Random Forest classifier trained on the Heart Disease dataset. The focus is on containerization and deployment, not model accuracy.

## Project structure

```
app/
  main.py          # FastAPI app and endpoints
  schemas.py       # Pydantic input/output models
model/
  heart_model.joblib
  metadata.json    # model type, feature list, test accuracy
scripts/train.py   # training script (re-creates model/)
data/heart.csv     # dataset used for training
Dockerfile
docker-compose.yml
requirements.txt
```

## Endpoints

| Method | Path       | Description                                   |
|--------|------------|-----------------------------------------------|
| GET    | `/health`  | Liveness check                                |
| GET    | `/info`    | Model type, feature list, test accuracy       |
| POST   | `/predict` | Returns `heart_disease: true/false` + probability |
| GET    | `/docs`    | Swagger UI                                    |

## Run locally with Docker

```bash
docker-compose build
docker-compose up
```

Then open http://localhost:8000/docs.

## Example requests

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}'
```

Response:

```json
{"heart_disease": true, "probability": 0.605}
```

## Retrain the model

```bash
pip install -r requirements.txt
python scripts/train.py            # uses data/heart.csv
python scripts/train.py path/to/heart.csv   # or the Kaggle CSV
```

The dataset in `data/heart.csv` is the 303-row UCI Cleveland heart disease data (same 14 columns as the
[Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)).
Drop in the Kaggle `heart.csv` and re-run the script to train on it instead.

## Deploy on Render

1. Push this repo to GitHub.
2. In Render: **New → Web Service**, connect the repo.
3. Language/Environment: **Docker**. Dockerfile path `./Dockerfile`, build context `.` (repo root).
4. Create the service and wait for the build. Render injects `$PORT`, which the Dockerfile uses.
5. Test: `curl https://<your-service>.onrender.com/health`

**Live URL:** _add your Render URL here_
