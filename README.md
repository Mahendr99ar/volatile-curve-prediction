# NIFTY50 Volatility Curve Prediction

This project predicts the implied volatilities of NIFTY50 index options using ensemble and deep learning models.

## 📁 Files Included

- `nifty50_volatility_prediction.py`: Main code for training, predicting, and submission generation.
- `requirements.txt`: Required Python libraries.
- `README.md`: Instructions on how to run the project.
- `sample_submission.csv`: Example output submission format.

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the full pipeline with your dataset:

```bash
python nifty50_volatility_prediction.py
```

3. Your output file `sample_submission.csv` will be generated and ready to upload.

## 📦 Notes

- Make sure `train.parquet` and `test.parquet` are placed in the same directory if you're replacing the demo data.
- TensorFlow is optional. If not installed, the deep learning model will be skipped.
