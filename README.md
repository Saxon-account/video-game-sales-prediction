# Video Game Sales Prediction

A machine learning model that **predicts global video game sales** from a game's attributes —
year, genre, platform, publisher, and regional sales figures.

This was my semester project for Intro to AI at The Village School, rebuilt here as a standalone
repo (originally `semester-project.ipynb` inside a coursework repo).

## Approach
- **Problem type:** Regression — predicting a continuous target (`Global_Sales`)
- **Dataset:** [Video Game Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) (`vgsales.csv`, 16,000+ games)
- **Exploration:** Correlation of release year and genre with global sales (Matplotlib)
- **Model:** `RandomForestRegressor` (scikit-learn), trained/tested with a train-test split
- **Evaluation:** Mean Absolute Error (MAE)
- **Preprocessing:** Feature selection and `StandardScaler` scaling

## Run it
```bash
pip install -r requirements.txt
jupyter notebook video-game-sales-prediction.ipynb
```
The notebook downloads the dataset via `kagglehub` (a Kaggle account/API key may be required),
or place `vgsales.csv` locally and adjust the path in the data-loading cell.

## Tech
Python · pandas · NumPy · Matplotlib · scikit-learn · Jupyter
