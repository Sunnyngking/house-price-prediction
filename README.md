#  House Price Prediction

A machine learning model that uses LightGBM regression and feature encoding to predict house sale prices from tabular housing data.

---

##  Overview

This project implements a house price prediction model using tabular data.  
Each feature (lot size, year built, neighborhood, etc.) contributes to the final decision using gradient boosting.  
Instead of a manual rule-based system, the model is trained using LightGBM with quantile regression to optimize prediction intervals.

---

##  Project Structure

```

house-price-prediction/
├── data/
│   ├── dataset.csv              # Training data with features + target
│   ├── test.csv                 # Test data for prediction
│   └── sample\_submission.csv    # Example format for competition submission
├── src/
│   └── main.py                  # Core script: trains model and generates submission
├── output/
│   └── submission.csv           # Model output predictions
├── README.md                    # Project overview (this file)

````

---

##  How to Run

1. Clone this repository:
```bash
git clone https://github.com/Sunnygking/house-price-prediction.git
cd house-price-prediction
````

2. Run the classifier:

```bash
python src/main.py
```

3. You’ll see:

* Model training log and feature encoding steps
* Predictions saved as a CSV
* Output formatted like `sample_submission.csv`

---

## Example Output

```
✅ All libraries successfully imported.
📥 Loading datasets...
🔍 Data loaded. Train shape: (1460, 81) | Test shape: (1459, 80)
🎯 Extracting target variable from training data...
🛠️ Encoding categorical features...
🧠 Training LightGBM quantile regression model...
📤 Saving predictions to ../output/submission.csv
✅ Done!
```

---

## Dependencies

This project only requires:

* `pandas`
* `numpy`
* `lightgbm`
* `scikit-learn`

Install with:

```bash
pip install pandas numpy lightgbm scikit-learn
```

---

## Author

**Sunnygking**
GitHub: [@Sunnygking](https://github.com/Sunnygking)

---

## License

This project is for educational and demonstration purposes. Feel free to explore, use, and improve it.
