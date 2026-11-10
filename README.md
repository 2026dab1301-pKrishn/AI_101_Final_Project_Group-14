# Startup Acquisition Prediction

A machine learning project that predicts whether a startup is likely to be acquired based on its total funding amount.

The project uses Logistic Regression to classify startups into two categories:

- **1 – Acquired**
- **0 – Not acquired**

It also provides model evaluation results in the terminal and an interactive interface built using Pygame.

## Project Structure

```text
AI_101_Final_Project_Group-14/
│
├── datasets/
│   └── data.csv
│
├── Notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── visualisation_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   ├── main.py
│   └── interactiveUI.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Model Overview

The project uses **Logistic Regression** with `funding_total_usd` as its input feature.

The preprocessing pipeline consists of:

1. Log transformation using `log1p` to handle highly skewed funding values.
2. Standardisation using `StandardScaler`.
3. Logistic Regression for classification.

These steps are combined into a Scikit-learn Pipeline.

## Model Evaluation

The model is evaluated using a stratified train-test split.

The evaluation includes:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Baseline accuracy
- 5-fold stratified cross-validation

After evaluation, the final predictor is trained on the complete dataset for interactive predictions.

## Installation

**Recommended Python version:** Python 3.12

Clone the repository:

```bash
git clone https://github.com/2026dab1301-pKrishn/AI_101_Final_Project_Group-14.git
```

Navigate into the project directory:

```bash
cd AI_101_Final_Project_Group-14
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**macOS / Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Project

From the project's root directory, execute:

```bash
python -m src.main
```

The application automatically:

1. Loads and prepares the dataset.
2. Initialises the Logistic Regression model.
3. Evaluates the model on the held-out test set.
4. Performs 5-fold cross-validation.
5. Trains the final predictor on all available records.
6. Launches the Pygame interface.

Evaluation results are printed in the terminal and displayed in the interactive interface.

## Interactive Interface

The Pygame interface allows users to:

- View model evaluation metrics.
- Examine the confusion matrix and classification report.
- View cross-validation results.
- Enter a startup's total funding amount.
- Generate an acquisition prediction.
- Toggle estimated class probabilities.
- Reset the prediction input.

## Project Configuration

The random seed can be controlled in `main.py`:

```python
USE_FIXED_SEED = False
RANDOM_STATE = 42 if USE_FIXED_SEED else None
```

- Setting `USE_FIXED_SEED` to `True` makes the train-test and cross-validation splits reproducible.
- Setting it to `False` allows new random splits on subsequent runs.
