# Startup Acquisition Prediction
```text
Prameya Krishn 2026DAB1301
Silvery Veronica 2026MCB1572
Meesala Jyothi Prakash 2026EEB1378
Rohit Raj 2026AIB1073

Dataset: https://huggingface.co/datasets/lia-prop13/startup-Investments-analysis
```
---

A machine learning project that explores whether a startup is likely to be acquired based on its total funding.

The project uses Logistic Regression to make predictions and includes an interactive Pygame interface where users can enter a funding amount and explore the model's predictions.

---

# 1. File Structure, Execution and What to Expect

## Project Structure

*AI GENERATED TABLE*
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

### What each file does
The notebooks document the experimentation process, while the Python files inside `src/` form the executable part of the project.

- `datasets/data.csv` – The dataset used to train and evaluate the model.
- `data_preprocessing.ipynb` – Initial exploration, cleaning and preprocessing of the dataset.
- `model_training.ipynb` – Experiments with different approaches to training the model.
- `model_evaluation.ipynb` – Evaluation of model performance using different metrics.
- `visualisation_analysis.ipynb` – Visual exploration of the dataset and its characteristics.
- `data.py` – Loads the dataset and performs the required checks.
- `model.py` – Contains the `Model` class, which handles training, prediction, probability estimation and evaluation.
- `main.py` – The main entry point. It loads the data, evaluates the model and launches the interface.
- `interactiveUI.py` – Contains the Pygame-based interactive interface.


## Running the Project

**Requirements:** Python 3.12 or a compatible Python version, along with the packages listed in `requirements.txt`.

First, clone the repository:

```bash
git clone https://github.com/2026dab1301-pKrishn/AI_101_Final_Project_Group-14.git
```

Move into the project directory:

```bash
cd AI_101_Final_Project_Group-14
```

Create and activate a virtual environment:

**Mac / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the project from the root directory:

```bash
python -m src.main
```

### What happens when you run it?

The program follows this sequence:

1. Loads the dataset and removes duplicate records.
2. Prepares the selected feature and target variable.
3. Splits the data into training and testing sets.
4. Trains and evaluates the Logistic Regression model.
5. Displays evaluation metrics, including accuracy, balanced accuracy, precision, recall and F1-score.
6. Displays the confusion matrix and classification report.
7. Compares the model against a simple baseline classifier.
8. Performs 5-fold stratified cross-validation.
9. Trains the final predictor using all available records.
10. Opens the interactive Pygame interface.

Once the interface opens, the user can enter a startup's total funding amount and request a prediction.

The interface also provides the option to display the estimated probabilities for both outcomes:
- Acquired
- Not acquired

---

# 2. Coding Process, Pitfalls and Their Solutions

This project was not built in one go. A considerable part of the work involved understanding the dataset, experimenting with different approaches and figuring out why certain things were not working as expected.

The following are some of the important decisions and challenges encountered during development.

## Understanding and preprocessing the dataset

**Problem:** The dataset contains duplicate records and the funding amounts are highly skewed which messes with the model, making it suboptimal to continue working with the raw data/values

**Solution:** We removed the duplicates and applied a logarithmic transformation using `np.log1p()` before standardising the feature to push the values into the -1 to 1 range. We also coded it so that the additional inputs are also processed in a similar manner before being put through the model for predictions.

## Choosing the feature

Due to a relatively larger number of features as compared to the availible instances, we decided to keep a one dimensional input vector(`"funding_total_usd"`) to maximise the training efficiency.
This also made the project easier to explain and demonstrate through an interactive interface.

## Data splitting and evaluation

We used `train_test_split()` to divide the dataset into training and testing sets.
Stratification was used to preserve the proportion of acquired and non-acquired startups in both sets.
We also experimented with different evaluation metrics instead of relying entirely on accuracy.
The model was compared against a `DummyClassifier` baseline(predicting the most frequent entry) and evaluated using 5-fold stratified cross-validation.
This gave us a better understanding of how the model performed across different data splits.

## Keeping the code organised
Initially, much of the experimentation took place in Jupyter notebooks.
As the project developed, we started separating the actual program into smaller Python files.
The idea was to avoid having everything inside one large script.
The `Model` class was introduced so that the training and prediction logic could be reused independently of the interface.
Similarly, the dataset loading process was separated into `data.py`.
This made it easier to modify individual parts without having to rewrite the entire program.

---

# 3. AI Acknowledgement

AI tools were used during the development of this project, primarily as a learning and assistance resource.
They helped us understand unfamiliar concepts, explore alternative approaches, identify errors and improve parts of the code. They were also useful when interpreting evaluation metrics and understanding how different components of Scikit-learn work together.

However, the project was developed through our own experimentation, decisions, implementation and testing. We examined the dataset, decided which features to use, explored different preprocessing techniques, evaluated the results and gradually organised the code into its final structure over the course of many hours.

In particular, the decisions surrounding feature selection, data preprocessing, model evaluation, code organisation and the interactive interface were part of our own development process.

Ultimately, the objective was not simply to produce a working machine learning model. It was to understand the process behind building one, from examining raw data to training, evaluating and interacting with a predictive model.

This project represents that process, including the experimentation, mistakes, corrections, decisions and the ***Sweat, Blood and Tears*** that went into it.
