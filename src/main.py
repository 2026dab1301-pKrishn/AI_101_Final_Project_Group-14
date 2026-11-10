
from src.data import load_dataset
from src.model import Model
from src.interactiveUI import StartupPredictionUI


USE_FIXED_SEED = False
RANDOM_STATE = 42 if USE_FIXED_SEED else None


def main():

    print("=" * 55)
    print("STARTUP ACQUISITION PREDICTION")
    print("=" * 55)

    print("\nLoading dataset...")
    data = load_dataset()

    print(f"Records available after duplicate removal: {len(data)}")

    predictor = Model(
        data=data,
        feature="funding_total_usd",
        target="status",
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    results = predictor.evaluate()
    baseline = predictor.baseline_accuracy()
    cv_results = predictor.cross_validate(folds=5)

    print(f"""
Test accuracy:          {results['accuracy']:.4f}
Balanced accuracy:      {results['balanced_accuracy']:.4f}
Precision (acquired):   {results['precision']:.4f}
Recall (acquired):      {results['recall']:.4f}
F1-score (acquired):    {results['f1']:.4f}

Confusion matrix [rows=true, columns=predicted]:
{results['confusion_matrix']}

Classification report:
{results['classification_report']}

Baseline accuracy:      {baseline:.4f}

5-fold cross-validation:
Fold scores: {cv_results['scores']}
Mean accuracy: {cv_results['mean']:.4f}
Standard deviation: {cv_results['std']:.4f}
""")
    predictor.train(full=True)

    app = StartupPredictionUI(
        predictor,
        results,
        baseline,
        cv_results,
        len(data),
    )

    app.run()


if __name__ == "__main__":
    main()
