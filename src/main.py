from src.data import load_dataset
from src.model import Model


USE_FIXED_SEED = False
RANDOM_STATE = 42 if USE_FIXED_SEED else None


def terminal_interface(predictor):
    print("\n" + "=" * 55)
    print("STARTUP ACQUISITION PREDICTOR")
    print("=" * 55)

    print("Pygame is not installed.")
    print("Using the terminal interface instead.")

    while True:
        print("\nEnter the total funding received by a startup.")
        print("Enter 'q' to quit.")
        funding = input("\nFunding (USD): ").strip()
        if funding.lower() == "q":
            print("\nExiting predictor...")
            break
        try:
            funding = float(funding)
            if funding < 0:
                raise ValueError
            prediction = predictor.predict(funding)[0]
            probability = predictor.predict_proba(funding)[0]
            if prediction == 1:
                print("\nPrediction:  ACQUIRED")
            else:
                print("\nPrediction: NOT ACQUIRED")
                
            print("\nPrediction probabilities:")
            print(f"Not acquired: {probability[0] * 100:.2f}%")
            print(f"Acquired:     {probability[1] * 100:.2f}%")

        except ValueError:
            print("\nPlease enter a valid non-negative funding amount.")

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
    print("\nEvaluating the model...")

    results = predictor.evaluate()
    baseline = predictor.baseline_accuracy()
    cv_results = predictor.cross_validate(folds=5)

    print(f"""
TEST SET EVALUATION
-------------------------------------------------------
Accuracy:               {results['accuracy']:.4f}
Balanced accuracy:      {results['balanced_accuracy']:.4f}
Precision (acquired):   {results['precision']:.4f}
Recall (acquired):      {results['recall']:.4f}
F1-score (acquired):    {results['f1']:.4f}

Confusion Matrix:
{results['confusion_matrix']}

Classification Report:
{results['classification_report']}

Baseline Accuracy:      {baseline:.4f}

5-FOLD CROSS-VALIDATION
-------------------------------------------------------
Fold scores: {cv_results['scores']}
Mean accuracy:          {cv_results['mean']:.4f}
Standard deviation:     {cv_results['std']:.4f}
""")

    print("\nTraining the final predictor on all available records...")
    predictor.train(full=True)
    print("Training complete.")

    try:
        from src.interactiveUI import StartupPredictionUI
        print("\nLaunching graphical interface...")
        app = StartupPredictionUI(
            predictor,
            results,
            baseline,
            cv_results,
            len(data),
        )
        app.run()

    except ModuleNotFoundError as error:
        if error.name != "pygame":
            raise
        terminal_interface(predictor)


if __name__ == "__main__":
    main()
