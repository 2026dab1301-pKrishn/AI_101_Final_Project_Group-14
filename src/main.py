
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

    print("\nEvaluating the model on the held-out test set...")
    results = predictor.evaluate()

    print(f"Test accuracy:          {results['accuracy']:.4f}")
    print(f"Balanced accuracy:      {results['balanced_accuracy']:.4f}")
    print(f"Precision (acquired):   {results['precision']:.4f}")
    print(f"Recall (acquired):      {results['recall']:.4f}")
    print(f"F1-score (acquired):    {results['f1']:.4f}")

    print("\nConfusion matrix [rows=true, columns=predicted]:")
    print(results["confusion_matrix"])

    print("\nClassification report:")
    print(results["classification_report"])

    baseline = predictor.baseline_accuracy()
    print(f"Baseline accuracy:      {baseline:.4f}")

    cv_results = predictor.cross_validate(folds=5)

    print("\n5-fold cross-validation:")
    print("Fold scores:", cv_results["scores"])
    print(f"Mean accuracy: {cv_results['mean']:.4f}")
    print(f"Standard deviation: {cv_results['std']:.4f}")

    print("\nTraining the final predictor on all available records...")
    predictor.train(full=True)
    print("Training complete.")

    print("\nLaunching interactive interface...")

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
