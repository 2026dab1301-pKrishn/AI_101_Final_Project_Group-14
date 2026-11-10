import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class Model:

    def __init__(
        self,
        data,
        feature="funding_total_usd",
        target="status",
        test_size=0.20,
        random_state=None,
    ):
        self.feature = feature
        self.target = target
        self.random_state = random_state

        self.data = data[[feature, target]].copy()
        self.data[feature] = pd.to_numeric(self.data[feature], errors="coerce")
        self.data[target] = pd.to_numeric(self.data[target], errors="coerce")
        self.data = self.data.dropna()

        if (self.data[feature] < 0).any():
            raise ValueError("Funding cannot be negative.")

        if set(self.data[target].unique()) != {0, 1}:
            raise ValueError("Status must contain both 0 and 1.")

        self.X = self.data[[feature]]
        self.y = self.data[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        self.pipeline = self.build_pipeline()
        self.is_trained = False

    @staticmethod
    def build_pipeline():
        return Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])

    def train(self):
        self.pipeline.fit(self.X_train, self.y_train)
        self.is_trained = True
        return self

    def fit_full(self):
        self.pipeline.fit(self.X, self.y)
        self.is_trained = True
        return self

    def predict(self, funding):
        if not self.is_trained:
            raise RuntimeError("Train the model first.")

        return self.pipeline.predict([[funding]])

    def predict_proba(self, funding):
        if not self.is_trained:
            raise RuntimeError("Train the model first.")

        return self.pipeline.predict_proba([[funding]])

    @property
    def classes_(self):
        return self.pipeline.named_steps["model"].classes_

    def evaluate(self):
        self.train()
        predictions = self.pipeline.predict(self.X_test)

        return {
            "accuracy": accuracy_score(self.y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, predictions),
            "precision": precision_score(self.y_test, predictions, zero_division=0),
            "recall": recall_score(self.y_test, predictions, zero_division=0),
            "f1": f1_score(self.y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(self.y_test, predictions),
            "classification_report": classification_report(
                self.y_test,
                predictions,
                target_names=["Not acquired", "Acquired"],
                zero_division=0,
            ),
        }

    def baseline_accuracy(self):
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(self.X_train, self.y_train)

        return baseline.score(self.X_test, self.y_test)

    def cross_validate(self, folds=5):
        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=self.random_state,
        )

        scores = cross_val_score(
            self.build_pipeline(),
            self.X_train,
            self.y_train,
            cv=cv,
            scoring="accuracy",
        )

        return {
            "scores": scores,
            "mean": scores.mean(),
            "std": scores.std(),
        }
