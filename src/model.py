import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import  StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


class Model:

    def __init__(self, data, feature, target, test_size=0.20, random_state=None):

        self.feature = feature
        self.target = target
        data = data[[self.feature, self.target]].dropna()
        self.X = data[[self.feature]]
        self.y = data[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y,
        )

        self.pipeline = Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        self.is_trained = False
        self.random_state = random_state

    def train(self, full=False):
        if full:
            self.pipeline.fit(self.X, self.y)
        else:
            self.pipeline.fit(self.X_train, self.y_train)
        self.is_trained = True

    def predict(self, funding):
        return self.pipeline.predict([[funding]])

    def predict_proba(self, funding):
        return self.pipeline.predict_proba([[funding]])

    def evaluate(self):
        self.train()
        predictions = self.pipeline.predict(self.X_test)

        return {
            "accuracy": accuracy_score(self.y_test, predictions),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, predictions),
            "precision": precision_score(self.y_test, predictions),
            "recall": recall_score(self.y_test, predictions),
            "f1": f1_score(self.y_test, predictions),
            "confusion_matrix": confusion_matrix(self.y_test, predictions),
            "classification_report": classification_report(
                self.y_test,
                predictions,
                target_names=["Not acquired", "Acquired"],
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

        pipeline = Pipeline([
            ("log", FunctionTransformer(np.log1p)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])

        scores = cross_val_score(
            pipeline,
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
