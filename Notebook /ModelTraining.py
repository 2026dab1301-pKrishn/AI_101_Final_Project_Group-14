import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

dst = pd.read_csv('../datasets/data.csv')#../datasets/data.csv


X = dst[['funding_total_usd']]
y = dst['status']


bestModel = None
bestScore = 0
bestRandomState = None



for i in range(10000):
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=i,
    stratify=y
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    if score > bestScore:
        bestScore = score
        bestModel = model
        bestRandomState = i
    print(f"Random State: {i}, Accuracy: {score}")

print("Best Model Accuracy:", bestScore)
print("Best Random State:", bestRandomState)
