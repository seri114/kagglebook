import numpy as np
import optuna
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# アイリスデータセットをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 目的関数の定義
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-5, 100)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    logistic_regression = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return 1 - accuracy

# 最適化の実行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 最適なハイパーパラメータを表示
print("Best parameters found:")
print(study.best_params)

# 最適なハイパーパラメータを用いたモデルでテストデータを評価
best_logreg = LogisticRegression(C=study.best_params['C'], penalty=study.best_params['penalty'], solver='liblinear')
best_logreg.fit(X_train, y_train)
y_pred = best_logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")