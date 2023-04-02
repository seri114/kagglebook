import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# アイリスデータセットをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 損失関数の定義（1 - 精度）
def objective(params):
    C = params['C']
    penalty = params['penalty']
    
    logistic_regression = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
    logistic_regression.fit(X_train, y_train)
    y_pred = logistic_regression.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {'loss': 1 - accuracy, 'status': STATUS_OK}

# ハイパーパラメータ空間の定義
space = {
    'C': hp.loguniform('C', -5, 2),
    'penalty': hp.choice('penalty', ['l1', 'l2'])
}

# 探索の実行
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# 最適なハイパーパラメータを表示
best_params['penalty'] = ['l1', 'l2'][best_params['penalty']]
print("Best parameters found:")
print(best_params)

# 最適なハイパーパラメータを用いたモデルでテストデータを評価
best_logreg = LogisticRegression(C=best_params['C'], penalty=best_params['penalty'], solver='liblinear')
best_logreg.fit(X_train, y_train)
y_pred = best_logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
