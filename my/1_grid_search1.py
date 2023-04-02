import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# アイリスデータセットをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ハイパーパラメータのリスト
C_list = [0.001, 0.01, 0.1, 1, 10, 100]
penalty_list = ['l1', 'l2']

# グリッドサーチで探索するハイパーパラメータの組み合わせを生成
grid_search_params = []
for p1 in C_list:
    for p2 in penalty_list:
        grid_search_params.append((p1, p2))

# KFold交差検証を設定
cv = KFold(n_splits=5)

# グリッドサーチの結果を格納するための辞書
grid_search_results = {}

# グリッドサーチの実行
for params in grid_search_params:
    C, penalty = params
    logistic_regression = LogisticRegression(C=C, penalty=penalty, solver='liblinear')
    scores = []

    for train_idx, val_idx in cv.split(X_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]

        logistic_regression.fit(X_train_cv, y_train_cv)
        y_pred = logistic_regression.predict(X_val_cv)
        accuracy = accuracy_score(y_val_cv, y_pred)
        scores.append(accuracy)

    mean_score = np.mean(scores)
    grid_search_results[params] = mean_score

# 最適なハイパーパラメータを表示
best_params = max(grid_search_results, key=grid_search_results.get)
print("Best parameters found:")
print(f"C: {best_params[0]}, penalty: {best_params[1]}")

# 最適なハイパーパラメータを用いてモデルを評価
best_logreg = LogisticRegression(C=best_params[0], penalty=best_params[1], solver='liblinear')
best_logreg.fit(X_train, y_train)
y_pred = best_logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
