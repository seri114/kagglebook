import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# アイリスデータセットをロード
iris = datasets.load_iris()
X = iris.data
y = iris.target

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# グリッドサーチで探索するハイパーパラメータの範囲を定義
param_grid = {
    'C':  [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
}

# グリッドサーチを設定
logistic_regression = LogisticRegression(solver='liblinear')
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')

# グリッドサーチの実行
grid_search.fit(X_train, y_train)

# 最適なハイパーパラメータを表示
print("Best parameters found:")
print(grid_search.best_params_)

# 最適なハイパーパラメータを用いたモデルでテストデータを評価
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
