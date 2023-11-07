"""
Biblioteca para treinar e salvar um modelo de regressão logística.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib

class Predict:
    def __init__(self, path: str, dataset: pd.DataFrame, x: tuple, y: str) -> None:
        self.path = path
        self.dataset = dataset
        self.x = x
        self.y = y
        self.__train_test_split()
        self.y_pred = None
        self.logistic_regression = None
        self.accuracy = None

    def save(self) -> None:
        self.__train()
        self.y_pred = joblib.dump(self.logistic_regression, self.__path())

    def load(self) -> None:
        self.logistic_regression = joblib.load(self.__path())
        self.y_pred = self.logistic_regression.predict(self.x_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

    def __train(self) -> None:
        self.logistic_regression = LogisticRegression()
        self.logistic_regression.fit(self.x_train, self.y_train)
        self.y_pred = self.logistic_regression.predict(self.x_test)
        self.accuracy = metrics.accuracy_score(self.y_test, self.y_pred)

    def __path(self) -> str:
        return f'{self.path}.pkl'

    def __x(self) -> pd.DataFrame:
        return self.dataset[self.x]
    
    def __y(self) -> pd.Series:
        return self.dataset[self.y]
    
    def __train_test_split(self) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.__x(), self.__y(), test_size=0.25, random_state=0)

