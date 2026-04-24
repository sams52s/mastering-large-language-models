# SETUP
import sys
import os

CUR_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.join(CUR_DIRECTORY, '..')
sys.path.insert(0, ROOT_DIRECTORY)

# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

# noinspection PyUnresolvedReferences
from task07_clf_naive_bayes.task import load_bow_train_test
# noinspection PyUnresolvedReferences
from envvars import LessonEnv
from tools_basics.helpers import get_config

class MyLogisticRegression:
    """Logistic Regression classifier."""

    def __init__(self):
        self.model = LogisticRegression()
    
    def _scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)
        return X_train, X_test


    def eval_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> LogisticRegression:
        """
        Train and evaluate the logistic regression model.
        :param X_train: Bag-of-words training features
        :param y_train: Training labels
        :param X_test: Bag-of-words test features
        :param y_test: Test labels
        """
        X_train, X_test = self._scale_features(X_train, X_test)
        self.model.fit(X_train, y_train)
        
        for name, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            proba = proba = self.model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
            plt.plot(*roc_curve(y, proba)[:2], label='%s AUC=%.4f' % (name, auc))

            acc = self.model.score(X, y)
            print(f"{name} accuracy: {acc:.3f}")
        
        plt.plot([0, 1], [0, 1], '--', color='black')
        plt.legend(fontsize='large')
        plt.grid()
        plt.show()

        return self.model

def main() -> None:
    conf = get_config(path=LessonEnv.CONF_PATH, root=LessonEnv.ROOT_DIRECTORY)
    
    X_train_bow, X_test_bow, y_train, y_test = load_bow_train_test(conf)
    logreg_model = MyLogisticRegression()
    logreg_model.eval_model(X_train_bow, y_train, X_test_bow, y_test)

if __name__ == '__main__':
    main()
