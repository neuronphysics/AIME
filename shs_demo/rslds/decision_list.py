import copy
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class RobustLogisticRegression(LogisticRegression):
    def fit(self, X, y, sample_weight=None):
        self._all_zeros = False
        self._all_ones = False
        if np.all(y == 1):
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 3.0
        elif np.all(y == 0) or y.size == 0:
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = -3.0
        else:
            super(RobustLogisticRegression, self).fit(X, y, sample_weight=sample_weight)


class DecisionList(object):
    _default_lr_params = dict(penalty="l2",
                              fit_intercept=True,
                              C=100.)

    def __init__(self, K, D, lr_params=None):
        self.K, self.D = K, D

        self.lr_params = copy.deepcopy(self._default_lr_params)
        if lr_params is not None:
            self.lr_params.update(lr_params)

        self.permutation = np.arange(K)
        self.weights = np.zeros((K-1,D))
        self.biases = np.zeros(K-1)

    def fit(self, X, y):
        K, D = self.K, self.D
        assert np.max(y) <= K-1
        assert np.min(y) >= 0
        y = y.astype(np.int)

        krem = list(range(K))
        Xrem = X.copy()
        yrem = y.copy()
        for level in range(K-1):
            if Xrem.size == 0:
                print("Level {} Trivial k: {}".format(level, krem[0]))
                self.permutation[level] = krem[0]
                self.weights[level] = 0.
                self.biases[level] = -10.
                krem.pop(0)
                continue

            lrs = []
            scores = []
            for k in krem:
                yk = (yrem == k)
                lr = RobustLogisticRegression(**self.lr_params)
                lr.fit(Xrem, yk)
                lrs.append(lr)

                if np.all(yk == 0):
                    scores.append(-np.inf)
                elif np.all(yk == 1):
                    scores.append(np.inf)
                else:
                    yjpred = lr.predict_proba(Xrem)
                    scores.append(log_loss(yk, yjpred))

            ibest = np.argmin(scores)
            kbest = krem[ibest]
            lrbest = lrs[ibest]
            print("Level {} Best k: {}".format(level, kbest))

            self.permutation[level] = kbest
            self.weights[level] = lrbest.coef_
            self.biases[level] = lrbest.intercept_

            krem.remove(kbest)

            Xrem = Xrem[yrem != kbest]
            yrem = yrem[yrem != kbest]

        assert len(krem) == 1
        self.permutation[-1] = krem[0]

