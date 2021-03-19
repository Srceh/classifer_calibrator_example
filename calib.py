import numpy

from sklearn import isotonic

from sklearn import linear_model


class iso:
    def __init__(self):
        self.model_list = []

    def fit(self, s, y):
        k = numpy.shape(s)[1]
        if k == 2:
            k = 1
        for i in range(0, k):
            mdl = isotonic.IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            mdl.fit(s[:, i], y[:, i])
            self.model_list.append(mdl)

    def predict_proba(self, s):
        k = numpy.shape(s)[1]
        if k == 2:
            k = 1
        s_hat = numpy.zeros_like(s)
        for i in range(0, k):
            s_hat[:, i] = self.model_list[i].predict(s[:, i])
        
        if k == 1:
            s_hat[:, 1] = 1 - s_hat[:, 0]
        return s_hat / numpy.sum(s_hat, axis=1).reshape(-1, 1)


class EMP:
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.a = None

    def fit(self, s, y):
        k = numpy.shape(s)[1]
        self.a = numpy.zeros((self.n_bins, k))
        bins = numpy.linspace(0, 1, self.n_bins + 1)
        for i in range(0, k):
            self.a[:, i] = (numpy.sum(((s[:, i] <= bins[1:].reshape(-1, 1)) & (s[:, i] >= bins[:-1].reshape(-1, 1))) * \
                                     y[:, i].reshape(1, -1), axis=1)) /  \
                           (numpy.sum(((s[:, i] <= bins[1:].reshape(-1, 1)) & (s[:, i] >= bins[:-1].reshape(-1, 1))),
                                     axis=1))

    def predict_proba(self, s):
        k = numpy.shape(s)[1]
        s_hat = numpy.zeros_like(s)
        bins = numpy.linspace(0, 1, self.n_bins+1)
        for i in range(0, k):
            tmp_s =((s[:, i] <= bins[1:].reshape(-1, 1)) & (s[:, i] >= bins[:-1].reshape(-1, 1))) * \
                   self.a[:, i].reshape(-1, 1)
            tmp_s[numpy.isnan(tmp_s)] = 0.0
            s_hat[:, i] = numpy.sum(tmp_s, axis=0)

        return s_hat / numpy.sum(s_hat, axis=1).reshape(-1, 1)
    
class MAT:
    def __init__(self):
        self.mdl = None
        
    def fit(self, s, y):
        self.mdl = linear_model.LogisticRegression()
        s[s[:, -1]<=1e-16, -1] = 1e-16
        logit_s = numpy.log(s[:, :-1] / s[:, -1].reshape(-1, 1))
        self.mdl.fit(logit_s, y)
        
    def predict_proba(self, s):
        s[s[:, -1]<=1e-16, -1] = 1e-16
        logit_s = numpy.log(s[:, :-1] / s[:, -1].reshape(-1, 1))
        return self.mdl.predict_proba(logit_s)
        
    
    
class BETA:
    def __init__(self):
        self.model_list = []
        
    def fit(self, s, y):
        k = numpy.shape(s)[1]
        if k == 2:
            k = 1
        for i in range(0, k):
            mdl = linear_model.LogisticRegression(C=1e16)
            mdl.fit(numpy.hstack([numpy.log(s[:, i]).reshape(-1, 1), numpy.log(1-s[:, i]).reshape(-1, 1)]), y[:, i].reshape(-1, 1))
            self.model_list.append(mdl)
            
    def predict_proba(self, s):
        k = numpy.shape(s)[1]
        if k == 2:
            k = 1
        s_hat = numpy.zeros_like(s)
        for i in range(0, k):
            s_hat[:, i] = self.model_list[i].predict_proba(numpy.hstack([numpy.log(s[:, i]).reshape(-1, 1), numpy.log(1-s[:, i]).reshape(-1, 1)]))[:, 1]
        
        if k == 1:
            s_hat[:, 1] = 1 - s_hat[:, 0]
        return s_hat / numpy.sum(s_hat, axis=1).reshape(-1, 1)