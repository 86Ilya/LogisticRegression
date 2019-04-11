# -*- coding: utf-8 -*-
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.special import expit as sigmoid


class LogisticRegression:
    def __init__(self):
        self.w = None
        self.loss_history = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=200,
              batch_size=200, verbose=False):
        """
        Train this classifier using stochastic gradient descent.

        Inputs:
        - X: N x D array of training data. Each training point is a D-dimensional
             column.
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        # Add a column of ones to X for the bias sake.
        X = LogisticRegression.append_biases(X)
        num_train, dim = X.shape
        if self.w is None:
            # lazily initialize weights
            self.w = np.random.randn(dim) * 0.01

        # Run stochastic gradient descent to optimize W
        self.loss_history = []
        for it in xrange(num_iters):
            indices = np.random.choice(range(num_train), size=batch_size, replace=True)
            X_batch = X[indices]
            y_batch = y[indices]
            # evaluate loss and gradient
            loss, gradW = self.loss(X_batch, y_batch, reg)
            self.loss_history.append(loss)
            # perform parameter update
            self.w -= learning_rate * gradW[:, 0]

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return self

    def predict_proba(self, X, append_bias=False):
        """
        Use the trained weights of this linear classifier to predict probabilities for
        data points.

        Inputs:
        - X: N x D array of data. Each row is a D-dimensional point.
        - append_bias: bool. Whether to append bias before predicting or not.

        Returns:
        - y_proba: Probabilities of classes for the data in X. y_pred is a 2-dimensional
          array with a shape (N, 2), and each row is a distribution of classes [prob_class_0, prob_class_1].
        """
        if append_bias:
            X = LogisticRegression.append_biases(X)
        theta_t = self.w.transpose()
        theta_t = self.w[:, np.newaxis]
        theta_t_csr = csr_matrix(theta_t)  # <class 'scipy.sparse.csr.csr_matrix'>
        predictions = np.asarray(sigmoid(((X * theta_t_csr).todense())))  # <type 'numpy.ndarray'> Nx1
        y_proba = predictions
        return y_proba

    def predict(self, X):
        """
        Use the ```predict_proba``` method to predict labels for data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """

        y_proba = self.predict_proba(X, append_bias=True)
        y_pred = y_proba >= 0.5
        y_pred = y_pred.astype(int)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """Logistic Regression loss function
        Inputs:
        - X: N x D array of data. Data are D-dimensional rows
        - y: 1-dimensional array of length N with labels 0-1, for 2 classes
        Returns:
        a tuple of:
        - loss as single float
        - gradient with respect to weights w; an array of same shape as w
        """
        dw = np.zeros_like(self.w)  # initialize the gradient as zero
        loss = 0
        m = X_batch.shape[0]
        # N - количество опытов
        # D - количество фич
        # гипотеза равна ϴtranspose * x, где x - матрица значений наших feature размерностью NxD, а ϴ - матрица
        # весов (коэф.) Dx1 для feature
        # Предположения для логистической регрессии вычисляются сигмоидной ф-ией:
        # 1/(1 + e^(-z)), где z - это наша гипотеза.
        # Размерность predicion = Nx1 ( X * theta : NxD * Dx1 : Nx1)
        # theta_t - это наша ϴtranspose
        # X_batch <class 'scipy.sparse.csr.csr_matrix'>
        theta_t = self.w.transpose()
        theta_t = self.w[:, np.newaxis]
        theta_t_csr = csr_matrix(theta_t)  # <class 'scipy.sparse.csr.csr_matrix'>

        predictions = np.asarray(sigmoid(((X_batch * theta_t_csr).todense())))  # <type 'numpy.ndarray'> Nx1

        # Compute loss and gradient. Your code should not contain python loops.
        y_batch = y_batch[:, np.newaxis]  # <type 'numpy.ndarray'> Nx1

        # numpy.ndarray поэлементно умножаем на такой же массив, затем вычитаем такие же типы и размерности
        error = (-y_batch * (np.log(predictions))) - ((1-y_batch) * (np.log(1-predictions)))
        loss = sum(error)

        # Классическое матричное умножение DxN * Nx1 = Dx1
        dw = X_batch.transpose() * csr_matrix(predictions - y_batch)
        dw = np.asarray(dw.todense())
        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.
        # Note that the same thing must be done with gradient.
        loss = (1.0 / m) * loss

        # Add regularization to the loss and gradient.
        # Note that you have to exclude bias term in regularization.
        loss = loss + reg / (2 * m) * sum((theta_t) ** 2)

        j_0 = 1.0/m * (dw)[0]
        j_1 = 1.0/m * (dw)[1:] + (reg/m) * theta_t[1:]
        grad = np.vstack((j_0[:, np.newaxis], j_1))

        return loss, grad

    @staticmethod
    def append_biases(X):
        return sparse.hstack((X, np.ones(X.shape[0])[:, np.newaxis])).tocsr()
