import numpy as np
from scipy.optimize import minimize
#from Experiment import Experiment

# class was constructed with the assistance of chatGPT

class SimplifiedThreePL:
    def __init__(self, experiment):
        # The experiment object will contain the data
        self.experiment = experiment
        self._base_rate = None  # Base rate parameter (c)
        self._logit_base_rate = None  # Logit of the base rate parameter
        self._discrimination = None  # Discrimination parameter (a)
        self._is_fitted = False  # Whether the model has been fitted

    def summary(self):
        return {
            'n_total': self.experiment.n_trials(),
            'n_correct': self.experiment.n_correct(),
            'n_incorrect': self.experiment.n_incorrect(),
            'n_conditions': self.experiment.n_conditions()
        }

    def predict(self, parameters):
        a, c = parameters
        # Calculate probability of correct response using the simplified 3PL formula
        # P(correct) = c + (1 - c) / (1 + exp(-a * (difficulty)))
        difficulties = self.experiment.difficulties()
        probabilities = c + (1 - c) / (1 + np.exp(-a * difficulties))
        return probabilities

    def negative_log_likelihood(self, parameters):
        a, c = parameters
        probabilities = self.predict(parameters)
        responses = self.experiment.responses()  # Correct or incorrect response (binary)
        log_likelihood = np.sum(responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities))
        return -log_likelihood  # Negative log likelihood

    def fit(self):
        # Use maximum likelihood estimation (MLE) to fit parameters
        initial_guess = [0.0, 0.5]  # Starting guess for parameters (a, c)
        result = minimize(self.negative_log_likelihood, initial_guess, method='L-BFGS-B', bounds=[(-5, 5), (0, 1)])
        if result.success:
            self._discrimination, self._base_rate = result.x
            self._logit_base_rate = np.log(self._base_rate / (1 - self._base_rate))  # Logit transformation
            self._is_fitted = True
        else:
            raise ValueError("Model fitting failed.")

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model not fitted yet.")
        return self._base_rate
