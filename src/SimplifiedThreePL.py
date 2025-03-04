import numpy as np
import scipy as stats
from scipy.optimize import minimize


# class was constructed using the help of chatGPT

class SimplifiedThreePL:
    def __init__(self, experiment):
        self.experiment = experiment  # Store the experiment data
        self.difficulty_params = np.array([2, 1, 0, -1, -2])  # Known difficulty levels
        self._discrimination = None  # Discrimination parameter (a)
        self._logit_base_rate = None  # Logit of base rate parameter (q)
        self._is_fitted = False  # Indicates if the model is fitted

    def summary(self):
        n_correct = np.sum(self.experiment.correct_counts)
        n_incorrect = np.sum(self.experiment.incorrect_counts)
        n_total = n_correct + n_incorrect
        n_conditions = len(self.difficulty_params)

        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": n_conditions
        }

    def predict(self, parameters):
        a, q = parameters
        c = 1 / (1 + np.exp(-q))  # Convert q to c using inverse logit
        theta = 0  # Fixed participant ability

        # Compute probability for each condition
        p_correct = c + (1 - c) / (1 + np.exp(-a * (theta - self.difficulty_params)))
        return p_correct

    def negative_log_likelihood(self, parameters):
        a, q = parameters
        c = 1 / (1 + np.exp(-q))  # Convert q to c using inverse logit
        theta = 0  # Fixed participant ability
        
        # Compute probability for each condition
        p_correct = c + (1 - c) / (1 + np.exp(-a * (theta - self.difficulty_params)))

        # Extract response counts from experiment data
        n_correct = np.array(self.experiment.correct_counts)
        n_incorrect = np.array(self.experiment.incorrect_counts)

        # Compute negative log-likelihood
        log_likelihood = (
            np.sum(n_correct * np.log(p_correct)) +
            np.sum(n_incorrect * np.log(1 - p_correct))
        )
        
        return -log_likelihood  # We minimize negative log-likelihood

    def fit(self):
        initial_guess = [1.0, 0.0]  # Initial values for [a, q]
        result = minimize(self.negative_log_likelihood, initial_guess, method="L-BFGS-B")

        if result.success:
            self._discrimination, self._logit_base_rate = result.x
            self._is_fitted = True
        else:
            raise RuntimeError("Optimization failed to converge")

    def get_discrimination(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self._discrimination

    def get_base_rate(self):
        if not self._is_fitted:
            raise ValueError("Model is not fitted yet.")
        return 1 / (1 + np.exp(-self._logit_base_rate))  # Convert q back to c