import numpy as np
from scipy.optimize import minimize

# class was constructed with assistance of chatGPT

class SimplifiedThreePL:
    def __init__(self, experiment):
        """Initialize the Simplified Three-Parameter Logistic model."""
        self.experiment = experiment
        self._base_rate = None
        self._logit_base_rate = None
        self._discrimination = None
        self._is_fitted = False
    
    def summary(self):
        """Return a summary of the experiment."""
        n_total = self.experiment.n_trials()
        n_correct = self.experiment.n_correct()
        n_incorrect = self.experiment.n_incorrect()
        n_conditions = self.experiment.n_conditions()
        
        return {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_conditions": n_conditions
        }
    
    def predict(self, parameters):
        """Predict the probability of a correct response given the parameters."""
        discrimination, base_rate = parameters
        difficulties = [sdt.d_prime() for sdt in self.experiment.conditions]  # Use d_prime as difficulty
        probabilities = []
        
        for difficulty in difficulties:
            logit = base_rate + discrimination * difficulty  # Logit function
            prob = 1 / (1 + np.exp(-logit))  # Sigmoid function for probability
            probabilities.append(prob)
        
        return probabilities
    
    def negative_log_likelihood(self, parameters):
        """Compute the negative log-likelihood."""
        probabilities = self.predict(parameters)
        responses = [sdt.n_correct_responses() / sdt.n_total_responses() for sdt in self.experiment.conditions]  # Proportion of correct responses
        
        # Calculate negative log likelihood
        nll = -np.sum(responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities))
        return nll
    
    def fit(self):
        """Fit the model using maximum likelihood estimation."""
        initial_guess = [0.0, 0.5]  # Starting guess for parameters (discrimination, base_rate)
        result = minimize(self.negative_log_likelihood, initial_guess, method='L-BFGS-B', bounds=[(-5, 5), (0, 1)])
        
        if result.success:
            self._discrimination, self._base_rate = result.x
            self._logit_base_rate = np.log(self._base_rate / (1 - self._base_rate))  # Convert to logit
            self._is_fitted = True
        else:
            raise ValueError("Model fitting failed")
    
    def get_discrimination(self):
        """Return the discrimination parameter."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return self._discrimination
    
    def get_base_rate(self):
        """Return the base rate parameter."""
        if not self._is_fitted:
            raise ValueError("Model not fitted yet")
        return self._base_rate
