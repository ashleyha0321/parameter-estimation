import sys
import os
import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL

class MockExperiment:
    """ A mock class to simulate the Experiment class for testing purposes. """
    def __init__(self, correct_counts, incorrect_counts):
        self.correct_counts = correct_counts
        self.incorrect_counts = incorrect_counts

class TestSimplifiedThreePL(unittest.TestCase):
    def setUp(self):
        """ Sets up test cases with predefined experiment data. """
        correct_counts = [55, 60, 75, 90, 95]
        incorrect_counts = [45, 40, 25, 10, 5]
        self.experiment = MockExperiment(correct_counts, incorrect_counts)
        self.model = SimplifiedThreePL(self.experiment)
    
    def test_initialization(self):
        """ Test that constructor properly initializes attributes. """
        self.assertFalse(self.model._is_fitted)
        self.assertIsNone(self.model._discrimination)
        self.assertIsNone(self.model._logit_base_rate)

    def test_invalid_initialization(self):
        """ Test constructor with mismatched data lengths. """
        with self.assertRaises(ValueError):
            SimplifiedThreePL(MockExperiment([50], [40, 10]))

    def test_summary(self):
        """ Test that summary returns correct values. """
        summary = self.model.summary()
        self.assertEqual(summary["n_total"], 100 * 5)
        self.assertEqual(summary["n_correct"], sum(self.experiment.correct_counts))
        self.assertEqual(summary["n_incorrect"], sum(self.experiment.incorrect_counts))
        self.assertEqual(summary["n_conditions"], 5)
    
    def test_predict_bounds(self):
        """ Test that predict() outputs values between 0 and 1. """
        params = [1.0, 0.0]
        predictions = self.model.predict(params)
        self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))
    
    def test_predict_effects(self):
        """ Test effects of parameters on predictions. """
        params1 = [1.0, -1.0]  # Lower base rate (q = -1, c < 0.5)
        params2 = [1.0, 1.0]   # Higher base rate (q = 1, c > 0.5)
        pred1 = self.model.predict(params1)
        pred2 = self.model.predict(params2)
        self.assertTrue(np.all(pred2 > pred1))  # Higher base rate should increase predictions
    
    def test_negative_log_likelihood(self):
        """ Test that NLL improves after fitting. """
        initial_nll = self.model.negative_log_likelihood([1.0, 0.0])
        self.model.fit()
        fitted_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))])
        self.assertLess(fitted_nll, initial_nll)
    
    def test_fit_estimates(self):
        """ Test that fitting produces reasonable estimates. """
        self.model.fit()
        self.assertTrue(0 < self.model.get_discrimination() < 10)  # Reasonable range
        self.assertTrue(0 < self.model.get_base_rate() < 1)  # Must be between 0 and 1
    
    def test_fit_consistency(self):
        """ Test that parameters remain stable across multiple fits. """
        self.model.fit()
        a1, c1 = self.model.get_discrimination(), self.model.get_base_rate()
        self.model.fit()
        a2, c2 = self.model.get_discrimination(), self.model.get_base_rate()
        self.assertAlmostEqual(a1, a2, places=2)
        self.assertAlmostEqual(c1, c2, places=2)
    
    def test_parameter_access_before_fit(self):
        """ Test that parameters cannot be accessed before fitting. """
        with self.assertRaises(ValueError):
            self.model.get_discrimination()
        with self.assertRaises(ValueError):
            self.model.get_base_rate()
    
    def test_integration(self):
        """ Full test case with a predefined dataset. """
        self.model.fit()
        predictions = self.model.predict([self.model.get_discrimination(), np.log(self.model.get_base_rate() / (1 - self.model.get_base_rate()))])
        observed_rates = np.array(self.experiment.correct_counts) / 100
        self.assertTrue(np.allclose(predictions, observed_rates, atol=0.1))  # Allow slight deviation
    
    def test_corruption_prevention(self):
        """ Test that users cannot modify private attributes. """
        with self.assertRaises(AttributeError):
            self.model._discrimination = 2.0  # Direct modification should be prevented

if __name__ == "__main__":
    unittest.main()

