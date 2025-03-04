import unittest
from src.SimplifiedThreePL import SimplifiedThreePL
from src.SignalDetection import SignalDetection
from src.Experiment import Experiment

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        # Create SignalDetection objects
        sdt1 = SignalDetection(hits=50, misses=10, falseAlarms=5, correctRejections=35)
        sdt2 = SignalDetection(hits=60, misses=20, falseAlarms=10, correctRejections=40)
        
        # Create an Experiment object and add conditions
        self.experiment = Experiment()
        self.experiment.add_condition(sdt1, label="Condition 1")
        self.experiment.add_condition(sdt2, label="Condition 2")
        
        # Initialize the SimplifiedThreePL model
        self.model = SimplifiedThreePL(self.experiment)

    def test_predict(self):
        # Example parameters for discrimination (a) and base rate (c)
        parameters = [1.0, 0.5]
        probabilities = self.model.predict(parameters)
        
        # Assert that the probabilities are calculated correctly
        self.assertEqual(len(probabilities), 2)  # We have two conditions
        self.assertTrue(0 <= probabilities[0] <= 1)  # Probabilities should be between 0 and 1
        self.assertTrue(0 <= probabilities[1] <= 1)
    
    def test_negative_log_likelihood(self):
        initial_params = [1.0, 0.5]  # Initial guess for the parameters
        nll = self.model.negative_log_likelihood(initial_params)
        
        # Assert that the negative log-likelihood is a valid number
        self.assertIsInstance(nll, float)
        self.assertGreater(nll, 0)  # NLL should be positive
    
    def test_fit(self):
        self.model.fit()
        
        # Assert that the model has been fitted
        self.assertTrue(self.model._is_fitted)
        self.assertIsNotNone(self.model.get_discrimination())
        self.assertIsNotNone(self.model.get_base_rate())
    
    def test_summary(self):
        summary = self.model.summary()
        
        # Assert that the summary contains the correct values
        self.assertIn("n_total", summary)
        self.assertIn("n_correct", summary)
        self.assertIn("n_incorrect", summary)
        self.assertIn("n_conditions", summary)
        self.assertEqual(summary["n_conditions"], 2)  # We added two conditions
    
    def test_multiple_fit_stability(self):
        self.model.fit()
        initial_discrimination = self.model.get_discrimination()
        initial_base_rate = self.model.get_base_rate()
        
        # Fit the model again and check if the parameters are stable
        self.model.fit()
        self.assertEqual(initial_discrimination, self.model.get_discrimination())
        self.assertEqual(initial_base_rate, self.model.get_base_rate())

if __name__ == '__main__':
    unittest.main()
