import unittest
import numpy as np
from SimplifiedThreePL import SimplifiedThreePL
from Experiment import Experiment  # Assuming you have the Experiment class

class TestSimplifiedThreePL(unittest.TestCase):

    def setUp(self):
        # Create an example experiment object
        # Assuming Experiment has methods to define trials, responses, etc.
        self.experiment = Experiment()  # You need to have the Experiment class implemented
        self.model = SimplifiedThreePL(self.experiment)

    def test_initialization(self):
        # Test constructor with valid inputs
        self.assertIsInstance(self.model, SimplifiedThreePL)
        
        # Test that an error is raised if you try to access parameters before fitting
        with self.assertRaises(ValueError):
            self.model.get_discrimination()

    def test_predict(self):
        # Assume predict() gives probabilities between 0 and 1
        probabilities = self.model.predict([1.0, 0.5])  # Example parameters a=1.0, c=0.5
        self.assertTrue(np.all(probabilities >= 0) and np.all(probabilities <= 1))

    def test_parameter_estimation(self):
        # Test that the model parameters are estimated correctly after fitting
        self.model.fit()
        discrimination = self.model.get_discrimination()
        base_rate = self.model.get_base_rate()
        self.assertIsNotNone(discrimination)
        self.assertIsNotNone(base_rate)

    def test_negative_log_likelihood(self):
        initial_params = [0.5, 0.2]
        nll = self.model.negative_log_likelihood(initial_params)
        self.assertGreater(nll, 0)

    def test_multiple_fit_stability(self):
        # Run multiple fits and check if results are stable
        self.model.fit()
        discrimination_first = self.model.get_discrimination()
        self.model.fit()
        discrimination_second = self.model.get_discrimination()
        self.assertAlmostEqual(discrimination_first, discrimination_second, delta=0.01)

    # You can also write corruption tests to ensure object consistency

if __name__ == '__main__':
    unittest.main()