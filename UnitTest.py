import unittest
from Python_Assigment_Hirak import MatchFunctions, pd

class TestMatchFunctions(unittest.TestCase):

    def setUp(self):
        self.ff = MatchFunctions()
        self.train_fun = pd.read_csv('train_fun.csv')
        self.ideal_fun = pd.read_csv('ideal_fun.csv')
        self.test_fun = pd.read_csv('test_fun.csv')

    def test_Identify_best_fits(self):
        result = self.ff.Identify_best_fits(self.train_fun, self.ideal_fun)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (4, 2))

    def test_search_ideal_match(self):
        result = self.ff.search_ideal_match(self.test_fun)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (20, 4))

    def test_prepare_graphs(self):
        x_fun = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        y1_fun = pd.DataFrame({'y1': [1, 4, 9, 16, 25]})
        y2_fun = pd.DataFrame({'y2': [2, 4, 6, 8, 10]})
        result = self.ff.prepare_graphs(x_fun, 0, y1_fun, 0, y2_fun, 0, show_plots=False)
        self.assertIsNotNone(result)