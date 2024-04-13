import unittest
from unittest.mock import patch
from claudeslens.benchmark import main as claudeslens_main


class TestClaudesLens(unittest.TestCase):
    @patch('claudeslens.benchmark.argparse.ArgumentParser.parse_args')
    @patch('claudeslens.benchmark.train')
    @patch('claudeslens.benchmark.load_MNIST', return_value=(None, None))
    def test_claudeslens_runs_mnist(self, mock_load_MNIST, mock_train, mock_parse_args):
        """
        Test that the main function can run with mocked components for MNIST.
        """
        mock_args = {
            'soda': False,
            'batch_size': 32,
            'epochs': 1,
            'models': ['pv', 'cv', 'pc', 'cc', 'cl'],
            'load_weights': False,
            'save_weights': False,
            'train': True,
            'benchmark': False,
            'log': False,
            'save_plots': False,
        }
        mock_parse_args.return_value = type('Args', (object,), mock_args)()
        claudeslens_main()

        mock_load_MNIST.assert_called()
        mock_train.assert_called()

    @patch('claudeslens.benchmark.argparse.ArgumentParser.parse_args')
    @patch('claudeslens.benchmark.train')
    @patch('claudeslens.benchmark.load_SODA', return_value=(None, None, None))
    def test_claudeslens_runs_soda(self, mock_load_SODA, mock_train, mock_parse_args):
        """
        Test that the main function can run with mocked components for SODA.
        """
        mock_args = {
            'soda': True,
            'batch_size': 32,
            'epochs': 1,
            'models': ['pv', 'cv', 'pc', 'cc', 'cl'],
            'load_weights': False,
            'save_weights': False,
            'train': True,
            'benchmark': False,
            'log': False,
            'save_plots': False,
        }
        mock_parse_args.return_value = type('Args', (object,), mock_args)()
        claudeslens_main()

        mock_load_SODA.assert_called()
        mock_train.assert_called()


if __name__ == '__main__':
    unittest.main()
