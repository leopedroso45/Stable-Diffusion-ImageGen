import unittest
from unittest.mock import patch
import sys
sys.path.append('../')
import sevsd.setup_device as setup_device

class TestSetupDevice(unittest.TestCase):
    @patch('torch.cuda.is_available', return_value=True)
    def test_cuda_available(self, mock_cuda):
        device = setup_device()
        self.assertEqual(str(device), 'cuda')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=True)
    def test_mps_available(self, mock_mps, mock_cuda):
        device = setup_device()
        self.assertEqual(str(device), 'mps')

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_cpu_only(self, mock_mps, mock_cuda):
        device = setup_device()
        self.assertEqual(str(device), 'cpu')

if __name__ == '__main__':
    unittest.main()
