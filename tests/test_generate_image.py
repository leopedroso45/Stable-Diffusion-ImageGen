import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.generate_image import generate_image

class TestGenerateImage(unittest.TestCase):

    @patch('sevsd.generate_image.torch')
    def test_generate_image(self, mock_torch):
        fake_args = ("prompt", None, 50, 1, 7.5)
        fake_pipeline = MagicMock()
        mock_output = {'images': [MagicMock()]}
        fake_pipeline.return_value = mock_output

        images = generate_image(fake_args, fake_pipeline, parallel_exec=True)

        self.assertEqual(len(images), len(mock_output['images']))
        mock_torch.cuda.empty_cache.assert_called()

    @patch('sevsd.generate_image.torch')
    def test_generate_image_runtime_error(self, mock_torch):
        fake_args = ("prompt", None, 50, 1, 7.5)
        fake_pipeline = MagicMock()

        fake_pipeline.side_effect = RuntimeError("Test error")

        images = generate_image(fake_args, fake_pipeline, parallel_exec=False)

        self.assertIsNone(images)
        mock_torch.cuda.empty_cache.assert_called()

if __name__ == '__main__':
    unittest.main()
