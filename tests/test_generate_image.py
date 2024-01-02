import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.generate_image import generate_image

class TestGenerateImage(unittest.TestCase):

    @patch('sevsd.generate_image.torch')
    def test_generate_image(self, mock_torch):
        fake_job = {"prompt": "prompt", "negative_prompt": None}
        fake_executor = {"inference_steps": 50, "num_of_exec": 1, "cfg_scale": 7.5}
        fake_pipeline = MagicMock()
        mock_output = {'images': [MagicMock()]}
        fake_pipeline.return_value = mock_output

        images = generate_image(fake_job, fake_pipeline, fake_executor, parallel_exec=True)

        self.assertEqual(len(images), len(mock_output['images']))
        mock_torch.no_grad.assert_called()

    @patch('sevsd.generate_image.torch')
    def test_generate_image_sequential_execution(self, mock_torch):
        fake_job = {"prompt": "prompt", "negative_prompt": None}
        fake_executor = {"inference_steps": 50, "num_of_exec": 10, "cfg_scale": 7.5}  # num_images = 10 para testar a execução sequencial
        fake_pipeline = MagicMock()
        fake_image = MagicMock()
        mock_output = {'images': [fake_image] * 10}
        fake_pipeline.return_value = mock_output

        images = generate_image(fake_job, fake_pipeline, fake_executor, parallel_exec=False)

        self.assertEqual(len(images), 10)
        self.assertEqual(images[0], fake_image)
        mock_torch.no_grad.assert_called()

    @patch('sevsd.generate_image.torch')
    def test_generate_image_runtime_error(self, mock_torch):
        fake_job = {"prompt": "prompt", "negative_prompt": None}
        fake_executor = {"inference_steps": 50, "num_of_exec": 1, "cfg_scale": 7.5}
        fake_pipeline = MagicMock()

        fake_pipeline.side_effect = RuntimeError("Test error")

        images = generate_image(fake_job, fake_pipeline, fake_executor, parallel_exec=False)

        self.assertIsNone(images)
        mock_torch.no_grad.assert_called()

if __name__ == '__main__':
    unittest.main()
