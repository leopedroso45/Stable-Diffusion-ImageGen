import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.setup_pipeline import setup_pipeline

class TestSetupPipeline(unittest.TestCase):

    @patch('sevsd.setup_pipeline.StableDiffusionPipeline')
    @patch('sevsd.setup_pipeline.setup_device')
    def test_setup_pipeline_from_pretrained(self, mock_setup_device, mock_StableDiffusionPipeline):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_StableDiffusionPipeline.from_pretrained.return_value = mock_pipeline
        config = ('model_path', 'cache_path')

        pipeline = setup_pipeline(config)

        mock_setup_device.assert_called_once()
        mock_StableDiffusionPipeline.from_pretrained.assert_called_once_with(
            config[0], cache_dir=config[1], use_safetensors=False, load_safety_checker=False, requires_safety_checker=False
        )

    @patch('sevsd.setup_pipeline.StableDiffusionPipeline')
    @patch('sevsd.setup_pipeline.setup_device')
    def test_setup_pipeline_from_single_file(self, mock_setup_device, mock_StableDiffusionPipeline):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_StableDiffusionPipeline.from_single_file.return_value = mock_pipeline
        config = ('model_file.safetensors', 'cache_path')

        pipeline = setup_pipeline(config)

        mock_setup_device.assert_called_once()
        mock_StableDiffusionPipeline.from_single_file.assert_called_once_with(
            config[0], cache_dir=config[1], use_safetensors=True, load_safety_checker=False, requires_safety_checker=False
        )

if __name__ == '__main__':
    unittest.main()
