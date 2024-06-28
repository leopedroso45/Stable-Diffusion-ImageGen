import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.setup_pipeline import setup_pipeline

class TestSetupPipeline(unittest.TestCase):

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_pretrained')
    @patch('sevsd.setup_pipeline.setup_device')
    def test_setup_pipeline_from_pretrained(self, mock_setup_device, mock_from_pretrained, mock_feature_extractor):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_feature_extractor_instance = MagicMock()
        mock_feature_extractor.return_value = mock_feature_extractor_instance
        config = 'model_path'

        pipeline = setup_pipeline(config)

        mock_setup_device.assert_called_once()
        mock_from_pretrained.assert_called_once_with(
            config, use_safetensors=False, safety_checker=None, feature_extractor=mock_feature_extractor_instance
        )
        mock_feature_extractor.assert_called_once_with(config)

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_single_file')
    @patch('sevsd.setup_pipeline.setup_device')
    def test_setup_pipeline_from_single_file(self, mock_setup_device, mock_from_single_file, mock_feature_extractor):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_single_file.return_value = mock_pipeline
        mock_feature_extractor_instance = MagicMock()
        mock_feature_extractor.return_value = mock_feature_extractor_instance
        config = 'model_file.safetensors'

        pipeline = setup_pipeline(config)

        mock_setup_device.assert_called_once()
        mock_from_single_file.assert_called_once_with(
            config, use_safetensors=True, safety_checker=None, feature_extractor=mock_feature_extractor_instance
        )
        mock_feature_extractor.assert_called_once_with(config)

if __name__ == '__main__':
    unittest.main()