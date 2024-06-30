import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.setup_pipeline import setup_pipeline

class TestSetupPipeline(unittest.TestCase):

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_pretrained')
    @patch('sevsd.setup_pipeline.setup_device')
    @patch('sevsd.setup_pipeline.EulerAncestralDiscreteScheduler.from_config')
    def test_setup_pipeline_from_pretrained(self, mock_scheduler_from_config, mock_setup_device, mock_from_pretrained, mock_feature_extractor):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_scheduler = MagicMock()
        mock_scheduler_from_config.return_value = mock_scheduler
        mock_feature_extractor_instance = MagicMock()
        mock_feature_extractor.return_value = mock_feature_extractor_instance
        config = 'model_path'
        loras = []

        pipeline = setup_pipeline(config, loras)

        mock_setup_device.assert_called_once()
        mock_from_pretrained.assert_called_once_with(
            config, use_safetensors=False, safety_checker=None, feature_extractor=mock_feature_extractor_instance
        )
        mock_feature_extractor.assert_called_once_with(config)
        mock_scheduler_from_config.assert_called_once()

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_single_file')
    @patch('sevsd.setup_pipeline.setup_device')
    @patch('sevsd.setup_pipeline.EulerAncestralDiscreteScheduler.from_config')
    def test_setup_pipeline_from_single_file(self, mock_scheduler_from_config, mock_setup_device, mock_from_single_file, mock_feature_extractor):
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_single_file.return_value = mock_pipeline
        mock_scheduler = MagicMock()
        mock_scheduler_from_config.return_value = mock_scheduler
        config = 'model_file.safetensors'
        loras = []

        pipeline = setup_pipeline(config, loras)

        mock_setup_device.assert_called_once()
        mock_from_single_file.assert_called_once_with(
            config, use_safetensors=True, safety_checker=None
        )
        mock_scheduler_from_config.assert_called_once()

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_pretrained')
    @patch('sevsd.setup_pipeline.setup_device')
    @patch('sevsd.setup_pipeline.EulerAncestralDiscreteScheduler.from_config')
    def test_setup_pipeline_with_loras(self, mock_scheduler_from_config, mock_setup_device, mock_from_pretrained, mock_feature_extractor):
        mock_scheduler_instance = MagicMock()
        mock_scheduler_from_config.return_value = mock_scheduler_instance
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_pipeline.scheduler = MagicMock()
        mock_pipeline.scheduler.config = MagicMock()
        mock_feature_extractor_instance = MagicMock()
        mock_feature_extractor.return_value = mock_feature_extractor_instance
        config = 'model_path'
        loras = ['./loras/lora1.safetensors', './loras/lora2.safetensors']

        pipeline = setup_pipeline(config, loras)

        mock_setup_device.assert_called_once()
        mock_from_pretrained.assert_called_once_with(
            config, use_safetensors=False, safety_checker=None, feature_extractor=mock_feature_extractor_instance
        )
        mock_feature_extractor.assert_called_once_with(config)
        mock_scheduler_from_config.assert_called_once()

        # Check if LoRA weights were loaded and fused correctly
        self.assertEqual(mock_pipeline.load_lora_weights.call_count, 2)
        mock_pipeline.load_lora_weights.assert_any_call('./loras/lora1.safetensors', weight_name='./loras/lora1.safetensors', adapter_name='lora1safetensors')
        mock_pipeline.load_lora_weights.assert_any_call('./loras/lora2.safetensors', weight_name='./loras/lora2.safetensors', adapter_name='lora2safetensors')
        mock_pipeline.set_adapters.assert_called_once_with(['lora1safetensors', 'lora2safetensors'], [1.0, 1.0])
        mock_pipeline.fuse_lora.assert_called_once()

    @patch('sevsd.setup_pipeline.AutoFeatureExtractor.from_pretrained')
    @patch('sevsd.setup_pipeline.StableDiffusionPipeline.from_pretrained')
    @patch('sevsd.setup_pipeline.setup_device')
    @patch('sevsd.setup_pipeline.EulerAncestralDiscreteScheduler.from_config')
    def test_setup_pipeline_with_embeddings(self, mock_scheduler_from_config, mock_setup_device, mock_from_pretrained, mock_feature_extractor):
        mock_scheduler_instance = MagicMock()
        mock_scheduler_from_config.return_value = mock_scheduler_instance
        mock_device = MagicMock()
        mock_setup_device.return_value = mock_device
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline
        mock_pipeline.scheduler = MagicMock()
        mock_pipeline.scheduler.config = MagicMock()
        mock_feature_extractor_instance = MagicMock()
        mock_feature_extractor.return_value = mock_feature_extractor_instance
        config = 'model_path'
        loras = []
        positive_embeddings = [MagicMock()]
        negative_embeddings = [MagicMock()]

        pipeline = setup_pipeline(config, loras, positive_embeddings=positive_embeddings, negative_embeddings=negative_embeddings)

        mock_setup_device.assert_called_once()
        mock_from_pretrained.assert_called_once_with(
            config, use_safetensors=False, safety_checker=None, feature_extractor=mock_feature_extractor_instance
        )
        mock_feature_extractor.assert_called_once_with(config)
        mock_scheduler_from_config.assert_called_once()

        self.assertEqual(pipeline.positive_embeddings, positive_embeddings)
        self.assertEqual(pipeline.negative_embeddings, negative_embeddings)

if __name__ == '__main__':
    unittest.main()