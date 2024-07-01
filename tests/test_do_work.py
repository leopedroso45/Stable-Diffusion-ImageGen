import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.do_work import do_work

class TestDoWork(unittest.TestCase):

    @patch('sevsd.do_work.load_all_embeddings')
    @patch('sevsd.do_work.process_task')
    @patch('sevsd.do_work.setup_pipeline')
    def test_do_work(self, mock_setup_pipeline, mock_process_task, mock_load_all_embeddings):
        mock_pipeline = MagicMock()
        mock_setup_pipeline.return_value = mock_pipeline

        positive_mock_embedding = MagicMock()
        negative_mock_embedding = MagicMock()
        mock_load_all_embeddings.return_value = [positive_mock_embedding, negative_mock_embedding]

        fake_models = [
            {
                "name": "model_path",
                "executor": {
                    "labels": [1, 2],
                    "num_of_exec": 2,
                    "cfg_scale": 7
                }
            }
        ]
        fake_jobs = [
            {"label": 1, "prompt": "prompt1", "negative_prompt": None},
            {"label": 2, "prompt": "prompt2", "negative_prompt": None}
        ]
        fake_path = "test_path"

        do_work(fake_models, fake_jobs, fake_path)

        mock_setup_pipeline.assert_called_once_with(
            "model_path",
            [],
            positive_embeddings=[positive_mock_embedding, negative_mock_embedding],
            negative_embeddings=[positive_mock_embedding, negative_mock_embedding]
        )

        mock_process_task.assert_any_call(fake_jobs[0], mock_pipeline, fake_models[0]['executor'], fake_path, True)
        mock_process_task.assert_any_call(fake_jobs[1], mock_pipeline, fake_models[0]['executor'], fake_path, True)

    @patch('sevsd.do_work.load_all_embeddings')
    @patch('sevsd.do_work.process_task')
    @patch('sevsd.do_work.setup_pipeline')
    def test_do_work_with_embeddings(self, mock_setup_pipeline, mock_process_task, mock_load_all_embeddings):
        mock_pipeline = MagicMock()
        mock_setup_pipeline.return_value = mock_pipeline

        positive_mock_embeddings = [MagicMock(), MagicMock()]
        negative_mock_embeddings = [MagicMock(), MagicMock()]
        mock_load_all_embeddings.side_effect = [positive_mock_embeddings, negative_mock_embeddings]

        fake_models = [
            {
                "name": "model_path",
                "executor": {
                    "labels": [1, 2],
                    "num_of_exec": 2,
                    "cfg_scale": 7
                }
            }
        ]
        fake_jobs = [
            {"label": 1, "prompt": "prompt1", "negative_prompt": None},
            {"label": 2, "prompt": "prompt2", "negative_prompt": None}
        ]
        fake_path = "test_path"
        positive_folders = ["positive_folder1", "positive_folder2"]
        negative_folders = ["negative_folder1", "negative_folder2"]

        do_work(fake_models, fake_jobs, fake_path, positive_embedding_folders=positive_folders, negative_embedding_folders=negative_folders)

        mock_setup_pipeline.assert_called_once_with(
            "model_path",
            [],
            positive_embeddings=positive_mock_embeddings,
            negative_embeddings=negative_mock_embeddings
        )

        mock_process_task.assert_any_call(fake_jobs[0], mock_pipeline, fake_models[0]['executor'], fake_path, True)
        mock_process_task.assert_any_call(fake_jobs[1], mock_pipeline, fake_models[0]['executor'], fake_path, True)

if __name__ == '__main__':
    unittest.main()