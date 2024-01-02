import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.do_work import do_work

class TestDoWork(unittest.TestCase):

    @patch('sevsd.do_work.process_task')
    @patch('sevsd.do_work.setup_pipeline')
    def test_do_work(self, mock_setup_pipeline, mock_process_task):
        mock_pipeline = MagicMock()
        mock_setup_pipeline.return_value = mock_pipeline

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
            {"label": 1, "details": ("prompt1", None, 50, 1, 7.5)},
            {"label": 2, "details": ("prompt2", None, 30, 2, 8.0)}
        ]
        fake_path = "test_path"

        do_work(fake_models, fake_jobs, fake_path)

        mock_process_task.assert_any_call(fake_jobs[0], mock_pipeline, fake_models[0]['executor'], fake_path, True)
        mock_process_task.assert_any_call(fake_jobs[1], mock_pipeline, fake_models[0]['executor'], fake_path, True)

if __name__ == '__main__':
    unittest.main()
