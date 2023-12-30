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

        fake_configs = [
            {"model_info": ("model_path", "cache_path"), "task_ids": [1, 2]}
        ]
        fake_tasks = [
            {"task_id": 1, "details": ("prompt1", None, 50, 1, 7.5)},
            {"task_id": 2, "details": ("prompt2", None, 30, 2, 8.0)}
        ]
        fake_path = "test_path"

        do_work(fake_configs, fake_tasks, fake_path)

        mock_setup_pipeline.assert_called_once_with(('model_path', 'cache_path'))
        mock_process_task.assert_any_call(("prompt1", None, 50, 1, 7.5), mock_pipeline, fake_path, True)
        mock_process_task.assert_any_call(("prompt2", None, 30, 2, 8.0), mock_pipeline, fake_path, True)

if __name__ == '__main__':
    unittest.main()
