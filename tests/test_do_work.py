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

        fake_configs = [('model_path', 'cache_path')]
        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_path = "test_path"

        do_work(fake_configs, fake_tasks, fake_path)

        mock_setup_pipeline.assert_called_once_with(fake_configs[0])
        mock_process_task.assert_called_once_with(fake_tasks, mock_pipeline, fake_path)

if __name__ == '__main__':
    unittest.main()
