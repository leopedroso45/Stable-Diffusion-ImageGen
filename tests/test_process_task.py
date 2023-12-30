import unittest
from unittest.mock import patch, MagicMock
import sys
sys.path.append('../')
from sevsd.process_task import check_cuda_and_clear_cache, process_task, check_os_path

class TestProcessTask(unittest.TestCase):

    @patch('sevsd.process_task.generate_image')
    def test_process_task(self, mock_generate_image):

        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_generate_image.return_value = [mock_image]

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "test_path"

        process_task(fake_tasks, fake_pipeline, fake_path, parallel_exec=True)

        mock_generate_image.assert_called_once_with(fake_tasks[0], fake_pipeline, True)
        mock_image.save.assert_called()


    @patch('sevsd.process_task.generate_image')
    def test_process_task_no_images(self, mock_generate_image):

        mock_generate_image.return_value = None

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "test_path"

        process_task(fake_tasks, fake_pipeline, fake_path, parallel_exec=True)

        mock_generate_image.assert_called_once_with(fake_tasks[0], fake_pipeline, True)

    @patch('sevsd.process_task.generate_image')
    @patch('sevsd.process_task.check_cuda_and_clear_cache')
    @patch('sevsd.process_task.print')
    def test_process_task_exception(self, mock_print, mock_check_cuda_and_clear_cache, mock_generate_image):
        mock_generate_image.side_effect = Exception("Test exception")

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "test_path"

        process_task(fake_tasks, fake_pipeline, fake_path, parallel_exec=True)

        mock_print.assert_any_call("Exception: Test exception")
        mock_check_cuda_and_clear_cache.assert_called()

    @patch('sevsd.process_task.check_os_path')
    @patch('sevsd.process_task.generate_image')
    def test_process_task_with_new_directory(self, mock_generate_image, mock_check_os_path):
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_generate_image.return_value = [mock_image]
        mock_check_os_path.return_value = "test_path"

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "test_path"

        process_task(fake_tasks, fake_pipeline, fake_path, True)

        mock_check_os_path.assert_called_once_with(fake_path)
        mock_generate_image.assert_called_once_with(fake_tasks[0], fake_pipeline, True)
        mock_image.save.assert_called()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('sevsd.process_task.generate_image')
    def test_process_task_existing_directory(self, mock_generate_image, mock_exists, mock_makedirs):

        mock_exists.return_value = True
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_generate_image.return_value = [mock_image]

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "existing_path"

        process_task(fake_tasks, fake_pipeline, fake_path, True)

        mock_makedirs.assert_not_called()
        mock_generate_image.assert_called_once_with(fake_tasks[0], fake_pipeline, True)
        mock_image.save.assert_called()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('sevsd.process_task.generate_image')
    def test_process_task_no_directory(self, mock_generate_image, mock_exists, mock_makedirs):

        mock_exists.return_value = False
        mock_makedirs.return_value = True
        mock_image = MagicMock()
        mock_image.save = MagicMock()
        mock_generate_image.return_value = [mock_image]

        fake_tasks = [("prompt", None, 50, 1, 7.5)]
        fake_pipeline = MagicMock()
        fake_path = "existing_path"

        process_task(fake_tasks, fake_pipeline, fake_path, parallel_exec=True)

        mock_makedirs.assert_called_once()
        mock_generate_image.assert_called_once_with(fake_tasks[0], fake_pipeline, True)
        mock_image.save.assert_called()


class TestCheckCudaAndClearCache(unittest.TestCase):

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    def test_check_cuda_and_clear_cache_with_cuda(self, mock_empty_cache, mock_is_available):
        check_cuda_and_clear_cache()
        mock_empty_cache.assert_called_once()

    @patch('torch.cuda.is_available', return_value=False)
    @patch('gc.collect')
    def test_check_cuda_and_clear_cache_without_cuda(self, mock_gc_collect, mock_is_available):
        check_cuda_and_clear_cache()
        mock_gc_collect.assert_called_once()
        
if __name__ == '__main__':
    unittest.main()
