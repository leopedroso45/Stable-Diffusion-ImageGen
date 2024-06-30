import unittest
from unittest.mock import patch, MagicMock
import os
import torch

from sevsd.load_embeddings import load_embeddings_from_folder

class TestLoadEmbeddingsFromFolder(unittest.TestCase):

    @patch('sevsd.load_embeddings.os.listdir')
    @patch('sevsd.load_embeddings.os.path.join')
    @patch('sevsd.load_embeddings.torch.load')
    def test_load_embeddings_from_folder(self, mock_torch_load, mock_path_join, mock_listdir):
        mock_listdir.return_value = ['embedding1.pt', 'embedding2.pt', 'not_an_embedding.txt']
        mock_path_join.side_effect = lambda folder, file_name: f'{folder}/{file_name}'
        fake_embedding = MagicMock()
        mock_torch_load.return_value = fake_embedding

        folder_path = '/fake/folder'
        embeddings = load_embeddings_from_folder(folder_path)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(emb == fake_embedding for emb in embeddings))
        mock_listdir.assert_called_once_with(folder_path)
        mock_path_join.assert_any_call(folder_path, 'embedding1.pt')
        mock_path_join.assert_any_call(folder_path, 'embedding2.pt')
        mock_torch_load.assert_any_call('/fake/folder/embedding1.pt')
        mock_torch_load.assert_any_call('/fake/folder/embedding2.pt')

    @patch('sevsd.load_embeddings.os.listdir')
    def test_load_embeddings_from_empty_folder(self, mock_listdir):        
        mock_listdir.return_value = []

        folder_path = '/empty/folder'
        embeddings = load_embeddings_from_folder(folder_path)

        self.assertEqual(len(embeddings), 0)
        mock_listdir.assert_called_once_with(folder_path)

    @patch('sevsd.load_embeddings.os.listdir')
    @patch('sevsd.load_embeddings.os.path.join')
    @patch('sevsd.load_embeddings.torch.load')
    def test_load_embeddings_with_no_pt_files(self, mock_torch_load, mock_path_join, mock_listdir):
        mock_listdir.return_value = ['not_an_embedding.txt', 'another_file.doc']
        
        folder_path = '/no/pt/files'
        embeddings = load_embeddings_from_folder(folder_path)

        self.assertEqual(len(embeddings), 0)
        mock_listdir.assert_called_once_with(folder_path)
        mock_torch_load.assert_not_called()

if __name__ == '__main__':
    unittest.main()