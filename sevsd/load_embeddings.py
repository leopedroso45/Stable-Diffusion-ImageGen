import torch
import os

def load_embeddings_from_folder(folder_path):
    r""" Load embeddings from a folder containing `.pt` files.

    Args:
        folder_path (str): The path to the folder containing the embeddings.

    Returns:
        list: A list of torch tensors containing the embeddings
    """
    embeddings = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(folder_path, file_name)
            embedding = torch.load(file_path)
            embeddings.append(embedding)
    return embeddings