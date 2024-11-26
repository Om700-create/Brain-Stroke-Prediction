import pickle

def save_object(obj, filename):
    """
    Save a Python object to a file using pickle.
    Args:
        obj (object): The Python object to save (model, pipeline, etc.).
        filename (str): The file path where the object will be saved.
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filename}")


def load_object(filename):
    """
    Load a Python object from a file using pickle.
    Args:
        filename (str): The file path from which to load the object.
    Returns:
        object: The loaded Python object.
    """
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filename}")
    return obj
