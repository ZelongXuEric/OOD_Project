import os
import json
import pickle
import numpy as np
import torch
import random
from typing import Any, Dict, List, Optional

def save_results(data: Any, filename: str, output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    print(f"Saving results to: {filepath}")
    try:
        if filename.endswith(".json"):
            def default_serializer(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist()
                if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                    return obj.item()
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, default=default_serializer, ensure_ascii=False)
        elif filename.endswith(".pkl"):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif filename.endswith(".npy"):
            np.save(filepath, data)
        elif filename.endswith(".pt"):
            torch.save(data, filepath)
        else:
            print(f"Warning: Unknown file extension '{filename.split('.')[-1]}'. Attempting to save with pickle.")
            filepath = os.path.join(output_dir, filename + ".pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        print("Save successful.")
    except Exception as e:
        print(f"Error saving results to {filepath}: {e}")

def load_results(filename: str, input_dir: str = "results") -> Any:
    filepath = os.path.join(input_dir, filename)
    print(f"Loading results from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: file not found {filepath}")
        return None
        
    try:
        if filename.endswith(".json"):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif filename.endswith(".pkl"):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        elif filename.endswith(".npy"):
            data = np.load(filepath, allow_pickle=True)
        elif filename.endswith(".pt"):
            data = torch.load(filepath)
        else:
            print(f"Warning: Unknown file extension '{filename.split('.')[-1]}'. Attempting to load with pickle.")
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e_pkl:
                print(f"Failed to load with pickle: {e_pkl}")
                return None
        print("Load successful.")
        return data
    except Exception as e:
        print(f"Error loading results from {filepath}: {e}")
        return None

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")

if __name__ == '__main__':
    print("\nTesting utility functions")
    # 1. Test saving and loading JSON
    test_data_json = {"a": 1, "b": [2, 3], "c": np.array([4, 5])}
    save_results(test_data_json, "test.json", output_dir="temp_test_results")
    loaded_data_json = load_results("test.json", input_dir="temp_test_results")
    print(f"Loaded JSON data: {loaded_data_json}")
    assert loaded_data_json['a'] == 1 and loaded_data_json['c'] == [4, 5]

    # 2. Test saving and loading Pickle
    test_data_pkl = {"a": torch.tensor([1.0, 2.0]), "b": {"c": 3}}
    save_results(test_data_pkl, "test.pkl", output_dir="temp_test_results")
    loaded_data_pkl = load_results("test.pkl", input_dir="temp_test_results")
    print(f"Loaded Pickle data: {loaded_data_pkl}")
    assert torch.equal(loaded_data_pkl['a'], torch.tensor([1.0, 2.0]))

    # Clean up temporary directory
    import shutil
    if os.path.exists("temp_test_results"):
        shutil.rmtree("temp_test_results")
        print("Temporary test directory cleaned up.")
