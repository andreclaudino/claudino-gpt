import torch
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_claudino_gpt_model(model, save_path, model_name="claudino_gpt"):
    """
    Save the ClaudinoGPT model to disk.
    
    Args:
        model: The ClaudinoGPT model instance to save
        save_path: Directory path where the model should be saved
        model_name: Name to use for saving the model file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the save directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Define the full path for the model file
        model_file_path = os.path.join(save_path, f"{model_name}.pth")
        
        # Save the model state dict
        torch.save(model.state_dict(), model_file_path)
        
        print(f"Model successfully saved to {model_file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def save_claudino_gpt_full_model(model, save_path, model_name="claudino_gpt"):
    """
    Save the complete ClaudinoGPT model including architecture.
    
    Args:
        model: The ClaudinoGPT model instance to save
        save_path: Directory path where the model should be saved
        model_name: Name to use for saving the model file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the save directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Define the full path for the complete model file
        model_file_path = os.path.join(save_path, f"{model_name}_complete.pth")
        
        # Save the entire model (architecture + state dict)
        torch.save(model, model_file_path)
        
        print(f"Complete model successfully saved to {model_file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving complete model: {e}")
        return False

def save_claudino_gpt_with_config(model, config, save_path, model_name="claudino_gpt"):
    """
    Save the ClaudinoGPT model along with its configuration.
    
    Args:
        model: The ClaudinoGPT model instance to save
        config: Configuration dictionary for the model
        save_path: Directory path where the model should be saved
        model_name: Name to use for saving the model file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the save directory exists
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        model_file_path = os.path.join(save_path, f"{model_name}_state_dict.pth")
        torch.save(model.state_dict(), model_file_path)
        
        # Save configuration
        config_file_path = os.path.join(save_path, f"{model_name}_config.json")
        import json
        with open(config_file_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model and config successfully saved to {model_file_path} and {config_file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving model with config: {e}")
        return False
    
def cleanup_old_models(
    source_folder: Union[str, Path], 
    total_models_to_keep: int
) -> bool:
    """
    Keep only the most recent files in the specified folder, deleting older ones.
    
    Args:
        source_folder: Path to the folder containing models to manage
        total_models_to_keep: Number of most recent files to keep (default: 5)
    
    Returns:
        bool: True if successful, False otherwise
    
    Raises:
        ValueError: If total_models_to_keep is negative
    """
    if total_models_to_keep < 0:
        raise ValueError("total_models_to_keep must be non-negative")
    
    try:
        # Convert to Path object for easier handling
        source_path = Path(source_folder)
        
        # Check if folder exists
        if not source_path.exists():
            print(f"Folder {source_folder} does not exist")
            return False
            
        if not source_path.is_dir():
            print(f"{source_folder} is not a directory")
            return False
        
        # Get all files in the folder
        files = [f for f in source_path.iterdir() if f.is_file()]
        
        # Sort files by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the specified number of most recent files
        files_to_delete = files[total_models_to_keep:]
        
        # Delete older files
        for file in files_to_delete:
            file.unlink()
            print(f"Deleted old model file: {file}")
            
        print(f"Successfully maintained {total_models_to_keep} most recent models")
        return True
        
    except Exception as e:
        print(f"Error cleaning up old models: {e}")
        return False
    

def load_claudino_gpt_model(
    model_path: Union[str, Path], 
    model_class: Optional[Any] = None,
    device: str = 'cpu'
) -> Optional[torch.nn.Module]:
    """
    Load a ClaudinoGPT model from disk using its state dictionary.
    
    Args:
        model_path: Full path to the model file
        model_class: The ClaudinoGPT model class to recreate the model structure (optional)
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        torch.nn.Module: Loaded model instance if successful, None otherwise
    """
    try:
        model_file_path = Path(model_path)
        
        # Check if model file exists
        if not model_file_path.exists():
            print(f"Model file {model_path} does not exist")
            return None
            
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # If model_class is provided, recreate the model and load state dict
        if model_class is not None:
            model = model_class()
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"Model successfully loaded from {model_path}")
            return model
        else:
            # Try to infer model class from the file name or directory structure
            # This assumes you have access to the model definition in your codebase
            print("Warning: model_class not provided. Please provide the model class to properly load the model.")
            return None
            
    except Exception as e:
        print(f"Error loading model from state dict: {e}")
        return None

def load_claudino_gpt_full_model(
    model_path: Union[str, Path], 
    device: str = 'cpu'
) -> Optional[torch.nn.Module]:
    """
    Load a complete ClaudinoGPT model including its architecture from disk.
    
    Args:
        model_path: Full path to the complete model file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        torch.nn.Module: Loaded model instance if successful, None otherwise
    """
    try:
        model_file_path = Path(model_path)
        
        # Check if file exists
        if not model_file_path.exists():
            print(f"Model file {model_path} does not exist")
            return None
            
        # Load the entire saved model
        model = torch.load(model_path, map_location=device)
        
        print(f"Complete model successfully loaded from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading complete model: {e}")
        return None

def load_claudino_gpt_model_with_config(
    model_path: Union[str, Path], 
    config_path: Optional[Union[str, Path]] = None,
    model_class: Optional[Any] = None,
    device: str = 'cpu'
) -> Optional[Dict[str, Any]]:
    """
    Load a ClaudinoGPT model from disk along with its configuration.
    
    Args:
        model_path: Full path to the model state dictionary file
        config_path: Path to the configuration file (optional)
        model_class: The ClaudinoGPT model class to recreate the model structure
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Dict containing 'model': loaded model and 'config': loaded configuration if available
    """
    try:
        # Convert to Path objects for easier handling
        model_path = Path(model_path)
        
        # Check if model file exists
        if not model_path.exists():
            print(f"Model file {model_path} does not exist")
            return None
            
        if not model_path.is_file():
            print(f"{model_path} is not a file")
            return None
        
        # Load model state dict
        model_state_dict = torch.load(model_path, map_location=device)
        
        # If config path is provided, load it
        config = {}
        if config_path:
            config_file_path = Path(config_path)
            if config_file_path.exists():
                import json
                with open(config_file_path, 'r') as f:
                    config = json.load(f)
            else:
                print(f"Config file {config_path} does not exist")
        
        # Return both loaded components
        return {
            'model': model_state_dict,
            'config': config
        }
        
    except Exception as e:
        print(f"Error loading model with config: {e}")
        return None

def load_claudino_gpt_model_from_state_dict(
    model_path: Union[str, Path], 
    model_class: Any,
    device: str = 'cpu'
) -> Optional[torch.nn.Module]:
    """
    Load a ClaudinoGPT model from disk using its state dictionary and recreate the model structure.
    
    Args:
        model_path: Full path to the model state dict file
        model_class: The ClaudinoGPT model class used for recreation
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Optional[torch.nn.Module]: Loaded model instance if successful, None otherwise
    """
    try:
        # Convert to Path object for easier handling
        model_file_path = Path(model_path)
        
        # Check if file exists
        if not model_file_path.exists():
            print(f"Model file {model_path} does not exist")
            return None
            
        if not model_file_path.is_file():
            print(f"{model_path} is not a file")
            return None
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=device)
        
        # Recreate the model instance and load the state dict
        model = model_class()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        print(f"Model successfully loaded from {model_path}")
        return model
        
    except Exception as e:
        print(f"Error loading model from state dict: {e}")
        return None
