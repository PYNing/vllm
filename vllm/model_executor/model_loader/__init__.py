import os
from pathlib import Path
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.loader import (BaseModelLoader,
                                                     get_model_loader)
from vllm.model_executor.model_loader.utils import (
    get_architecture_class_name, get_model_architecture)
from aied_tools import extract_single_tar_with_password
from vllm.logger import init_logger
logger = init_logger(__name__)

def delete_safetensors_files(folder_path: str):
    """
    Delete all .safetensors files in the specified folder.
    
    Args:
        folder_path: Path to the target directory
        
    Returns:
        List of deleted filenames
        
    Raises:
        ValueError: If the path is invalid or not a directory
        PermissionError: If lacking filesystem permissions
        OSError: For other filesystem-related errors
    """
    target_dir = Path(folder_path)
    
    # Validate directory exists
    if not target_dir.exists():
        logger.error(f"Path does not exist: {folder_path}")
        raise ValueError(f"Path does not exist: {folder_path}")
    
    if not target_dir.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        raise ValueError(f"Path is not a directory: {folder_path}")

    deleted_files = []
    
    # Get all matching files (throws OSError for permission issues)
    safetensors_files = list(target_dir.glob('*.safetensors'))
    logger.info(f"Found {len(safetensors_files)} .safetensors files to delete")
    
    # Delete each file (throws OSError/PermissionError on failure)
    for file in safetensors_files:
        file.unlink()
        deleted_files.append(file.name)
        logger.info(f"Deleted: {file.name}")
    
    logger.info(f"Successfully deleted {len(deleted_files)} files")
    return deleted_files

def get_model(*, vllm_config: VllmConfig) -> nn.Module:
    model_path = vllm_config.model_config.model
    use_encrypt_model = os.environ["USE_ENCRYPT_MODEL"].strip().lower() in ['true', '1', 'yes']
    
    if use_encrypt_model:
        delete_safetensors_files(model_path)
        bin_path = os.path.join(model_path, 'safetensors.ifbin')
        if not os.path.exists(bin_path):
            raise ValueError(f"{bin_path} not exist")
        
        logger.info(f'Extracting: {bin_path}')
        extract_single_tar_with_password(bin_path)
        logger.info(f'Done')    

    loader = get_model_loader(vllm_config.load_config)
    model = loader.load_model(vllm_config=vllm_config)
    
    if use_encrypt_model:
        delete_safetensors_files(model_path)
         
    return model


__all__ = [
    "get_model", "get_model_loader", "BaseModelLoader",
    "get_architecture_class_name", "get_model_architecture"
]
