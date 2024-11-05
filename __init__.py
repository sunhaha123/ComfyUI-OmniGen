from typing import Dict, List, Any
import importlib.util
import os
import sys
from pathlib import Path

# Type definitions
NodeClassDict = Dict[str, Any]
NodeDisplayDict = Dict[str, str]

# Global variables
NODE_CLASS_MAPPINGS: NodeClassDict = {}
NODE_DISPLAY_NAME_MAPPINGS: NodeDisplayDict = {}
WEB_DIRECTORY = "./web"

def load_module(file_path: Path, module_name: str) -> None:
    """Load a single Python module and update mappings"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Update mappings if they exist
        if hasattr(module, "NODE_CLASS_MAPPINGS"):
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    except Exception:
        pass

def load_modules_from_directory(directory: Path) -> None:
    """Load all Python modules from specified directory"""
    if not directory.exists() or not directory.is_dir():
        return

    for file_path in directory.glob("*.py"):
        if file_path.name != "__init__.py":
            load_module(file_path, file_path.stem)

def sort_mappings() -> None:
    """Sort NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS"""
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    NODE_CLASS_MAPPINGS = dict(sorted(
        NODE_CLASS_MAPPINGS.items(),
        key=lambda x: NODE_DISPLAY_NAME_MAPPINGS.get(x[0], x[0])
    ))
    NODE_DISPLAY_NAME_MAPPINGS = dict(sorted(
        NODE_DISPLAY_NAME_MAPPINGS.items(),
        key=lambda x: x[1]
    ))

def load_javascript(web_directory: str) -> List[Dict[str, str]]:
    """Return JavaScript file configurations"""
    return [{"path": "refreshNode.js"}]

# Initialize module loading
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Load modules from current directory and py subdirectory
load_modules_from_directory(current_dir)
load_modules_from_directory(current_dir / "py")
sort_mappings()

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS", 
    "WEB_DIRECTORY",
    "load_javascript"
]

