"""Centralized Rich console utilities."""
from typing import Optional
from rich.console import Console


def get_console() -> Console:
    """Get the global Rich console instance.
    
    Returns:
        Console instance (global if available, otherwise creates new one)
    """
    try:
        return __builtins__.rich_console
    except (AttributeError, NameError):
        return Console()


def get_verbose() -> bool:
    """Get the global verbose mode setting.
    
    Returns:
        True if verbose mode is enabled, False otherwise
    """
    try:
        return __builtins__.verbose_mode
    except (AttributeError, NameError):
        return False



