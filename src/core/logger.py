import logging
import sys
from pathlib import Path

from typing import Optional

def setup_logger(name: str = "ethical_simulation", log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Richtet einen zentralen Logger für die Simulation ein.
    
    Args:
        name: Name des Loggers
        log_level: Logging-Level (default: INFO)
        log_file: Optionaler Pfad zu einer Log-Datei
        
    Returns:
        Konfigurierter Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Verhindere doppelte Handler, falls setup_logger mehrfach aufgerufen wird
    if logger.handlers:
        return logger
        
    # Formatter definieren
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

# Globaler Logger für einfachen Import
logger = setup_logger()
