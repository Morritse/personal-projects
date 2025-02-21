import logging
import sys

def setup_logging():
    """Configure logging with proper formatting and levels."""
    # Create formatters
    console_formatter = logging.Formatter(
        '%(message)s'  # Clean console output
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'  # Detailed file output
    )
    
    # Console handler - only show INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # File handler - show everything
    file_handler = logging.FileHandler('trading.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add our handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from components
    logging.getLogger('components').setLevel(logging.WARNING)
    
    # Websocket noise reduction
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    
    return root_logger
