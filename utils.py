from datetime import datetime

def print_log(message: str, **kwargs) -> None:
    """
    Print a log message with a timestamp.
    
    Args:
        message (str): The message to log.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")[:-3]  # Format timestamp to milliseconds
    print(f"[{timestamp}] {message}", **kwargs)