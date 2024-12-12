from datetime import datetime

def current_date() -> str:
    """
    Get the current date formatted as "DD Month YYYY".

    Returns:
        str: Current date in "DD Month YYYY" format.
    """
    current_date = datetime.now()
    return current_date.strftime("%d %B %Y")
