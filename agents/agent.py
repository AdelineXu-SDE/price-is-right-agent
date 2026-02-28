import logging


class Agent:
    """
    Base class for all agents in the system.

    Provides:
    - A shared interface for agent identity (name, color)
    - Consistent, colorized logging so multi-agent traces are easy to read
    """

    # ANSI colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BG_BLACK = '\033[40m'
    RESET = '\033[0m'

    name: str = ""
    color: str = '\033[37m'

    def log(self, message):
        """Log an info message tagged with this agent's name and color."""
        color_code = self.BG_BLACK + self.color
        message = f"[{self.name}] {message}"
        logging.info(color_code + message + self.RESET)
