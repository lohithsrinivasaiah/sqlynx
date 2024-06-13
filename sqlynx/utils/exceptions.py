from __future__ import annotations


class MissingEnvVarError(Exception):
    """
    Exception raised when a required environment variable is missing.
    """

    def __init__(self, variable_name: str) -> None:
        """
        Initialize MissingEnvVarError.

        Args:
            variable_name (str): The name of the missing environment variable.
        """
        self.variable_name: str = variable_name
        message = (
            f"Environment variable `{self.variable_name}` is not set.\n"
            f"Please ensure `{self.variable_name}` is defined in your .env "
            f"file or set it in your terminal session:\n"
            f"    export {self.variable_name}=YOUR_VALUE"
        )
        super().__init__(message)


class DatabaseConnectionError(Exception):
    """
    Exception raised when the connection to the database fails.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
