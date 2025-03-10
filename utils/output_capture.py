import sys

class CaptureOutput:
    """
    Utility class to capture standard output.
    Useful for capturing the output from agent.print_response()
    """
    def __init__(self):
        self.value = ""
        self._redirect_stdout()

    def _redirect_stdout(self):
        self.old_stdout = sys.stdout
        sys.stdout = self

    def write(self, string):
        self.value += string

    def flush(self):
        pass

    def reset(self):
        self.value = ""

    def restore_stdout(self):
        sys.stdout = self.old_stdout