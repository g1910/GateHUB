__all__ = ['setup_logger']


class Logger(object):
    def __init__(self, log_file, command=None):
        self.log_file = log_file
        if command:
            self._write(command)

    def _print(self, log):
        print(log)

    def _write(self, log):
        with open(self.log_file, 'a+') as f:
            f.write(log + '\n')

    def output_print(self, log):
        self._print(log)
        self._write(log)


def setup_logger(log_file, command=None):
    return Logger(log_file, command)
