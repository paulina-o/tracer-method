class FileException(Exception):
    """ An exception indicating problems reading data from a file. """
    def __init__(self, message):
        self.message = message
