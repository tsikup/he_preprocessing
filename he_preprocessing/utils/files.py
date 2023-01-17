import fnmatch
import os


def locate_file(pattern, root=os.curdir):
    """Locate all files matching supplied filename pattern
    in and below supplied root directory."""
    for path, dirs, files in os.walk(os.path.abspath(root)):
        files = os.listdir(os.path.abspath(root))
        for filename in fnmatch.filter(files, pattern):
            yield filename
