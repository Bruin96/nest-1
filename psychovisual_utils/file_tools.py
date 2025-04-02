import os


def get_file_list(directory, ext="mp4", recursive=False):
    """Return the list of files from a directory
    which posess the given extension"""
    extension = ("." + ext).lower()
    if recursive:
        file_list = []
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(extension):
                    file_list.append(os.path.join(root, f))
    else:
        file_list = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(extension)
        ]
    return file_list
