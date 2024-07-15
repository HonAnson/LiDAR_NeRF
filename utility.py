import os



def listFiles(directory):
    """Return list of files names in directory"""
    try:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: You do not have permission to access the directory '{directory}'.")
        return []


if __name__ == "__main__":
    pass
