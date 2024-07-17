import os
import sys
import time


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

def clearLinePrintStuff(message):
    """
    Deletes the previous line in the terminal and prints a new custom message.
    Parameters:
    message (str): The new message to print.
    """
    # Move the cursor up one line and clear the line
    sys.stdout.write('\033[F\033[K')
    sys.stdout.flush()
    
    # Print the new message
    sys.stdout.write(message + '\n')
    sys.stdout.flush()


if __name__ == "__main__":
    pass
