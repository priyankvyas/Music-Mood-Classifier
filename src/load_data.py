# Import the required libraries
import os
import pandas as pd

# Load the GTZAN dataset from the data directory
def load_gtzan(base_dir = "./data/gtzan", file_ext = ".wav"):
    data = []

    for label in os.listdir(base_dir):
        label_dir = base_dir + f"/{label}"
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.endswith(file_ext):
                    file_path = label_dir + f"/{file}"
                    data.append({"file_path": file_path, "label": label})

    # Put the data in a DataFrame and return it
    return pd.DataFrame(data)