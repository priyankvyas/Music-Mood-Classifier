# Import required libraries
from load_data import load_gtzan
from preprocess import preprocess, remove_corrupted_files
from sklearn.model_selection import train_test_split

# Load the data into a Pandas DataFrame
dataframe = load_gtzan()

# Check and remove corrupted files
dataframe["file_path"] = dataframe["file_path"].apply(remove_corrupted_files)
dataframe = dataframe.dropna(subset=["file_path"])

# Prepare the DataFrame for training
dataframe = preprocess(dataframe)

# Ensure that there are no missing values in the DataFrame
assert(dataframe.isna().sum() == 0)

# Get the training, validation, and test DataFrames
