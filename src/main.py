# Import required libraries
from load_data import load_gtzan
from preprocess import preprocess

# Load the data into a Pandas DataFrame
dataframe = load_gtzan()

# Prepare the DataFrame for training
dataframe = preprocess(dataframe)

# Ensure that there are no missing values in the DataFrame
assert(dataframe.isna().sum() == 0)
