# Import the required libraries
from sklearn.preprocessing import LabelEncoder

# Reassign labels from genres to moods
genre_to_mood = {
    "blues": "sad",
    "classical": "calm",
    "country": "sad",
    "disco": "happy",
    "hiphop": "energetic",
    "jazz": "calm",
    "metal": "energetic",
    "pop": "happy",
    "reggae": "calm",
    "rock": "energetic"
}

encoder = LabelEncoder()

# Add a column for the mood label and return the dataset
def add_mood_labels(dataframe, mapping = genre_to_mood):
    dataframe['mood'] = dataframe['label'].map(mapping)
    
    # Handle mappings that are not present in the provided or default mapping dictionary
    dataframe['mood'] = dataframe['mood'].fillna('neutral')
    return dataframe

# Encode categorical variables to numerical labels
def encode_categorical(dataframe):
    dataframe['mood_encoded'] = encoder.fit_transform(dataframe['mood'])
    return dataframe

# Perform the preprocessing of the DataFrame in a single function
def preprocess(dataframe):
    dataframe = add_mood_labels(dataframe)
    dataframe = encode_categorical(dataframe)
    return dataframe