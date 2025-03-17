# Import the required libraries
from sklearn.preprocessing import LabelEncoder, StandardScaler
from feature_extraction import *
import pandas as pd

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

# Initialize the encoder and the scaler
encoder = LabelEncoder()
scaler = StandardScaler()


# Check each file and remove the files that are corrupted
def remove_corrupted_files(file_path):
    try:
        _ = librosa.load(file_path, duration=30)
        return file_path
    except Exception as e:
        print(f"Skipping corrupt file {file_path}: {e}")
        return None

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
    
    # Extract audio features and append them to the DataFrame
    dataframe['mfcc'] = dataframe['file_path'].apply(extract_mfcc)
    mfcc_dataframe = pd.DataFrame(dataframe['mfcc'].tolist(), columns=[f'mfcc_{i+1}' for i in range(13)])
    dataframe = pd.concat([dataframe, mfcc_dataframe], axis=1)
    dataframe['chroma'] = dataframe['file_path'].apply(extract_chroma)
    chroma_dataframe = pd.DataFrame(dataframe['chroma'].tolist(), columns=[f'chroma_{i+1}' for i in range(12)])
    dataframe = pd.concat([dataframe, chroma_dataframe], axis=1)
    dataframe['spectral_contrast'] = dataframe['file_path'].apply(extract_spectral_contrast)
    spectral_contrast_dataframe = pd.DataFrame(dataframe['spectral_contrast'].tolist(), columns=[f'spectral_contrast_{i+1}' for i in range(7)])
    dataframe = pd.concat([dataframe, spectral_contrast_dataframe], axis=1)
    dataframe['tonnetz'] = dataframe['file_path'].apply(extract_tonnetz)
    tonnetz_dataframe = pd.DataFrame(dataframe['tonnetz'].tolist(), columns=[f'tonnetz_{i+1}' for i in range(6)])
    dataframe = pd.concat([dataframe, tonnetz_dataframe], axis=1)
    dataframe['zcr'] = dataframe['file_path'].apply(extract_zcr)
    dataframe['rolloff'] = dataframe['file_path'].apply(extract_rolloff)

    dataframe = scale(dataframe)
    
    return dataframe

# Scale the features
def scale(dataframe):
    
    # Get all the audio feature columns
    feature_columns = [col for col in dataframe.columns if col not in ["file_path", "label", "mood"]]
    dataframe[feature_columns] = scaler.fit_transform(dataframe[feature_columns])

    return dataframe