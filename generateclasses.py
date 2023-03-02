import numpy as np
import pandas as pd
import sklearn.preprocessing

# Load the fer2013 dataset
data = pd.read_csv('fer2013.csv')

# Extract the emotion labels from the dataset
labels = data['emotion']

# Convert the emotion labels to integer values
le = sklearn.preprocessing.LabelEncoder()
int_labels = le.fit_transform(labels)

# Save the integer-encoded emotion labels as a numpy array
np.save('classes.npy', int_labels)
