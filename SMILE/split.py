import pandas as pd
from sklearn.model_selection import train_test_split

# Load your data from the 'Cleaned_data.txt' file into a DataFrame
data = pd.read_csv('SMILE/Cleaned_Data.txt', sep=', ',  header=None, names=['Image1', 'Image2', 'Label'])

# Split the data into training (70%), validation (15%), and testing (15%) sets
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Now you have your data split into training, validation, and testing sets
# You can access the data and labels as follows:
train_images1, train_images2, train_labels = train_data['Image1'], train_data['Image2'], train_data['Label']
val_images1, val_images2, val_labels = val_data['Image1'], val_data['Image2'], val_data['Label']
test_images1, test_images2, test_labels = test_data['Image1'], test_data['Image2'], test_data['Label']
