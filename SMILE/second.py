import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow as tf

# Configuration for GPU memory allocation
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Load a pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained FaceNet model
facenet = cv2.dnn.readNetFromTensorflow('./facenet-master')

# Function to detect and align faces in an image and extract embeddings
def detect_align_and_extract_embeddings(image_path, output_folder):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    embeddings = []

    # Loop through detected faces
    for i, (x, y, w, h) in enumerate(faces):
        # Extract the detected face region
        face = image[y:y+h, x:x+w]

        # Resize the face to a common size for embedding extraction (e.g., 160x160 pixels)
        aligned_face = cv2.resize(face, (160, 160))

        # Preprocess the aligned face for FaceNet
        blob = cv2.dnn.blobFromImage(aligned_face, 1.0 / 127.5, (160, 160), (127.5, 127.5, 127.5), swapRB=True)

        # Pass the preprocessed face through FaceNet to obtain embeddings
        facenet.setInput(blob)
        embedding = facenet.forward()

        # Save the aligned face image
        aligned_face_filename = f"{i}_{os.path.basename(image_path)}"
        aligned_face_path = os.path.join(output_folder, aligned_face_filename)
        cv2.imwrite(aligned_face_path, aligned_face)

        # Append the embedding to the list
        embeddings.append({'Image1': aligned_face_path, 'Embedding': embedding.flatten()})

    return embeddings

# Read the original data from the text file
data = pd.read_csv('Cleaned_data.txt', delimiter='\t', header=None, names=['Image1', 'Image2', 'Label'])

# Create a new DataFrame for the processed data
new_data = pd.DataFrame(columns=['Image1', 'Image2', 'Label', 'Embedding'])

# Process each row in the original data
for index, row in data.iterrows():
    image1_path = row['Image1']
    image2_path = row['Image2']
    label = row['Label']

    # Detect, align, and extract embeddings for Image1 and Image2, saving them in the "train_processed" folder
    embeddings1 = detect_align_and_extract_embeddings(image1_path, 'train_processed')
    embeddings2 = detect_align_and_extract_embeddings(image2_path, 'train_processed')

    # Create new rows in the DataFrame for each aligned face pair and their embeddings
    for emb1, emb2 in zip(embeddings1, embeddings2):
        new_data = new_data.append({'Image1': emb1['Image1'], 'Image2': emb2['Image1'], 'Label': label, 'Embedding': emb1['Embedding']}, ignore_index=True)

# Save the processed data to a new text file
new_data.to_csv('Data_with_embeddings.txt', sep=',', header=False, index=False)
