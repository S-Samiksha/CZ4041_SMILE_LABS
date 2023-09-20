import dlib
import cv2

# Initialize the face detector and shape predictor from Dlib
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

# Function to detect and align faces in an image
def detect_and_align_faces(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    aligned_faces = []

    # Loop through detected faces
    for face in faces:
        # Get facial landmarks
        landmarks = shape_predictor(gray, face)

        # Extract the coordinates of the eyes
        left_eye_x, left_eye_y = landmarks.part(36).x, landmarks.part(36).y
        right_eye_x, right_eye_y = landmarks.part(45).x, landmarks.part(45).y

        # Calculate the angle between the eyes
        angle = -1 * (180 / 3.141592) * \
            (atan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x))

        # Rotate the image to align the eyes horizontally
        rotated_image = cv2.warpAffine(
            image, cv2.getRotationMatrix2D((face.width() / 2, face.height() / 2), angle, 1), (face.width(), face.height()))

        aligned_faces.append(rotated_image)

    return aligned_faces

# Example usage:
image_path = "path/to/your/image.jpg"
aligned_faces = detect_and_align_faces(image_path)

# The 'aligned_faces' variable now contains aligned face images (if faces were detected)
