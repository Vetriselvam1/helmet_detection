import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import os

# Load the model
model = load_model("C:/Users/sakth/PycharmProjects/helmetdetction/keras_model.h5", compile=False)

# Load the labels
class_names = [class_name.strip() for class_name in open("C:/Users/sakth/PycharmProjects/helmetdetction/labels.txt", "r").readlines()]

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Function to capture images continuously when "No Helmet" is detected
def capture_images_continuously():
    # Open the camera (default is the first camera device)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Press 'q' to stop capturing. Will capture images continuously when 'No Helmet' is detected.")

    # Counter for captured images
    captured_images = 0
    save_path = "C:/Users/sakth/PycharmProjects/helmetdetction/photos"  # Path to save images
    save_path1 = "C:/Users/sakth/PycharmProjects/helmetdetction/diffhelmetphotos"
    # Ensure the photos directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path1):
        os.makedirs(save_path1)

    # Variable to track if photos should be captured
    capturing = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Show the captured frame
        cv2.imshow("Capture Image - Press 'q' to stop", frame)

        # Predict using the current frame
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array
        prediction = model.predict(data)
        index = np.argmax(prediction)

        # If "No Helmet" (Class 1) is detected, start capturing photos
        if index == 1:  # Class 1 corresponds to "No Helmet"
            if not capturing:
                capturing = True
                print("No Helmet detected, starting photo capture.")
            # Capture the frame and save it
            capture_path = os.path.join(save_path1, f"photo{captured_images + 1}.jpg")
            cv2.imwrite(capture_path, frame)
            captured_images += 1
            print(f"Image {captured_images} saved as {capture_path}")
        if index == 2:  # Class 1 corresponds to "No Helmet"
            if not capturing:
                capturing = True
                print("No Helmet detected, starting photo capture.")
            # Capture the frame and save it
            capture_path = os.path.join(save_path, f"photo{captured_images + 1}.jpg")
            cv2.imwrite(capture_path, frame)
            captured_images += 1
            print(f"Image {captured_images} saved as {capture_path}")
        elif index == 0:  # Class 0 corresponds to "Helmet"
            if capturing:
                print("Helmet detected, stopping image capture.")
                capturing = False

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture stopped by user.")
            break

    # Release the camera and close the OpenCV window
    cap.release()
    cv2.destroyAllWindows()

# Start capturing images if no helmet is detected
capture_images_continuously()
