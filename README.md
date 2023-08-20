# Wound-Healing-Time-Prediction-Model

The wound healing time prediction model, developed using Python, aims to forecast the duration it takes for wounds to heal. Leveraging deep learning algorithms, the model processes various input factors such as wound images' intensity and depth. These features are extracted and analyzed to create predictive insights. Through this model, healthcare professionals can anticipate the time required for wounds to heal, aiding in patient care and treatment planning.

## Prerequisites
In order to run the Python script, your system must have the following programs/packages installed and the contact number should be saved in your phone (You can use bulk contact number saving procedure of email). There is a way without saving the contact number but has the limitation to send the attachment.
* Python 3.8: Download it from https://www.python.org/downloads
* Tensorflow: Run in command prompt **pip install Tensorflow**
* Tensorflow_hub : Run in command prompt **pip install --upgrade tensorflow-hub**
* cv2: Run in command prompt **pip install opencv-python**
* Numpy: Run in command prompt **pip install Numpy**
* KerasLayer: ("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4")

 ## Code
```
def estimate_healing_time(age_classification, depth_classification):
    """ Estimates healing time based on cross-classification of age and depth.

    Args:
        age_classification: Age classification of the wound ("New," "Old," or "Intermediate").
        depth_classification: Depth classification of the wound ("Shallow," "Moderate," or "Deep").

    Returns:
        An estimated healing time in days.
    """

    healing_time_map = {
        ("New", "Shallow"): "2-4 weeks",
        ("New", "Moderate"): "3-5 weeks",
        ("New", "Deep"): "4-6 weeks",
        ("Old", "Shallow"): "2-4 weeks",
        ("Old", "Moderate"): "3-5 weeks",
        ("Old", "Deep"): "5-7 weeks",
        ("Intermediate", "Shallow"): "1-3 weeks",
        ("Intermediate", "Moderate"): "2-4 weeks",
        ("Intermediate", "Deep"): "3-6 weeks"
    }

    return healing_time_map.get((age_classification, depth_classification), "Unknown")

# Example age and depth classifications
age_classification = "Old"
depth_classification = "Deep"

# Estimate healing time based on cross-classification
healing_time = estimate_healing_time(age_classification, depth_classification)

# Print the estimated healing time
print("Estimated Healing Time:", healing_time)
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import cv2
import os

# Load the pre-trained MobileNetV2 model from TensorFlow Hub
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4")
])

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize to match the model input shape
    image = np.array(image) / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Classify the image
def classify_image(image_path):
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    return predictions

# Define depth ranges
depth_ranges = {
    'Shallow': (4.5, 5.9),
    'Moderate': (6.0, 7.8),
    'Deep': (7.9, float('inf'))
}
# Define intensity ranges (update with your specific ranges)
intensity_ranges = {
    'Low': (0, 0.3),
    'Medium': (0.31, 0.7),
    'High': (0.71, 1.0)
}
intensity_range = None
for intensity_label, (min_intensity, max_intensity) in intensity_ranges.items():
    if min_intensity <= intensity <= max_intensity:
        intensity_range = intensity_label
        break

if intensity_range is None:
    intensity_range = "Unknown"

# Define healing time map based on age, depth, and intensity
healing_time_map = {
    ("New", "Shallow", "Low"): "2-4 weeks",
    ("New", "Shallow", "Medium"): "3-5 weeks",
    ("New", "Shallow", "High"): "4-6 weeks",
    # ... (add more combinations based on your requirements)
}

# Actual image path
image_path = '/content/drive/MyDrive/data/train_images/0016.png'

# Classify the image and get the predicted class
predictions = classify_image(image_path)

# Get the class index with the highest probability
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]
# Determine the depth based on class index and confidence
if 400 <= class_index <= 450:
    depth_range = 'Shallow'
elif 800 <= class_index <= 1000:
    depth_range = 'Deep'
else:
    depth_range = 'Unknown'

# Calculate the average pixel intensity within the segmented wound region
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
intensity = np.mean(gray[binary == 255])

# Determine intensity range based on intensity value
for intensity_label, (min_intensity, max_intensity) in intensity_ranges.items():
    if min_intensity <= intensity <= max_intensity:
        intensity_range = intensity_label
        break

# Determine age classification (you might have this from previous steps)
age_classification = "New"  # Replace with the actual classification

# Estimate healing time based on age, depth, and intensity classifications
estimated_healing_time = healing_time_map.get((age_classification, depth_range, intensity_range), "Unknown")

# Print the results for the provided image
print("Image Path:", image_path)
print("Class Index:", class_index)
print("Depth Range:", depth_range)
print("Estimated Intensity:", intensity)
print("Intensity Range:", intensity_range)
print("Estimated Healing Time:", estimated_healing_time)
