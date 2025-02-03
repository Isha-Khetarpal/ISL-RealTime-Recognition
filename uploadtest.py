from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog  # Import Tk and filedialog from tkinter

# Load the ISL detection model (update this path as necessary)
model = load_model(r'D:\SIH\by cnn\ISL_Classification_Model.h5')

def predict_image_category(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    return prediction

def handle_upload(img_path):
    # Open and display the image
    img = PIL.Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes for better display
    plt.show()

    # Predict the category of the uploaded image
    prediction = predict_image_category(img_path)

    # ISL categories (replace these with actual gesture labels)
    categories = [
        '1', '2', '3', '4', '5',
        '6', '7', '8', '9', 'gesture_A',
        'gesture_B', 'gesture_C', 'gesture_D', 'gesture_E', 'gesture_F',
        'gesture_G', 'gesture_H', 'gesture_I', 'gesture_J', 'gesture_K',
        'gesture_L', 'gesture_M', 'gesture_N', 'gesture_O', 'gesture_P',
        'gesture_Q', 'gesture_R', 'gesture_S', 'gesture_T', 'gesture_U',
        'gesture_V', 'gesture_W', 'gesture_X', 'gesture_Y', 'gesture_Z'
    ]

    predicted_category = categories[np.argmax(prediction)]
    print('Predicted Category:', predicted_category)

def upload_and_predict():
    # Open a file dialog to select an image
    root = Tk()
    root.withdraw()  # Hide the root window
    img_path = filedialog.askopenfilename(title='Select an Image for ISL Prediction')
    
    if img_path:
        handle_upload(img_path)

# Call the upload and predict function
upload_and_predict()
