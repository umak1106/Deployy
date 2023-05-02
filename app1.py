import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


def app():
    model_path = 'tomnod_2_100epochs_Adam.h5'
    model = load_model(model_path)

    def predict(image_file):
        # Load the image
        img = image.load_img(image_file, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.vgg16.preprocess_input(
            x)  # adjust preprocessing function as needed

        # Make predictions
        preds = model.predict(x)

        # Return the prediction
        return preds[0][0]

    # Create a Streamlit app
    st.title("Damage Detection on Post-Hurricane Satellite Imagery")

    # Create a file uploader
    uploaded_file = st.file_uploader("Choose an image...",
                                     type=["jpg", "jpeg", "png"])

    # If an image was uploaded
    if uploaded_file is not None:
        # Make a prediction and display the result
        prediction = predict(uploaded_file)


        if prediction == 1.0:
            st.success(
                'Not Damaged')
        else:
            st.error(
                'Damaged')





