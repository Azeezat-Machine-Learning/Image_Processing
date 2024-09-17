import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Title of the app
st.title("Image Processing and Predictive System")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Step 2: Load the image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array for further processing
    img_array = np.array(image)

    # Step 3: Grayscale conversion option
    grayscale = st.checkbox("Convert to grayscale")

    # If the user wants to convert to grayscale
    if grayscale:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        st.image(
            img_array, caption="Grayscale Image", use_column_width=True, clamp=True
        )

    # Step 4: Resize option
    resize = st.checkbox("Resize image")

    if resize:
        # Input dimensions for resizing
        width = st.number_input("Enter width", min_value=1, max_value=1000, value=200)
        height = st.number_input("Enter height", min_value=1, max_value=1000, value=200)
        img_array = cv2.resize(img_array, (width, height))
        st.image(
            img_array, caption=f"Resized Image: {width}x{height}", use_column_width=True
        )

    # Step 5: Image normalization (scaling) to [0, 1] for model input
    img_array = img_array / 255.0

    # Step 6: Reshaping image for model input (assuming the model was trained on images of a specific shape)
    # Replace (200, 200, 3) with your model's input shape
    input_shape = (200, 200, 3) if not grayscale else (200, 200, 1)

    try:
        # If grayscale, we need to reshape differently
        if grayscale:
            img_array = img_array.reshape(1, img_array.shape[0], img_array.shape[1], 1)
        else:
            img_array = img_array.reshape(
                1, img_array.shape[0], img_array.shape[1], img_array.shape[2]
            )

        st.write(f"Image reshaped to: {img_array.shape}")

    except Exception as e:
        st.error(f"Error reshaping image: {e}")

    # Now the image is ready for prediction (assuming you have a pre-trained model)
    # You can load and use your model here:
    # Example:
    # model = load_model('your_model.h5')
    # prediction = model.predict(img_array)
    # st.write(f"Prediction: {prediction}")
