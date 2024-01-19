import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image
from keras.applications.vgg16 import VGG16
from sklearn.metrics.pairwise import cosine_similarity


col1, col2 = st.columns(2)

with col1:
   st.header("Upload Here!")
   uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

with col2:
   st.header("Furniture Category")
   option = st.selectbox(
    "Select the furniture's category: ",
    ('Chair', 'Table', 'Sofa', 'Bed', 'TV-Console', 'Storage'))
   option = option.lower()


vgg16 = VGG16(weights='imagenet', include_top=False, pooling='max', input_shape=(224, 224, 3))

# Disable training for VGG16 layers
for model_layer in vgg16.layers:
    model_layer.trainable = False

def load_image(image_path):
    input_image = Image.open(image_path)
    resized_image = input_image.resize((224, 224))
    return resized_image

def get_image_embeddings(object_image):
    # Resize the image to (224, 224)
    object_image = object_image.resize((224, 224))

    # Convert the image to a numpy array
    image_array = keras_image.img_to_array(object_image)

    # Remove the alpha channel if it exists
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Add an extra dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the image embedding
    image_embedding = vgg16.predict(image_array)

    return image_embedding

def get_similarity_score(first_image, second_image):
    first_image_vector = get_image_embeddings(first_image)
    second_image_vector = get_image_embeddings(second_image)
    similarity_score = cosine_similarity(first_image_vector, second_image_vector).reshape(1,)
    return similarity_score

def show_image(image_object):
    st.image(image_object, use_column_width=True)


if uploaded_file is not None:
    # Load the uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

    # Compute similarity scores and display closest furniture image
    closest_furniture = {
        'name': 'none',
        'score': 0
    }

    images_directory = f'/Users/jingyuan/Desktop/ScanTeak/Scanteak_Furniture_Detector_Matcher/scanteak_images/{option}'
    for filename in os.listdir(images_directory): #Goes through all of the images in that category
        image_path = os.path.join(images_directory, filename) #Image path of each image
        if filename.lower() == '.ds_store':
            continue
        similarity_score = get_similarity_score(uploaded_image, load_image(image_path)) #Gets both uploaded image and image inside the category similarity score
        if similarity_score > closest_furniture['score']: #Is trying to find the image with the highest similarity score as uploaded image
            closest_furniture['name'] = filename #Stores the details of the most similar image
            closest_furniture['score'] = similarity_score

    # Display closest furniture image
    closest_image_path = os.path.join(images_directory, closest_furniture['name'])
    closest_image = Image.open(closest_image_path)
    st.image(closest_image, caption=f"Closest Furniture: {closest_furniture['name']}", use_column_width=True)
