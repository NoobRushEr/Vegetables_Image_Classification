import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

data_cat = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon",
]

model = load_model("Image_classify.keras")

def main():
    st.set_page_config(
        page_title="Vegetables Image Classification",
        page_icon="vegetable",
    )

    st.header('Vegetables Image Classification')
    img_width = 180
    img_height = 180


    label = "Select an image"
    image = st.file_uploader(label, type=['jpg', 'png', 'jpeg'], accept_multiple_files=False, key=None, help=None, on_change=None)
    if image is not None:
        img = Image.open(image)
        st.image(img, width=200)
    else:
        image = "default.jpg"
        st.image(image, width=200)

    image_load = tf.keras.utils.load_img(image, target_size=(img_width, img_height))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)

    score = tf.nn.softmax(predict)

    st.write(
        "Veg/Fruit in image is {} with accuracy of {:0.2f}".format(
            data_cat[np.argmax(score)], np.max(score) * 100
        )
    )

if __name__=='__main__':
    main()