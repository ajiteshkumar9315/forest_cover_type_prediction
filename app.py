import pandas as pd
import numpy as np
import streamlit as st
import pickle
from PIL import Image
import pyarrow.lib as _lib


# loading trained model
# rfc = pickle.load(open('rfc.pkl', 'rb'))
with open('rfc.pkl', 'rb') as model_file:
    rfc = pickle.load(model_file)

# creating web app
st.title('Forest Cover Type Prediction')
image = Image.open('img.png')
st.image(image, use_column_width=True)
user_input = st.text_input('Enter all cover type feature')

if user_input:
    user_input = user_input.split(',')
    feature = np.array([user_input], dtype=np.float64)
    prediction = rfc.predict(feature).reshape(1,-1)
    prediction = int(prediction[0])
    # st.write(prediction)

    # create the cover type dictionary
    cover_type_dict = {
        1: {"name": "Spruce/Fir", "image": "img_1.png"},
        2: {"name": "Lodgepole Pine", "image": "img_2.png"},
        3: {"name": "Ponderosa Pine", "image": "img_3.png"},
        4: {"name": "Cottonwood/Willow", "image": "img_4.png"},
        5: {"name": "Aspen", "image": "img_5.png"},
        6: {"name": "Douglas-fir", "image": "img_6.png"},
        7: {"name": "Krummholz", "image": "img_7.png"}
    }
    cover_type_info = cover_type_dict.get(prediction)

    if cover_type_info is not None:
        forest_name = cover_type_info['name']
        forest_img = cover_type_info['image']

        col1, col2 = st.columns([2,3])
        with col1:
            st.write('This is predict cover type')
            st.write(f"{forest_name}")
        with col2:
            final_img = Image.open(forest_img)
            st.image(final_img, caption = forest_name, use_column_width=True)


