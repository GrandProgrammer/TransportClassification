# PYTHON -M VENV VENV
import platform

import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

plt = platform.system()
if plt=='Linux':  pathlib.PosixPath = pathlib.WindowsPath


# TITLE
st.title('TRANSPORT CLASSIFICATION QILUVCHI MODEL ')

# PIC UPLOADDD ....
file = st.file_uploader("PIC UPLOAD", type=['png', 'jpeg', 'jpg', 'gif', 'svg'])


if file:
        st.image(file)
        # PIL - Image CONVERT

        img = PILImage.create(file)

        # MODEL....
        model = load_learner('transport_mnodel.pkl')

        # PREDICTION
        prediction, prediction_id, probability = model.predict(img)  # model.predict(filename)
        st.success(f"PREDICTION : {prediction}")
        st.info(f"PROBABILITY : {probability[prediction_id]*100:.1f}%")


        # PLOTTING....
        fig = px.bar(x=probability*100, y = model.dls.vocab)
        st.plotly_chart(fig)



















