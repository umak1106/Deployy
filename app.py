import numpy as np
import streamlit as st
from keras.models import load_model
import app1
import app2




PAGES = {
    "Flood Damage Detection":app1 ,
    "Wildfire Detection" : app2,




}
st.sidebar.title('Disaster Damage Detection')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()