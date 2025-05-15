import streamlit as st

# third parties imports
import numpy as np 
import pandas as pd

# -- add new imports here --
import matplotlib.pyplot as plt
from collections import Counter

# local imports
from constants import Constant as C
from loaders import load_ratings
from loaders import load_items

st.title("Hello Streamlit-er 👋")
st.markdown(
    """ 
    Test to display the items.
    """
)

df_items = load_items()
st.dataframe(df_items)