import streamlit as st  # type: ignore
from WeightedProjTools import *

st.title("Welcome to the Weighted Projective Tools App!")

st.markdown("""
This program does the following:
- Calculates the dimension of the space of a-weighted homogeneous polynomials of degree d in n+1 variables (linear system).
- Reduces the weights so they are well-formed.
- Reduces the degree when possible and calculates the dimension of the linear system again.
""")
