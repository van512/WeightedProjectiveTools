import streamlit as st
from WeightedProjTools import *

# Ensure that WeightedProjTools does not invoke Tkinter or GUI elements.

# Streamlit app
st.title("Weight Reduction and Linear System Dimension Calculator")

st.markdown("""
This program does the following:
- Calculates the dimension of the space of a-weighted homogeneous polynomials of degree d in n+1 variables (linear system).
- Reduces the weights so they are well-formed.
- Reduces the degree when possible and calculates the dimension of the linear system again.
""")

# Input fields
weights_input = st.text_input("Enter the weights (a_0, ..., a_n) separated by spaces (a_i>0):")
degree_input = st.text_input("Enter the degree (d):")

if st.button("Calculate"):
    try:
        a = list(map(int, weights_input.split()))
        d = int(degree_input)

        w = Weights(a)
        Cad = LinearSystem(w, d)  # C_a[X]_d

        if Cad.isreducible:
            result_string = (
                f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the well-formed weight {w.wellformed_weights}.\n"
                f"Thus we have the following isomorphisms:\n"
                f"P({w.weights}) = P({w.reduced_weights}) = P({w.wellformed_weights}).\n\n"
                f"The dimension of the associated linear system of degree {d} is dim C\_{w.weights}[X]\_{d} = {Cad.dim_before}.\n\n"
                f"The degree is reducible.\n"
                f"The reduced degree is {Cad.reduced_degree} and the corresponding well-formed degree is {Cad.wellformed_degree}.\n\n"
                f"We see that the dimensions of the linear systems agree:\n"
                f"dim C\_{w.wellformed_weights}[X]\_{Cad.wellformed_degree} = {Cad.dim_after}"
            )
        else:
            result_string = (
                f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the well-formed weight {w.wellformed_weights}.\n"
                f"Thus we have the following equalities:\n"
                f"P({w.weights}) = P({w.reduced_weights}) = P({w.wellformed_weights}).\n\n"
                f"The dimension of the associated linear system of degree {d} is dim C_{w.weights}[X]_{d} = {Cad.dim_before}.\n"
                f"The degree is not reducible."
            )

        st.success(result_string)

    except ValueError:
        st.error("Error: Please enter valid integers.")
