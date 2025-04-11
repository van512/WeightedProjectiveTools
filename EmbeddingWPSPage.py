import streamlit as st # type: ignore
from WeightedProjTools import *

st.title("Embedding of Weighted Projective Space")



# Input fields
weights_input = st.text_input("Enter the weights (a₀, ..., aₙ) separated by spaces or commas (e.g., 1 2 3 or 1,2,3):")

def format_weights(weights):
    return ", ".join(str(x) for x in weights)

if st.button("Calculate"):
    try:
        # Allow both commas and spaces
        cleaned_input = weights_input.replace(",", " ")
        a = list(map(int, cleaned_input.split()))
        if not all(x > 0 for x in a):
            raise ValueError("All weights must be positive integers.")

        w = Weights(a)
        wps = WeightedProjectiveSpace(w)  # P(a)

        st.write(
            f"The weight `{w.weights}` reduces to `{w.reduced_weights}` and is equivalent to the well-formed weight `{w.wellformed_weights}`."
        )

        st.latex(
            rf"\mathbb{{P}}({format_weights(w.weights)}) = \mathbb{{P}}({format_weights(w.reduced_weights)}) = \mathbb{{P}}({format_weights(w.wellformed_weights)})"
        )

        st.write(f"The weighted projective space ℙ({w.wellformed_weights}) can be embedded into a classical projective space of dimension {wps.embedding_dimension}.")
        
        st.latex(
            rf"\mathbb{{P}}({format_weights(w.wellformed_weights)}) = \text{{Proj}}(\mathbb{{C}}[X]_{{{format_weights(w.wellformed_weights)}}}) = \text{{Proj}}(\mathbb{{C}}[X]_{{{wps.embedding_linear_system.degree}}}) = \text{{Proj}}(\mathbb{{C}}[x_0,\dots,x_{{{wps.embedding_dimension}}}]) = \mathbb{{P}}^{{{wps.embedding_dimension}}}"
        )

        st.latex(
            rf"\dim \mathcal{{C}}_{{{format_weights(w.wellformed_weights)}}}[X]_{{{wps.embedding_linear_system.degree}}} = {wps.embedding_linear_system.dimension}"
        )

    except ValueError:
        st.error("Error: Please enter only positive integers separated by spaces or commas.")
