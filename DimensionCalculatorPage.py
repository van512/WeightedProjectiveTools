import streamlit as st # type: ignore
from WeightedProjTools import *

st.title("Linear System Dimension Calculator")


# Input fields
weights_input = st.text_input("Enter the weights (a_0, ..., a_n) separated by spaces (a_i>0):")
degree_input = st.text_input("Enter the degree (d):")

if st.button("Calculate"):
    try:
        a = list(map(int, weights_input.split())) # do such that can split also by comma or anything 
        d = int(degree_input)

        w = Weights(a)
        Cad = LinearSystem(w, d)  # C_a[X]_d
        wps = WeightedProjectiveSpace(w)  # P(a)

        if Cad.isreducible:
            result_string = (
                f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the well-formed weight {w.wellformed_weights}.\n"
                f"Thus we have the following isomorphisms:\n"
                f"P({w.weights}) = P({w.reduced_weights}) = P({w.wellformed_weights}).\n\n"
                #f"The dimension of the associated linear system of degree {d} is dim C\_{w.weights}[X]\_{d} = {Cad.dimension}.\n\n"
                f"The degree is reducible.\n"
                f"The reduced degree is {Cad.reduced_degree} and the corresponding well-formed degree is {Cad.wellformed_degree}.\n\n"
                #f"We see that the dimensions of the linear systems agree:\n"
                f"dim C\_{w.wellformed_weights}[X]\_{Cad.wellformed_degree} = {Cad.dimension}\n\n"
            )
        else:
            result_string = (
                f"The weight {w.weights} reduces to {w.reduced_weights} and it is equivalent to the well-formed weight {w.wellformed_weights}.\n"
                f"Thus we have the following equalities:\n"
                f"P({w.weights}) = P({w.reduced_weights}) = P({w.wellformed_weights}).\n\n"
                f"The dimension of the associated linear system of degree {d} is dim C\_{w.weights}[X]\_{d} = {Cad.dimension}.\n"
                f"The degree is not reducible.\n\n"
            )

        embedding_string = (
            f"The weighted projective space P({w.wellformed_weights}) can be embedded into the projective space of dimension {wps.embedding_dimension}. We have :\n\n"
            f"P({w.wellformed_weights}) = Proj(C[X]\_{w.wellformed_weights}) = Proj(C[X]\_{w.wellformed_weights}^{wps.embedding_linear_system.degree}) = Proj(C[x_0,...,x_{wps.embedding_dimension}]) = P^{wps.embedding_dimension}\n"
            f"This is because dim C\_{w.wellformed_weights}[X]\_{wps.embedding_linear_system.degree} = {wps.embedding_linear_system.dimension}.\n\n"
        )

        st.success(result_string+embedding_string)


        st.write(
            f"The weight `{w.weights}` reduces to `{w.reduced_weights}` and it is equivalent to the well-formed weight `{w.wellformed_weights}`."
        )

        st.latex(
            f"P({w.weights}) = P({w.reduced_weights}) = P({w.wellformed_weights})"
        )

        st.write("The degree is reducible.")
        st.write(
            f"The reduced degree is `{Cad.reduced_degree}` and the corresponding well-formed degree is `{Cad.wellformed_degree}`."
        )

        st.latex(
            f"\\dim \\mathcal{{C}}_{{{w.wellformed_weights}}}[X]_{{{Cad.wellformed_degree}}} = {Cad.dimension}"
        )



    except ValueError:
        st.error("Error: Please enter positive integers.")
