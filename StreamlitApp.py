import streamlit as st # type: ignore


home_page = st.Page("HomePage.py", title="Home", icon=":material/home:")
#weight_normalizer_page = st.Page("WeightNormalizerPage.py", title="Weight Normalizer", icon=":material/format_list_bulleted:")
dimcalc_page = st.Page("DimensionCalculatorPage.py", title="Linear System Dimension Calculator", icon=":material/add_circle:")
sheaf_page = st.Page("SheafAmplenessPage.py", title="Sheaf Ampleness Checker", icon=":material/add_circle:")
embedding_page = st.Page("EmbeddingWPSPage.py", title="", icon=":material/delete:")


pg = st.navigation([home_page, dimcalc_page, embedding_page])
st.set_page_config(page_title="Data manager", page_icon=":material/edit:")
pg.run()


#maybe do multipage as follows :
# one page reduces the weights of weighted projective space
# one page calculates dimension of linear system (also reduces the weights)
# one page calculates the embedding dimension of the weighted projective space
# one page that checks if a sheaf is very ample or not
