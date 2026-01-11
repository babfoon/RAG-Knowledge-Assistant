import streamlit as st
from app.query import generate_response

st.title('RAG Knowledge Assistant')
query = st.text_input('Ask away:')
if st.button('Submit'):
  response = generate_response(query, 'Document Context...')
  st.write(reponse)
