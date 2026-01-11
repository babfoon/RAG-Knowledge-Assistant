import streamlit as st
from app.main import build_index, query_index

st.title('RAG Knowledge Assistant')

upload_file = st.file_uploader('Upload a document', type = ['pdf'])
if upload_file:
  with open('data/uploaded_document.pdf', 'wb') as f:
    f.write(uploaded_file.read())
  st.write('File successfully uploaded!')
  if st.button('Build Index'):
    build_index()
    st.write('Index has been built successfully!')

query = st.text_input('Ask away:')
if st.button('Submit'):
  response = query_index(query)
  st.write('Response:', response)
