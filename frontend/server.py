import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.main import build_index, query_index
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
UPLOAD_FOLDER = 'data'
INDEX_FOLDER = 'embeddings'
DOCUMENT_NAME = 'upload_document.pdf'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route to the home page
@app.route('/')
def index():
  return render_template('index.html')

# Route upload of PDF
@app.route('/upload', methods=['POST'])
def upload():
  if 'file' not in request.files:
    return 'No file part', 400

  file = request.files['file']
  if file.name == '':
    return 'No selected file', 400

  if file:
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], DOCUMENT_NAME)
    file.save(filepath)
    print(f'File uploaded and saved to {filepath}')

    # Build the FAISS index
    build_index()
    return redirect(url_for('ask'))

# Route for questions
@app.route('/ask', methods=['GET', 'POST'])
def ask():
  if request.method == 'POST':
    query = request.form.get('question')
    if query:
      answer = query_index(query)
      return render_template('ask.html', question=query, answer=answer)

  return render_template('ask.html', question=None, answer=None)

if __name__ == '__main__':
  if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
  if not os.path.exists(INDEX_FOLDER):
    os.makedirs(INDEX_FOLDER)

  app.run(debug=True, host='0.0.0.0', port=5000)
