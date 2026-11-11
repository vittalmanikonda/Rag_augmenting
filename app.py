import os
from flask import Flask, request, jsonify, render_template_string
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Note: We import the query helper from the interactive session 
# or define it locally to ensure smooth integration.
from __main__ import ask_local_model_with_db, descriptions

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Colab Local RAG Server</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f7f6; }
        .container { max-width: 700px; margin: auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        h2 { color: #333; }
        .section { margin-bottom: 25px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
        input[type="file"], input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #responseArea { background: #eef; padding: 15px; border-radius: 5px; margin-top: 15px; display: none; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📚 Local RAG Document Server</h2>

        <div class="section">
            <h3>1. Upload a Document (.md or .txt)</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" id="fileInput" accept=".md,.txt" required>
                <button type="submit">Upload & Index</button>
            </form>
            <div id="uploadStatus" style="margin-top:10px; color: green; font-weight: bold;"></div>
        </div>

        <div class="section">
            <h3>2. Query the Document</h3>
            <input type="text" id="queryInput" placeholder="Ask a question...">
            <button onclick="submitQuery()">Ask Model</button>
            <div id="responseArea"></div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData();
            formData.append('file', document.getElementById('fileInput').files[0]);

            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerText = "Uploading and processing...";

            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            statusDiv.innerText = result.message || result.error;
        };

        async function submitQuery() {
            const question = document.getElementById('queryInput').value;
            const responseArea = document.getElementById('responseArea');
            responseArea.style.display = 'block';
            responseArea.innerText = "Generating response...";

            const response = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });
            const result = await response.json();
            responseArea.innerText = result.answer || result.error;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload_file():
    global descriptions
    if 'file' not in request.files:
        return jsonify({"error": "No file partition provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join('/content', file.filename)
    file.save(file_path)

    try:
        loader = TextLoader(file_path)
        text_doc = loader.load()

        split_condition = [("##", "title")]
        splitter = MarkdownHeaderTextSplitter(split_condition)
        doc_splits = splitter.split_text(text_doc[0].page_content)
        doc_list = [split.page_content for split in doc_splits]

        # Re-index chunks securely
        import __main__
        __main__.descriptions = Chroma.from_texts(
            doc_list,
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            persist_directory="/tmp/chroma_db"
        )
        return jsonify({"message": f"Successfully uploaded and indexed '{file.filename}'!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"error": "Question missing"}), 400

    try:
        answer = ask_local_model_with_db(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
