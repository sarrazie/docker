from flask import Flask, request, jsonify
import fasttext
import psutil
import requests
import tempfile
import gzip
import os

app = Flask(__name__)

# URL für das FastText-Modell
MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.de.300.bin.gz"  

# Lade das FastText-Modell beim Start des Servers
def download_and_extract_model():
    print(f"Downloading FastText model from {MODEL_URL}...")
    response = requests.get(MODEL_URL, stream=True)

    if response.status_code != 200:
       raise Exception(f"Failed to download model: {response.status_code}")

# Temporäre Datei für die komprimierte Datei erstellen
    temp_compressed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gz")
    with open(temp_compressed_path.name, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

# Temporäre Datei für die entpackte Datei erstellen
    temp_model_path = tempfile.NamedTemporaryFile(delete=False)
    with gzip.open(temp_compressed_path.name, "rb") as compressed_file:
         with open(temp_model_path.name, "wb") as uncompressed_file:
            uncompressed_file.write(compressed_file.read())

# Entferne die komprimierte Datei, da sie nicht mehr benötigt wird
    os.unlink(temp_compressed_path.name)

    return temp_model_path.name

# Modell beim Start laden
MODEL_PATH = download_and_extract_model()
print(f"FastText model extracted to {MODEL_PATH}!")

# Lade das Modell
model = fasttext.load_model(MODEL_PATH)
print("FastText model loaded successfully!")

# API-Endpoint: Vektor für ein Wort abrufen
@app.route("/get_vector", methods=["POST"])
def get_vector():
    data = request.json
    word = data.get("word")

    if not word:
        return jsonify({"error": "No word provided"}), 400

# Vektor für das Wort abrufen
    vector = model.get_word_vector(word).tolist()
    return jsonify({"word": word, "vector": vector})


# API-Status überprüfen
@app.route("/", methods=["GET"])
def health_check():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 3)  # In GB
    return jsonify({"status": "API is running", "memory_usage_gb": memory_usage})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
