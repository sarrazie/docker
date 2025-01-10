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

# Der Speicherort des Modells im Docker-Volume
MODEL_VOLUME_PATH = "/mnt/fasttext-model"  # Mount-Punkt des Docker-Volumes

# Lade das FastText-Modell, wenn es noch nicht im Volume existiert
def download_and_extract_model():
    model_file_path = os.path.join(MODEL_VOLUME_PATH, "cc.de.300.bin")

    # Prüfen, ob das Modell bereits existiert
    if not os.path.exists(model_file_path):
        # Das Modell herunterladen, wenn es noch nicht vorhanden ist
        try:
            response = requests.get(MODEL_URL, stream=True)

            if response.status_code != 200:
                return None, f"Failed to download model: {response.status_code}"

            # Temporäre Datei für die komprimierte Datei erstellen
            temp_compressed_path = tempfile.NamedTemporaryFile(delete=False, suffix=".gz")
            with open(temp_compressed_path.name, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)

            # Temporäre Datei für die entpackte Datei erstellen
            os.makedirs(MODEL_VOLUME_PATH, exist_ok=True)
            with gzip.open(temp_compressed_path.name, "rb") as compressed_file:
                with open(model_file_path, "wb") as uncompressed_file:
                    uncompressed_file.write(compressed_file.read())

            # Entferne die komprimierte Datei
            os.unlink(temp_compressed_path.name)
            return model_file_path, "Model downloaded and extracted successfully."

        except Exception as e:
            return None, f"Error during download or extraction: {str(e)}"

    return model_file_path, "Model already exists."

# Lade das Modell beim Start, wenn es noch nicht im Volume existiert
MODEL_PATH, message = download_and_extract_model()

if MODEL_PATH is None:
    model_status = {"error": message}
else:
    model_status = {"success": message}

# Lade das Modell
model = None
if MODEL_PATH is not None:
    try:
        model = fasttext.load_model(MODEL_PATH)
    except Exception as e:
        model_status = {"error": f"Failed to load model: {str(e)}"}

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
    # Rückgabe des Modellstatus und des Speicherverbrauchs
    return jsonify({
        "status": "API is running",
        "model_status": model_status,
        "memory_usage_gb": memory_usage
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)

