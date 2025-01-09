import requests
import json

# URL der API
url = "http://localhost:5000/get_vector"

# Anfrage für das Wort "Hund"
data = {"word": "Hund"}
response = requests.post(url, json=data)

if response.status_code == 200:
    print("Antwort von /get_vector:", response.json())
else:
    print("Fehler:", response.status_code)

# Ähnlichkeit zwischen "Hund" und "Katze"
url_similarity = "http://localhost:5000/similarity"
data_similarity = {"word1": "Hund", "word2": "Katze"}
response_similarity = requests.post(url_similarity, json=data_similarity)

if response_similarity.status_code == 200:
    print("Antwort von /similarity:", response_similarity.json())
else:
    print("Fehler:", response_similarity.status_code)
