import os
import torch
import warnings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Wyłączenie zbędnych komunikatów od bibliotek
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# --- KONFIGURACJA ---
COLLECTION_NAME = "moje_projekty"
MODEL_NAME = 'all-MiniLM-L6-v2'
BASE_PATH = "../" 

# 1. Start silnika
print("🤖 Inicjalizacja AI... (Cierpliwości)")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
client = QdrantClient(path="./vector_db")

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def index_all_projects():
    print(f"📂 Przeszukuję pliki w: {os.path.abspath(BASE_PATH)}...")
    points = []
    idx = 0
    
    for root, _, files in os.walk(BASE_PATH):
        if any(skip in root for skip in [".venv", "__pycache__", ".git", "vector_db"]):
            continue
            
        for file in files:
            if file.endswith((".py", ".txt", ".md", ".csv", ".json", ".html")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read().strip()
                        if len(content) < 10: continue
                        
                        vector = model.encode(content[:1500]).tolist()
                        points.append(PointStruct(
                            id=idx, 
                            vector=vector, 
                            payload={"name": file, "path": file_path, "content": content[:150]}
                        ))
                        idx += 1
                except:
                    continue # Ciche pomijanie błędnych plików

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Baza gotowa! Zapamiętano {len(points)} plików.")

def search_ai():
    print("\n" + "—" * 50)
    print("      🔍 LOKALNA WYSZUKIWARKA WIKTORA")
    print("       (wpisz 'q' aby wyjść z programu)")
    print("—" * 50)
    
    while True:
        query = input("\n[PYTANIE]: ")
        if query.lower() == 'q': break
        if not query.strip(): continue
            
        query_vector = model.encode(query).tolist()
        response = client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=3)
        results = response.points
        
        if not results:
            print("❌ Nic nie znalazłem.")
            continue
            
        print("\n[NAJLEPSZE TRAFIENIA]:")
        for i, res in enumerate(results):
            # Obliczanie procentu trafności
            score = round(res.score * 100, 1)
            print(f"{i+1}. ⭐ {res.payload['name']} [{score}% trafności]")
            print(f"   📂 Ścieżka: {res.payload['path']}")
            print(f"   📝 Skrót: {res.payload['content']}...")
            print("-" * 20)

if __name__ == "__main__":
    count = client.count(collection_name=COLLECTION_NAME).count
    if count == 0:
        index_all_projects()
    else:
        print(f"ℹ️ Silnik gotowy. Masz w pamięci {count} plików.")
    
    search_ai()