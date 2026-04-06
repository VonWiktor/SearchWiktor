import os
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# --- KONFIGURACJA ---
COLLECTION_NAME = "moje_projekty"
MODEL_NAME = 'all-MiniLM-L6-v2'
BASE_PATH = "../" # Szukamy w folderze nadrzędnym

# 1. Inicjalizacja Modelu
print("🚀 Odpalam silnik AI (ładowanie modelu)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# 2. Inicjalizacja Bazy Danych
client = QdrantClient(path="./vector_db")

# 3. Bezpieczne utworzenie kolekcji (bez przestarzałych funkcji)
if not client.collection_exists(COLLECTION_NAME):
    print(f"📦 Tworzę nową bazę wektorową: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

def index_all_projects():
    print(f"🔍 Skanuję folder: {os.path.abspath(BASE_PATH)}")
    points = []
    idx = 0
    
    # TUTAJ BYŁA BRAKUJĄCA CZĘŚĆ KODU:
    for root, _, files in os.walk(BASE_PATH):
        # Omijamy foldery systemowe i wirtualne środowiska
        if any(skip in root for skip in [".venv", "__pycache__", ".git", "vector_db"]):
            continue
            
        for file in files:
            if file.endswith((".py", ".txt", ".md", ".csv", ".json", ".html")):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        short_content = content[:1500].strip()
                        
                        if len(short_content) < 10: 
                            continue
                        
                        vector = model.encode(short_content).tolist()
                        points.append(PointStruct(
                            id=idx, 
                            vector=vector, 
                            payload={
                                "name": file, 
                                "path": file_path, 
                                "content": short_content[:200]
                            }
                        ))
                        idx += 1
                        print(f"➕ Wczytano: {file}")
                except Exception as e:
                    print(f"❌ Błąd przy pliku {file}: {e}")

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ Zapisano {len(points)} plików do bazy!")
    else:
        print("⚠️ Nie znaleziono plików do zaindeksowania.")

def search_ai():
    print("\n" + "="*40)
    print(" 🧠 TWOJA WYSZUKIWARKA AI JEST GOTOWA ")
    print("="*40)
    print("(Wpisz 'q' aby wyjść)\n")
    
    while True:
        query = input("Czego szukasz?: ")
        if query.lower() == 'q': 
            print("Zamykanie...")
            break
            
        if not query.strip(): 
            continue
            
        query_vector = model.encode(query).tolist()
        
        # 4. Nowa funkcja szukania (zastępuje zepsute client.search)
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=3
        )
        
        results = response.points
        
        if not results:
            print("Brak trafień w bazie.\n")
            continue
            
        print("\n[WYNIKI SZUKANIA]:")
        for i, res in enumerate(results):
            name = res.payload.get('name', 'Brak nazwy')
            path = res.payload.get('path', 'Brak ścieżki')
            content = res.payload.get('content', '')
            
            print(f"{i+1}. 📄 {name} (Trafność: {res.score:.2f})")
            print(f"   Ścieżka: {path}")
            print(f"   Fragment: {content}...")
            print("-" * 30)
        print()

if __name__ == "__main__":
    # Bezpieczne sprawdzanie ilości plików w bazie
    ilosc_plikow = client.count(collection_name=COLLECTION_NAME).count
    
    if ilosc_plikow == 0:
        index_all_projects()
    else:
        print(f"ℹ️ Baza wczytana. Ilość plików w pamięci: {ilosc_plikow}")
    
    search_ai()