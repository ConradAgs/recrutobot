from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import json
import numpy as np
import os
import traceback
import logging
import requests
from pathlib import Path
import tempfile
import gzip
from huggingface_hub import InferenceClient
import hashlib
import pathlib

# =======================
# Configuration du logging
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# URLs des fichiers Blob
# =======================
BLOB_FILE_URLS = {
    "embeddings": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/embedding_compressed.npz",
    "offers": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/jobs_catalogue2.json.gz"
}

# =======================
# Clé Hugging Face
# =======================
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "sentence-transformers/all-mpnet-base-v2"

# =======================
# Application FastAPI
# =======================
app = FastAPI(title="RecrutoBot", description="Avec Vercel Blob Storage")

# Configuration des templates
try:
    templates_path = Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_path))
except Exception as e:
    logger.error(f"Erreur templates: {e}")
    # Fallback pour Vercel
    templates = Jinja2Templates(directory="templates")

# =======================
# Hugging Face Embeddings
# =======================
def get_embedding(text: str):
    """Génère un embedding avec Hugging Face"""
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Token Hugging Face non configuré")
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=HF_API_TOKEN)
        
        embeddings = client.feature_extraction(
            text,
            model=HF_MODEL
        )
        
        emb = np.array(embeddings, dtype=np.float32)
        
        if emb.ndim == 2:
            emb = emb[0]
            
        logger.info(f"✅ Embedding généré - Shape: {emb.shape}")
        return emb
        
    except Exception as e:
        logger.error(f"❌ Erreur Hugging Face: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération embedding: {e}")

# =======================
# DataStore
# =======================
class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False
        self.cache_dir = pathlib.Path("/tmp/datastore")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, name: str) -> pathlib.Path:
        return self.cache_dir / name

    async def load_data(self):
        if self.data_loaded:
            return True

        try:
            logger.info("📥 Chargement depuis cache ou Blob Store...")

            # ===============================
            # 1. Charger les embeddings
            # ===============================
            emb_path = self._get_cache_path("embeddings_compressed.npz")

            if not emb_path.exists():
                logger.info("🧠 Téléchargement des embeddings...")
                emb_response = requests.get(BLOB_FILE_URLS["embeddings"], timeout=120)
                emb_response.raise_for_status()

                # Sauvegarde compressée
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
                    f.write(emb_response.content)
                    tmp_path = f.name

                data = np.load(tmp_path)
                os.unlink(tmp_path)

                # Sauvegarde compressée localement
                np.savez_compressed(emb_path, embeddings=data['embeddings'].astype(np.float32))
                logger.info(f"💾 Embeddings sauvegardés en cache ({emb_path})")

            # Charger depuis cache compressé
            self.offers_emb = np.load(emb_path)['embeddings']
            logger.info(f"✅ Embeddings chargés ({self.offers_emb.shape})")

            # ===============================
            # 2. Charger les offres
            # ===============================
            offers_path = self._get_cache_path("offers.json.gz")

            if not offers_path.exists():
                logger.info("📋 Téléchargement des offres...")
                json_response = requests.get(BLOB_FILE_URLS["offers"], timeout=60)
                json_response.raise_for_status()

                with open(offers_path, "wb") as f:
                    f.write(json_response.content)
                logger.info("💾 Offres sauvegardées en cache")

            # Charger depuis cache
            with gzip.open(offers_path, "rb") as f:
                self.offers = json.loads(f.read().decode('utf-8'))

            logger.info(f"✅ {len(self.offers)} offres chargées")

            self.data_loaded = True
            return True

        except Exception as e:
            logger.error(f"❌ Erreur load_data: {e}")
            logger.error(traceback.format_exc())
            return False

data_store = DataStore()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        if not data_store.data_loaded:
            success = await data_store.load_data()
            if not success:
                return HTMLResponse("""
                    <html><body>
                    <h1>Erreur de chargement</h1>
                    <p>Impossible de charger les données. Réessayez plus tard.</p>
                    </body></html>
                """)
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Erreur read_root: {e}")
        return HTMLResponse(f"<h1>Erreur: {str(e)}</h1>")

@app.post("/api/search")
async def search_offers(request: Request):
    try:
        if not data_store.data_loaded:
            success = await data_store.load_data()
            if not success:
                raise HTTPException(status_code=500, detail="Données non chargées")

        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt vide")

        query_emb = get_embedding(prompt)

        if data_store.offers_emb is None:
            raise HTTPException(status_code=500, detail="Embeddings manquants")

        # Similarité cosinus
        norms = np.linalg.norm(data_store.offers_emb, axis=1)
        query_norm = np.linalg.norm(query_emb)
        cos_scores = np.dot(data_store.offers_emb, query_emb) / (norms * query_norm)

        good_indices = np.where(cos_scores > 0.3)[0]

        if len(good_indices) == 0:
            return JSONResponse({
                "results": [],
                "message": "Aucune offre trouvée. Reformulez votre recherche.",
                "count": 0,
                "search_term": prompt
            })

        results = []
        for i in good_indices:
            offer = data_store.offers[i]
            results.append({
                "id": offer.get("id", ""),
                "intitule": offer.get("intitule", "Titre non disponible"),
                "description": offer.get("description", "Description non disponible")[:250] + "...",
                "lieuTravail": offer.get("lieuTravail", {}),
                "typeContrat": offer.get("typeContrat", ""),
                "typeContratLibelle": offer.get("typeContratLibelle", ""),
                "experienceLibelle": offer.get("experienceLibelle", ""),
                "salaire": offer.get("salaire", {}),
                "entreprise": offer.get("entreprise", {}),
                "origineOffre": offer.get("origineOffre", {}),
                "score": float(cos_scores[i])
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        
        return JSONResponse({
            "results": results[:20],  # Limiter à 20 résultats
            "message": f"{len(results)} offres trouvées pour '{prompt}'",
            "count": len(results),
            "search_term": prompt
        })

    except Exception as e:
        logger.error(f"Erreur search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    })

# Route pour debug
@app.get("/debug")
async def debug():
    return JSONResponse({
        "blob_urls": BLOB_FILE_URLS,
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0,
        "embeddings_shape": data_store.offers_emb.shape if data_store.offers_emb is not None else None
    })


