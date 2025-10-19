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
import asyncio

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
# DataStore amélioré avec cache
# =======================
class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False
        self._loading = False
        self._load_lock = asyncio.Lock()

    async def load_data(self):
        # Éviter les chargements multiples simultanés
        async with self._load_lock:
            if self.data_loaded or self._loading:
                return True
            
            self._loading = True
            try:
                logger.info("📥 Chargement depuis Vercel Blob Store...")
                
                # 1. Charger les embeddings
                logger.info("🧠 Téléchargement des embeddings...")
                emb_response = requests.get(BLOB_FILE_URLS["embeddings"], timeout=120)
                emb_response.raise_for_status()
                
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
                    f.write(emb_response.content)
                    tmp_path = f.name
                
                # Charger le NPZ
                data = np.load(tmp_path)
                self.offers_emb = data['embeddings'].astype(np.float32)
                os.unlink(tmp_path)
                
                # 2. Charger les offres
                logger.info("📋 Téléchargement des offres...")
                json_response = requests.get(BLOB_FILE_URLS["offers"], timeout=60)
                json_response.raise_for_status()
                
                # Décompresser
                self.offers = json.loads(gzip.decompress(json_response.content).decode('utf-8'))
                
                self.data_loaded = True
                logger.info(f"✅ {len(self.offers)} offres chargées")
                return True
                
            except Exception as e:
                logger.error(f"❌ Erreur load_data: {e}")
                logger.error(traceback.format_exc())
                self._loading = False
                return False
            finally:
                self._loading = False

data_store = DataStore()

# =======================
# Chargement au démarrage
# =======================
@app.on_event("startup")
async def startup_event():
    """Charge les données au démarrage de l'application"""
    logger.info("🚀 Démarrage de l'application - Chargement des données...")
    await data_store.load_data()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        # Vérifier si les données sont chargées, sinon essayer de charger
        if not data_store.data_loaded:
            success = await data_store.load_data()
            if not success:
                return templates.TemplateResponse("error.html", {
                    "request": request,
                    "error": "Données temporairement indisponibles. Réessayez dans quelques instants."
                })
        
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Erreur read_root: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": f"Erreur technique: {str(e)}"
        })

@app.post("/api/search")
async def search_offers(request: Request):
    try:
        # NE PAS recharger les données à chaque requête
        if not data_store.data_loaded:
            # Si les données ne sont pas chargées, essayer une fois
            success = await data_store.load_data()
            if not success:
                raise HTTPException(status_code=503, detail="Service temporairement indisponible")

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
            "results": results[:20],
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

@app.get("/reload-data")
async def reload_data():
    """Endpoint manuel pour recharger les données si nécessaire"""
    success = await data_store.load_data()
    return JSONResponse({
        "success": success,
        "message": "Rechargement des données effectué" if success else "Erreur lors du rechargement",
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    })

@app.get("/debug")
async def debug():
    return JSONResponse({
        "blob_urls": BLOB_FILE_URLS,
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0,
        "embeddings_shape": data_store.offers_emb.shape if data_store.offers_emb is not None else None
    })
