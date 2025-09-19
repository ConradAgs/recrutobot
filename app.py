# app.py - Version finale avec Vercel Blob
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

# =======================
# Configuration du logging
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# Application FastAPI
# =======================
app = FastAPI(title="RecrutoBot", description="Avec Vercel Blob Storage")

templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# =======================
# DataStore avec Vercel Blob
# =======================
# === CONFIGURATION BLOB STORE ===
BLOB_FILE_URLS = {
    "embeddings": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/embedding_compressed.npz",
    "offers": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/jobs_catalogue2.json.gz"
}

class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False

    async def load_data(self):
        if self.data_loaded:
            return True
        
        try:
            logger.info("📥 Chargement depuis Vercel Blob Store...")
            
            # 1. Charger les embeddings compressés
            logger.info("🧠 Chargement des embeddings...")
            emb_response = requests.get(
                BLOB_FILE_URLS["embeddings"],
                timeout=120  # Timeout plus long pour gros fichier
            )
            emb_response.raise_for_status()
            logger.info("✅ Embeddings téléchargés")
            
            # Sauvegarder temporairement et charger
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
                f.write(emb_response.content)
                tmp_path = f.name
            
            # Charger le fichier NPZ
            data = np.load(tmp_path)
            self.offers_emb = data['embeddings'].astype(np.float32)
            os.unlink(tmp_path)
            logger.info(f"✅ Embeddings chargés: shape {self.offers_emb.shape}")
            
            # 2. Charger le JSON compressé
            logger.info("📋 Chargement des offres...")
            json_response = requests.get(
                BLOB_FILE_URLS["offers"],
                timeout=60
            )
            json_response.raise_for_status()
            logger.info("✅ JSON téléchargé")
            
            # Décompresser le JSON
            self.offers = json.loads(gzip.decompress(json_response.content).decode('utf-8'))
            logger.info(f"✅ {len(self.offers)} offres chargées")
            
            self.data_loaded = True
            logger.info("🎉 Toutes les données sont chargées!")
            return True
            
        except requests.exceptions.Timeout:
            logger.error("❌ Timeout lors du téléchargement des fichiers")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ Erreur HTTP: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            logger.error(traceback.format_exc())
            return False

# =======================
# Routes FastAPI (inchangées)
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if not data_store.data_loaded:
        success = await data_store.load_data()
        if not success:
            return templates.TemplateResponse("error.html", {
                "request": request,
                "message": "Impossible de charger les données. Veuillez réessayer."
            })
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search_offers(request: Request):
    try:
        if not data_store.data_loaded:
            success = await data_store.load_data()
            if not success:
                raise HTTPException(status_code=500, detail="Erreur lors du chargement des données")

        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Requête vide")

        query_emb = get_embedding(prompt)

        if data_store.offers_emb is None:
            raise HTTPException(status_code=500, detail="Embeddings non chargés")

        # Calcul des similitudes cosinus
        cos_scores = np.dot(data_store.offers_emb, query_emb) / (
            np.linalg.norm(data_store.offers_emb, axis=1) * np.linalg.norm(query_emb)
        )

        good_indices = [i for i, score in enumerate(cos_scores) if score > 0.3]

        if not good_indices:
            return JSONResponse({
                "results": [],
                "message": "Je n'ai trouvé aucune offre correspondante. Pouvez-vous reformuler ?",
                "count": 0,
                "search_term": prompt
            })

        results = []
        for i in good_indices:
            score = float(cos_scores[i])
            offer = data_store.offers[i]
            results.append({
                "id": offer.get("id"),
                "intitule": offer.get("intitule") or "Titre non disponible",
                "description": offer.get("description") or "Description non disponible",
                "lieuTravail": offer.get("lieuTravail"),
                "typeContrat": offer.get("typeContrat"),
                "typeContratLibelle": offer.get("typeContratLibelle"),
                "experienceLibelle": offer.get("experienceLibelle"),
                "salaire": offer.get("salaire"),
                "entreprise": offer.get("entreprise"),
                "origineOffre": offer.get("origineOffre"),
                "score": score
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return JSONResponse({
            "results": results,
            "message": f"J'ai trouvé {len(results)} offres correspondant à '{prompt}'",
            "count": len(results),
            "search_term": prompt
        })

    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok",
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    })

# =======================
# Page d'erreur simple
# =======================
@app.get("/error")
async def error_page(request: Request):
    return templates.TemplateResponse("error.html", {"request": request})
