from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import gdown
import traceback
from typing import List, Dict, Any
import uvicorn
import logging
from huggingface_hub import hf_hub_download
import os

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# Configuration des fichiers
# =======================


HF_TOKEN = os.getenv("HF_TOKEN")

# =======================
# Application FastAPI
# =======================
app = FastAPI(title="RecrutoBot", description="Version 100% locale sans API externe")

# Créer les répertoires nécessaires
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =======================
# Fonctions utilitaires
# =======================
def import_json(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def download_files():
    """Télécharge les fichiers depuis Google Drive si absents"""
    for filename, file_id in files.items():
        if not os.path.exists(filename):
            try:
                logger.info(f"Téléchargement de {filename}...")
                url = f"https://drive.google.com/uc?id={file_id}"
                output = gdown.download(url, filename, quiet=False)

                if not output or not os.path.exists(filename) or os.path.getsize(filename) < 1000:
                    raise RuntimeError(f"{filename} n'a pas été téléchargé correctement")

                logger.info(f"{filename} téléchargé avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur avec {filename}: {e}")
                return False
    return True


# =======================
# Initialisation des données
# =======================
class DataStore:
    def __init__(self):
        self.model = None
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return True

        try:
            logger.info("📥 Téléchargement depuis Hugging Face...")

            embedding_path = hf_hub_download(
                repo_id="ConradAgs/recrutobot-data",
                filename="embedding.npy",
                token=HF_TOKEN
            )
            offers_path = hf_hub_download(
                repo_id="ConradAgs/recrutobot-data",
                filename="jobs_catalogue2.json",
                token=HF_TOKEN
            )

            logger.info("Chargement des embeddings...")
            embedding = np.load(embedding_path, allow_pickle=True)
            self.offers_emb = torch.tensor(embedding.astype(np.float32))

            logger.info("🤖 Chargement du modèle...")
            self.model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

            logger.info("📋 Chargement des offres d'emploi...")
            self.offers = import_json(offers_path)

            self.data_loaded = True
            logger.info(f"📈 {len(self.offers)} offres chargées")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {e}")
            logger.error(traceback.format_exc())
            return False

# Instance globale pour stocker les données
data_store = DataStore()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Page d'accueil avec l'interface de recherche"""
    # Charger les données au premier accès
    if not data_store.data_loaded:
        data_store.load_data()

    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search_offers(request: Request):
    """Endpoint pour effectuer une recherche d'offres"""
    try:
        # Charger les données si nécessaire
        if not data_store.data_loaded:
            if not data_store.load_data():
                raise HTTPException(status_code=500, detail="Erreur lors du chargement des données")

        # Récupérer la requête de recherche
        data = await request.json()
        prompt = data.get("prompt", "")

        if not prompt:
            raise HTTPException(status_code=400, detail="Requête de recherche vide")

        # Encoder la requête
        query_emb = data_store.model.encode(prompt, convert_to_tensor=True)

        # Vérifier que les embeddings sont chargés
        if data_store.offers_emb is None:
            raise HTTPException(status_code=500, detail="Embeddings non chargés")

        cos_scores = util.cos_sim(query_emb, data_store.offers_emb)[0]
        good_indices = [i for i, score in enumerate(cos_scores) if score > 0.3]  # Seuil réduit à 0.3

        if not good_indices:
            return JSONResponse({
                "results": [],
                "message": "Je n'ai trouvé aucune offre correspondante. Pouvez-vous reformuler ?",
                "count": 0,
                "search_term": prompt
            })

        # Préparer les résultats
        results = []
        for i in good_indices:
            score = cos_scores[i]
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
                "score": float(score)
            })

        # Trier par score
        results.sort(key=lambda x: x["score"], reverse=True)

        return JSONResponse({
            "results": results,
            "message": f"J'ai trouvé {len(results)} offres correspondant à '{prompt}'",
            "count": len(results),
            "search_term": prompt
        })

    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint pour vérifier l'état de l'application"""
    return JSONResponse({
        "status": "ok",
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    })

# =======================
# Point d'entrée
# =======================
if __name__ == "__main__":
    # Charger les données au démarrage
    data_store.load_data()

    # Démarrer le serveur
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
