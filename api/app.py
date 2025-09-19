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

# =======================
# Configuration du logging
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =======================
# Configuration des fichiers Google Drive
# =======================
files = {
    "embedding.npy": "176y-qT1aYgry5m6hT2dRyEV4J-CcOlKj",
    "jobs_catalogue2.json": "1gzZCk3mtDXp8Y_siloYpCOJiJVCHY663"
}

# =======================
# Clé et modèle Hugging Face
# =======================
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "sentence-transformers/all-mpnet-base-v2"

# =======================
# Application FastAPI
# =======================
app = FastAPI(title="RecrutoBot", description="Version sur Vercel")

templates_path = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))

# =======================
# Fonctions utilitaires
# =======================
def import_json(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def download_files():
    """Télécharge les fichiers depuis Google Drive"""
    try:
        for filename, file_id in files.items():
            file_path = Path(filename)
            if not file_path.exists():
                try:
                    logger.info(f"Téléchargement de {filename}...")
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    session = requests.Session()
                    response = session.get(url, stream=True)
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                            response = session.get(url, stream=True)
                            break
                    response.raise_for_status()
                    with open(filename, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    logger.info(f"{filename} téléchargé avec succès")
                except Exception as e:
                    logger.error(f"❌ Erreur avec {filename}: {e}")
                    return False
        return True
    except Exception as e:
        logger.error(f"❌ Erreur lors du téléchargement: {e}")
        return False

# =======================
# Hugging Face embeddings
# =======================
def get_embedding(prompt: str):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        emb = np.array(response.json(), dtype=np.float32)
        if emb.ndim == 2:
            emb = emb[0]
        return emb
    except Exception as e:
        logger.error(f"❌ Erreur Hugging Face: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération d'embeddings: {e}")

# =======================
# DataStore
# =======================
class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return True
        try:
            if not download_files():
                logger.error("❌ Impossible de télécharger les fichiers nécessaires")
                return False

            logger.info("📥 Chargement des embeddings...")
            self.offers_emb = np.load("embedding.npy", allow_pickle=True).astype(np.float32)

            logger.info("📋 Chargement des offres d'emploi...")
            self.offers = import_json("jobs_catalogue2.json")

            self.data_loaded = True
            logger.info(f"📈 {len(self.offers)} offres chargées")
            return True

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des données: {e}")
            logger.error(traceback.format_exc())
            return False

data_store = DataStore()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if not data_store.data_loaded:
        data_store.load_data()
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search_offers(request: Request):
    try:
        if not data_store.data_loaded and not data_store.load_data():
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
