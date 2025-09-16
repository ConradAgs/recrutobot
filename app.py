from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import traceback
import uvicorn
import logging
from huggingface_hub import hf_hub_download
import asyncio
from concurrent.futures import ThreadPoolExecutor

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

# =======================
# Initialisation des données
# =======================
class DataStore:
    def __init__(self):
        self.model = None
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False
        self.data_loading = False

    async def load_data(self):
        """Charge les données de façon asynchrone"""
        if self.data_loaded or self.data_loading:
            return True
            
        self.data_loading = True
        logger.info("📥 Début du téléchargement depuis Hugging Face...")
        
        try:
            # Télécharger les fichiers en parallèle avec ThreadPoolExecutor
            with ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                
                # Tâches pour télécharger les fichiers
                embedding_task = loop.run_in_executor(
                    executor, 
                    lambda: hf_hub_download(
                        repo_id="ConradAgs/recrutobot-data",
                        filename="embedding.npy",
                        token=HF_TOKEN,
                        repo_type="dataset"
                    )
                )
                
                offers_task = loop.run_in_executor(
                    executor,
                    lambda: hf_hub_download(
                        repo_id="ConradAgs/recrutobot-data",
                        filename="jobs_catalogue2.json",
                        token=HF_TOKEN,
                        repo_type="dataset"
                    )
                )
                
                # Attendre que les deux téléchargements soient terminés
                embedding_path, offers_path = await asyncio.gather(embedding_task, offers_task)
    
            logger.info("Chargement des embeddings...")
            embedding = np.load(embedding_path, allow_pickle=True)
            self.offers_emb = torch.tensor(embedding.astype(np.float32))
    
            logger.info("🤖 Chargement du modèle...")
            self.model = SentenceTransformer("all-mpnet-base-v2", device="cpu")
    
            logger.info("📋 Chargement des offres d'emploi...")
            with open(offers_path, "r", encoding="utf-8") as f:
                self.offers = json.load(f)
    
            self.data_loaded = True
            self.data_loading = False
            logger.info(f"📈 {len(self.offers)} offres chargées")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement: {str(e)}")
            logger.error(traceback.format_exc())
            self.data_loading = False
            return False

# Instance globale pour stocker les données
data_store = DataStore()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Page d'accueil avec l'interface de recherche"""
    # Démarrer le chargement des données en arrière-plan si ce n'est pas déjà fait
    if not data_store.data_loaded and not data_store.data_loading:
        asyncio.create_task(data_store.load_data())
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search_offers(request: Request):
    """Endpoint pour effectuer une recherche d'offres"""
    try:
        # Vérifier si les données sont en cours de chargement
        if data_store.data_loading:
            return JSONResponse({
                "results": [],
                "message": "⚠️ Les données sont en cours de chargement, veuillez réessayer dans quelques secondes...",
                "count": 0,
                "search_term": ""
            })
        
        # Charger les données si nécessaire
        if not data_store.data_loaded:
            success = await data_store.load_data()
            if not success:
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
        good_indices = [i for i, score in enumerate(cos_scores) if score > 0.3]

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
        "data_loading": data_store.data_loading,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    })

@app.get("/status")
async def status():
    """Endpoint pour vérifier l'état du chargement"""
    return {
        "data_loaded": data_store.data_loaded,
        "data_loading": data_store.data_loading,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0
    }

@app.head("/")
async def head_root():
    return {}

# =======================
# Point d'entrée pour Render
# =======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
