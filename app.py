from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import json
import numpy as np
import os
import traceback
import logging
import requests
from huggingface_hub import InferenceClient
import asyncio

# =======================
# Configuration
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "sentence-transformers/all-mpnet-base-v2"

app = FastAPI(title="RecrutoBot", description="Version d'urgence")

# Configuration des templates
try:
    templates = Jinja2Templates(directory="templates")
except:
    templates = None

# =======================
# Hugging Face Embeddings
# =======================
def get_embedding(text: str):
    """Génère un embedding avec Hugging Face"""
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Token Hugging Face non configuré")
    
    try:
        client = InferenceClient(token=HF_API_TOKEN)
        embeddings = client.feature_extraction(text, model=HF_MODEL)
        emb = np.array(embeddings, dtype=np.float32)
        
        if emb.ndim == 2:
            emb = emb[0]
            
        logger.info(f"✅ Embedding généré - Shape: {emb.shape}")
        return emb
        
    except Exception as e:
        logger.error(f"❌ Erreur Hugging Face: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur génération embedding: {e}")

# =======================
# DataStore minimaliste
# =======================
class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False

    async def load_sample_data(self):
        """Charge des données d'exemple sans blob storage"""
        try:
            logger.info("📥 Chargement des données d'exemple...")
            
            # Créer des données d'exemple minimales
            self.offers = [
                {
                    "id": "1",
                    "intitule": "Développeur Python",
                    "description": "Développement d'applications web avec Python et FastAPI",
                    "lieuTravail": {"libelle": "Paris"},
                    "typeContrat": "CDI",
                    "typeContratLibelle": "CDI",
                    "experienceLibelle": "Débutant accepté",
                    "salaire": {"libelle": "35-40k"},
                    "entreprise": {"nom": "TechCorp"},
                    "origineOffre": {"url": "#"}
                },
                {
                    "id": "2", 
                    "intitule": "Data Scientist",
                    "description": "Analyse de données et machine learning",
                    "lieuTravail": {"libelle": "Lyon"},
                    "typeContrat": "CDI",
                    "typeContratLibelle": "CDI", 
                    "experienceLibelle": "Expérimenté",
                    "salaire": {"libelle": "45-50k"},
                    "entreprise": {"nom": "DataCompany"},
                    "origineOffre": {"url": "#"}
                }
            ]
            
            # Créer des embeddings factices pour la démo
            self.offers_emb = np.random.randn(len(self.offers), 768).astype(np.float32)
            
            self.data_loaded = True
            logger.info(f"✅ {len(self.offers)} offres d'exemple chargées")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur load_sample_data: {e}")
            return False

data_store = DataStore()

# =======================
# Chargement au démarrage
# =======================
@app.on_event("startup")
async def startup_event():
    """Charge les données d'exemple au démarrage"""
    logger.info("🚀 Démarrage - Chargement données d'exemple...")
    await data_store.load_sample_data()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        if not data_store.data_loaded:
            await data_store.load_sample_data()
            
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        else:
            return HTMLResponse("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>RecrutoBot</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .container { max-width: 800px; margin: 0 auto; }
                        .search-box { margin: 20px 0; }
                        input[type="text"] { width: 300px; padding: 10px; }
                        button { padding: 10px 20px; background: #007acc; color: white; border: none; cursor: pointer; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🔍 RecrutoBot</h1>
                        <p><strong>Mode démonstration</strong> - Recherche d'offres d'emploi</p>
                        
                        <div class="search-box">
                            <input type="text" id="searchInput" placeholder="Ex: Développeur Python à Paris...">
                            <button onclick="search()">Rechercher</button>
                        </div>
                        
                        <div id="results"></div>
                        
                        <script>
                            async function search() {
                                const prompt = document.getElementById('searchInput').value;
                                if (!prompt) return;
                                
                                const response = await fetch('/api/search', {
                                    method: 'POST',
                                    headers: {'Content-Type': 'application/json'},
                                    body: JSON.stringify({prompt: prompt})
                                });
                                
                                const data = await response.json();
                                displayResults(data);
                            }
                            
                            function displayResults(data) {
                                const resultsDiv = document.getElementById('results');
                                if (data.results.length === 0) {
                                    resultsDiv.innerHTML = '<p>Aucune offre trouvée. Essayez d\'autres termes.</p>';
                                    return;
                                }
                                
                                let html = `<h3>${data.message}</h3>`;
                                data.results.forEach(offer => {
                                    html += `
                                        <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px;">
                                            <h4>${offer.intitule}</h4>
                                            <p>${offer.description}</p>
                                            <p><strong>Lieu:</strong> ${offer.lieuTravail.libelle}</p>
                                            <p><strong>Contrat:</strong> ${offer.typeContratLibelle}</p>
                                            <p><strong>Entreprise:</strong> ${offer.entreprise.nom}</p>
                                            <p><strong>Score:</strong> ${(offer.score * 100).toFixed(1)}%</p>
                                        </div>
                                    `;
                                });
                                resultsDiv.innerHTML = html;
                            }
                        </script>
                    </div>
                </body>
                </html>
            """)
            
    except Exception as e:
        logger.error(f"Erreur read_root: {e}")
        return HTMLResponse(f"""
            <html><body>
                <h1>RecrutoBot</h1>
                <p>Erreur: {str(e)}</p>
            </body></html>
        """)

@app.post("/api/search")
async def search_offers(request: Request):
    try:
        if not data_store.data_loaded:
            await data_store.load_sample_data()

        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt vide")

        query_emb = get_embedding(prompt)

        if data_store.offers_emb is None:
            raise HTTPException(status_code=500, detail="Embeddings manquants")

        # Similarité cosinus avec données d'exemple
        norms = np.linalg.norm(data_store.offers_emb, axis=1)
        query_norm = np.linalg.norm(query_emb)
        cos_scores = np.dot(data_store.offers_emb, query_emb) / (norms * query_norm)

        good_indices = np.where(cos_scores > 0.1)[0]  # Seuil plus bas pour la démo

        results = []
        for i in good_indices:
            offer = data_store.offers[i]
            results.append({
                "id": offer.get("id", ""),
                "intitule": offer.get("intitule", "Titre non disponible"),
                "description": offer.get("description", "Description non disponible"),
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
            "results": results,
            "message": f"{len(results)} offres trouvées pour '{prompt}'" if results else "Aucune offre exacte trouvée. Voici les offres disponibles:",
            "count": len(results),
            "search_term": prompt,
            "demo_mode": True
        })

    except Exception as e:
        logger.error(f"Erreur search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "ok", 
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0,
        "demo_mode": True
    })

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({"status": "no favicon"})
