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
import time
from datetime import datetime

# =======================
# Configuration
# =======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URLs des fichiers Blob
BLOB_FILE_URLS = {
    "embeddings": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/embedding_compressed.npz",
    "offers": "https://a76pgx7uu8agygvt.public.blob.vercel-storage.com/jobs_catalogue2.json.gz"
}

HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
HF_MODEL = "sentence-transformers/all-mpnet-base-v2"

app = FastAPI(title="RecrutoBot")

try:
    templates = Jinja2Templates(directory="templates")
except:
    templates = None

# =======================
# Hugging Face Embeddings
# =======================
def get_embedding(text: str):
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
# DataStore optimisé
# =======================
class DataStore:
    def __init__(self):
        self.offers = []
        self.offers_emb = None
        self.data_loaded = False
        self._loading = False
        self._load_lock = asyncio.Lock()
        self.last_load_attempt = 0
        self.load_attempt_count = 0
        
    async def load_real_data_with_retry(self):
        """Charge les vraies données avec gestion d'erreurs et retry"""
        async with self._load_lock:
            if self.data_loaded or self._loading:
                return True
                
            # Éviter de spammer en cas d'erreur
            current_time = time.time()
            if current_time - self.last_load_attempt < 300:  # 5 minutes entre les tentatives
                logger.info("⏳ Tentative de chargement trop récente, attente...")
                return False
                
            self._loading = True
            self.last_load_attempt = current_time
            self.load_attempt_count += 1
            
            try:
                logger.info(f"🔄 Tentative de chargement #{self.load_attempt_count}...")
                
                # 1. Charger les embeddings
                logger.info("📥 Téléchargement des embeddings...")
                emb_response = requests.get(BLOB_FILE_URLS["embeddings"], timeout=120)
                
                if emb_response.status_code == 403:
                    logger.error("❌ Quota blob storage dépassé (403 Forbidden)")
                    raise HTTPException(status_code=503, detail="Quota de stockage dépassé. Réessayez demain.")
                elif emb_response.status_code != 200:
                    logger.error(f"❌ Erreur HTTP {emb_response.status_code} pour les embeddings")
                    return False
                    
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
                json_response = requests.get(BLOB_FILE_URLS["offers"], timeout=120)
                
                if json_response.status_code == 403:
                    logger.error("❌ Quota blob storage dépassé (403 Forbidden)")
                    raise HTTPException(status_code=503, detail="Quota de stockage dépassé. Réessayez demain.")
                elif json_response.status_code != 200:
                    logger.error(f"❌ Erreur HTTP {json_response.status_code} pour les offres")
                    return False
                    
                json_response.raise_for_status()
                
                # Décompresser
                self.offers = json.loads(gzip.decompress(json_response.content).decode('utf-8'))
                
                self.data_loaded = True
                logger.info(f"🎉 DONNÉES RÉELLES CHARGÉES: {len(self.offers):,} offres")
                logger.info(f"📊 Shape des embeddings: {self.offers_emb.shape}")
                
                return True
                
            except HTTPException:
                raise  # Relancer les HTTPException
            except Exception as e:
                logger.error(f"❌ Erreur lors du chargement: {e}")
                if "403" in str(e):
                    logger.error("🚫 ACCÈS BLOQUÉ - Quota dépassé")
                    raise HTTPException(
                        status_code=503, 
                        detail="Quota de stockage Vercel dépassé. L'application sera fonctionnelle après le reset quotidien (minuit UTC)."
                    )
                return False
            finally:
                self._loading = False

data_store = DataStore()

# =======================
# Chargement au démarrage
# =======================
@app.on_event("startup")
async def startup_event():
    """Tente de charger les données au démarrage"""
    logger.info("🚀 Démarrage - Tentative de chargement des données réelles...")
    await data_store.load_real_data_with_retry()

# =======================
# Routes FastAPI
# =======================
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        # Afficher la page même si les données ne sont pas chargées
        if templates:
            return templates.TemplateResponse("index.html", {
                "request": request,
                "data_loaded": data_store.data_loaded,
                "offers_count": len(data_store.offers) if data_store.data_loaded else 0
            })
        else:
            # Page HTML simple
            if data_store.data_loaded:
                status_html = f"""
                    <div style="background: #d4edda; color: #155724; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3>✅ Données chargées avec succès</h3>
                        <p><strong>{len(data_store.offers):,} offres</strong> disponibles pour la recherche</p>
                    </div>
                """
            else:
                status_html = f"""
                    <div style="background: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0;">
                        <h3>⏳ Données en cours de chargement</h3>
                        <p>Les données réelles sont temporairement indisponibles (quota de stockage dépassé).</p>
                        <p><em>L'application se réactivera automatiquement après le reset quotidien (minuit UTC).</em></p>
                        <button onclick="location.reload()">Réessayer</button>
                    </div>
                """
            
            return HTMLResponse(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>RecrutoBot - Données Réelles</title>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
                        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                        .search-box {{ margin: 30px 0; }}
                        input[type="text"] {{ 
                            width: 70%; 
                            padding: 12px; 
                            font-size: 16px; 
                            border: 2px solid #ddd; 
                            border-radius: 5px; 
                            margin-right: 10px;
                        }}
                        button {{ 
                            padding: 12px 25px; 
                            background: #3498db; 
                            color: white; 
                            border: none; 
                            border-radius: 5px; 
                            cursor: pointer; 
                            font-size: 16px;
                        }}
                        button:hover {{ background: #2980b9; }}
                        button:disabled {{ background: #95a5a6; cursor: not-allowed; }}
                        #results {{ margin-top: 20px; }}
                        .offer {{ 
                            border: 1px solid #e1e8ed; 
                            padding: 20px; 
                            margin: 15px 0; 
                            border-radius: 8px; 
                            background: #fafafa;
                        }}
                        .offer h4 {{ margin: 0 0 10px 0; color: #2c3e50; }}
                        .offer .score {{ 
                            background: #3498db; 
                            color: white; 
                            padding: 2px 8px; 
                            border-radius: 10px; 
                            font-size: 12px;
                            float: right;
                        }}
                        .loading {{ color: #7f8c8d; text-align: center; padding: 20px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>🔍 RecrutoBot - Recherche d'Offres Réelles</h1>
                        {status_html}
                        
                        <div class="search-box">
                            <input type="text" id="searchInput" placeholder="Ex: Développeur Python à Paris, Data Scientist, Ingénieur DevOps...">
                            <button onclick="search()" id="searchBtn">Rechercher</button>
                        </div>
                        
                        <div id="results"></div>
                    </div>

                    <script>
                        function updateButtonState() {{
                            const btn = document.getElementById('searchBtn');
                            btn.disabled = {str(not data_store.data_loaded).lower()};
                            if (btn.disabled) {{
                                btn.innerHTML = 'Données en chargement...';
                            }} else {{
                                btn.innerHTML = 'Rechercher';
                            }}
                        }}

                        async function search() {{
                            const prompt = document.getElementById('searchInput').value;
                            if (!prompt) return;
                            
                            const resultsDiv = document.getElementById('results');
                            resultsDiv.innerHTML = '<div class="loading">🔍 Recherche en cours...</div>';
                            
                            try {{
                                const response = await fetch('/api/search', {{
                                    method: 'POST',
                                    headers: {{'Content-Type': 'application/json'}},
                                    body: JSON.stringify({{prompt: prompt}})
                                }});
                                
                                if (response.status === 503) {{
                                    resultsDiv.innerHTML = `
                                        <div style="background: #fff3cd; color: #856404; padding: 20px; border-radius: 5px;">
                                            <h3>⏳ Service temporairement indisponible</h3>
                                            <p>Les données réelles sont en cours de chargement. Réessayez dans quelques minutes.</p>
                                            <p><em>Quota de stockage quotidien: reset à minuit UTC</em></p>
                                        </div>
                                    `;
                                    return;
                                }}
                                
                                const data = await response.json();
                                displayResults(data);
                            }} catch (error) {{
                                resultsDiv.innerHTML = '<div style="color: red;">Erreur lors de la recherche</div>';
                                console.error('Error:', error);
                            }}
                        }}
                        
                        function displayResults(data) {{
                            const resultsDiv = document.getElementById('results');
                            
                            if (data.results.length === 0) {{
                                resultsDiv.innerHTML = `
                                    <div style="text-align: center; padding: 40px; color: #7f8c8d;">
                                        <h3>🔍 Aucune offre trouvée</h3>
                                        <p>Essayez avec d'autres termes de recherche.</p>
                                        <p>Exemples: "Développeur web", "Data analyst", "Ingénieur cloud"</p>
                                    </div>
                                `;
                                return;
                            }}
                            
                            let html = `<h3>🔍 ${{data.message}}</h3>`;
                            data.results.forEach(offer => {{
                                const scorePercent = (offer.score * 100).toFixed(1);
                                html += `
                                    <div class="offer">
                                        <span class="score">${{scorePercent}}%</span>
                                        <h4>${{offer.intitule}}</h4>
                                        <p>${{offer.description}}</p>
                                        <p><strong>📍 Lieu:</strong> ${{offer.lieuTravail.libelle || 'Non spécifié'}}</p>
                                        <p><strong>📄 Contrat:</strong> ${{offer.typeContratLibelle || offer.typeContrat || 'Non spécifié'}}</p>
                                        <p><strong>🏢 Entreprise:</strong> ${{offer.entreprise.nom || 'Non spécifiée'}}</p>
                                        <p><strong>💼 Expérience:</strong> ${{offer.experienceLibelle || 'Non spécifiée'}}</p>
                                    </div>
                                `;
                            }});
                            resultsDiv.innerHTML = html;
                        }}
                        
                        // Initialiser l'état du bouton
                        updateButtonState();
                        
                        // Permettre la recherche avec Enter
                        document.getElementById('searchInput').addEventListener('keypress', function(e) {{
                            if (e.key === 'Enter') {{
                                search();
                            }}
                        }});
                    </script>
                </body>
                </html>
            """)
            
    except Exception as e:
        logger.error(f"Erreur read_root: {e}")
        return HTMLResponse(f"""
            <html>
            <body style="font-family: Arial; padding: 20px;">
                <h1>❌ Erreur technique</h1>
                <p>{str(e)}</p>
                <button onclick="location.reload()">Recharger</button>
            </body>
            </html>
        """)

@app.post("/api/search")
async def search_offers(request: Request):
    """Recherche dans les offres réelles uniquement"""
    try:
        # Vérifier que les données sont chargées
        if not data_store.data_loaded:
            # Tenter de recharger une fois
            success = await data_store.load_real_data_with_retry()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Les données réelles ne sont pas encore disponibles. Réessayez dans quelques minutes."
                )

        data = await request.json()
        prompt = data.get("prompt", "")
        if not prompt:
            raise HTTPException(status_code=400, detail="Veuillez saisir une recherche")

        # Générer l'embedding de la requête
        query_emb = get_embedding(prompt)

        # Calculer la similarité cosinus
        norms = np.linalg.norm(data_store.offers_emb, axis=1)
        query_norm = np.linalg.norm(query_emb)
        cos_scores = np.dot(data_store.offers_emb, query_emb) / (norms * query_norm)

        # Sélectionner les offres avec un score > 0.3
        good_indices = np.where(cos_scores > 0.3)[0]

        if len(good_indices) == 0:
            return JSONResponse({
                "results": [],
                "message": f"Aucune offre ne correspond à '{prompt}'. Essayez avec d'autres termes.",
                "count": 0,
                "search_term": prompt
            })

        # Préparer les résultats
        results = []
        for i in good_indices:
            offer = data_store.offers[i]
            results.append({
                "id": offer.get("id", f"offre-{i}"),
                "intitule": offer.get("intitule", "Titre non disponible"),
                "description": offer.get("description", "Description non disponible")[:300] + "...",
                "lieuTravail": offer.get("lieuTravail", {}),
                "typeContrat": offer.get("typeContrat", ""),
                "typeContratLibelle": offer.get("typeContratLibelle", ""),
                "experienceLibelle": offer.get("experienceLibelle", ""),
                "salaire": offer.get("salaire", {}),
                "entreprise": offer.get("entreprise", {}),
                "origineOffre": offer.get("origineOffre", {}),
                "score": float(cos_scores[i])
            })

        # Trier par score décroissant
        results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"🔍 Recherche '{prompt}': {len(results)} résultats sur {len(data_store.offers):,} offres")
        
        return JSONResponse({
            "results": results[:25],  # Limiter à 25 résultats
            "message": f"{len(results):,} offres trouvées pour '{prompt}'",
            "count": len(results),
            "search_term": prompt,
            "total_offers": len(data_store.offers)
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@app.get("/health")
async def health_check():
    """Endpoint de santé"""
    return JSONResponse({
        "status": "ok" if data_store.data_loaded else "waiting",
        "data_loaded": data_store.data_loaded,
        "offers_count": len(data_store.offers) if data_store.data_loaded else 0,
        "last_load_attempt": data_store.last_load_attempt,
        "load_attempt_count": data_store.load_attempt_count
    })

@app.get("/reload")
async def reload_data():
    """Forcer le rechargement des données"""
    data_store.data_loaded = False
    success = await data_store.load_real_data_with_retry()
    return JSONResponse({
        "success": success,
        "offers_count": len(data_store.offers) if success else 0,
        "message": "Données rechargées avec succès" if success else "Échec du rechargement"
    })

@app.get("/favicon.ico")
async def favicon():
    return JSONResponse({"status": "no favicon"})

@app.get("/stats")
async def get_stats():
    """Statistiques des données"""
    if not data_store.data_loaded:
        raise HTTPException(status_code=503, detail="Données non chargées")
    
    return JSONResponse({
        "total_offers": len(data_store.offers),
        "embeddings_shape": data_store.offers_emb.shape if data_store.offers_emb is not None else None,
        "data_loaded": data_store.data_loaded,
        "load_attempt_count": data_store.load_attempt_count
    })
