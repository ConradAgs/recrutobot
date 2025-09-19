import os
from vercel_blob import put
import asyncio
import requests

async def upload_to_vercel_blob():
    # Vos IDs Google Drive
    files = {
        "embedding.npy": "176y-qT1aYgry5m6hT2dRyEV4J-CcOlKj",
        "jobs_catalogue2.json": "1gzZCk3mtDXp8Y_siloYpCOJiJVCHY663"
    }
    
    for filename, file_id in files.items():
        print(f"📤 Upload de {filename}...")
        
        # Télécharger depuis Google Drive
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # Gérer la confirmation Google Drive
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                url = f"https://drive.google.com/uc?export=download&confirm={value}&id={file_id}"
                response = session.get(url, stream=True)
                break
        
        response.raise_for_status()
        file_data = response.content
        
        # Upload vers Vercel Blob
        blob = await put(filename, file_data, {
            'access': 'public',
            'addRandomSuffix': False
        })
        
        print(f"✅ {filename} uploadé: {blob.url}")
        print(f"📋 Path: {blob.pathname}")

if __name__ == "__main__":
    asyncio.run(upload_to_vercel_blob())