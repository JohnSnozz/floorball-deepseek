# scrape_reglement.py
import requests
from bs4 import BeautifulSoup
import html2text
import json
import os

def scrape_unihockey_reglement():
    """
    Scrape das Unihockey-Reglement von swiss unihockey
    """
    url = "https://swissunihockey.tlex.ch/app/de/texts_of_law/3-4"
    
    # Headers für besseres Scraping
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Text extrahieren
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        
        # Hauptinhalt finden (kann je nach Website-Struktur angepasst werden)
        content_div = soup.find('div', class_='content') or soup.find('main') or soup
        
        if content_div:
            text_content = h.handle(str(content_div))
        else:
            text_content = h.handle(response.text)
        
        # Text aufbereiten
        lines = text_content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Nur längere Zeilen behalten
                cleaned_lines.append(line)
        
        reglement_text = '\n\n'.join(cleaned_lines)
        
        # In Datei speichern
        os.makedirs('data', exist_ok=True)
        with open('data/unihockey_reglement.txt', 'w', encoding='utf-8') as f:
            f.write(reglement_text)
        
        print(f"✓ Reglement gespeichert: {len(reglement_text)} Zeichen")
        return reglement_text
        
    except Exception as e:
        print(f"❌ Fehler beim Scraping: {e}")
        return None

if __name__ == "__main__":
    scrape_unihockey_reglement()