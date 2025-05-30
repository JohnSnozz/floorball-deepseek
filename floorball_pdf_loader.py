# pdf_loader.py - PDF Reglement Loader
import os
from pathlib import Path
import pypdf

def load_reglement_from_pdf(pdf_path="data/unihockey_reglement.pdf"):
    """
    Lade Reglement aus PDF-Datei
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ PDF nicht gefunden: {pdf_path}")
        print("📥 Bitte lade das Reglement-PDF in den data/ Ordner")
        return None
    
    try:
        # PDF lesen
        with open(pdf_file, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            # Text aus allen Seiten extrahieren
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                full_text += f"\n\n--- Seite {page_num + 1} ---\n\n{text}"
            
            # Text aufräumen
            lines = full_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:  # Nur längere Zeilen
                    cleaned_lines.append(line)
            
            reglement_text = '\n\n'.join(cleaned_lines)
            
            # Als TXT speichern
            txt_path = "data/unihockey_reglement.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(reglement_text)
            
            print(f"✅ PDF geladen: {len(pdf_reader.pages)} Seiten")
            print(f"✅ Text gespeichert: {len(reglement_text)} Zeichen")
            
            return reglement_text
            
    except Exception as e:
        print(f"❌ Fehler beim PDF-Laden: {e}")
        return None

if __name__ == "__main__":
    load_reglement_from_pdf()