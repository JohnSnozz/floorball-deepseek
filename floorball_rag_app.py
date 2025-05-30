# app.py - Chainlit RAG App für Unihockey-Reglement
import os
import asyncio
import chainlit as cl
from pathlib import Path

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Ollama konfigurieren
Settings.llm = Ollama(
    model="deepseek-r1:7b", 
    base_url="http://localhost:11434",
    request_timeout=120.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Globale Variablen
INDEX = None
QUERY_ENGINE = None

async def load_documents():
    """Lade das Unihockey-Reglement"""
    
    reglement_path = Path("data/unihockey_reglement.txt")
    pdf_path = Path("data/unihockey_reglement.pdf")
    
    # Prüfe verschiedene Quellen
    if not reglement_path.exists():
        # 1. Versuche PDF zu laden
        if pdf_path.exists():
            print("📄 Lade Reglement aus PDF...")
            from pdf_loader import load_reglement_from_pdf
            content = load_reglement_from_pdf()
        # 2. Versuche Web-Scraping
        else:
            print("🌐 Versuche Web-Scraping...")
            from scrape_reglement import scrape_unihockey_reglement
            content = scrape_unihockey_reglement()
            
        if not content:
            raise Exception("❌ Kein Reglement gefunden! Bitte PDF in data/ Ordner legen.")
    
    # Dokument laden
    if reglement_path.exists():
        with open(reglement_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Document erstellen
        document = Document(
            text=content,
            metadata={
                "source": "Swiss Unihockey Reglement",
                "type": "regulation",
                "url": "https://swissunihockey.tlex.ch/app/de/texts_of_law/3-4"
            }
        )
        return [document]
    
    return []

async def create_index():
    """Erstelle oder lade Vector Index"""
    global INDEX, QUERY_ENGINE
    
    storage_dir = "storage"
    
    # Prüfe ob Index bereits existiert
    if os.path.exists(storage_dir):
        try:
            print("📂 Lade bestehenden Index...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            INDEX = load_index_from_storage(storage_context)
            print("✅ Index geladen")
        except:
            INDEX = None
    
    # Erstelle neuen Index falls notwendig
    if INDEX is None:
        print("🔄 Erstelle neuen Index...")
        
        # Dokumente laden
        documents = await load_documents()
        
        if not documents:
            raise Exception("Keine Dokumente gefunden!")
        
        # Text splitter
        node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50
        )
        
        # Index erstellen
        INDEX = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser,
            show_progress=True
        )
        
        # Index speichern
        INDEX.storage_context.persist(persist_dir=storage_dir)
        print("✅ Index erstellt und gespeichert")
    
    # Query Engine erstellen
    QUERY_ENGINE = INDEX.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )
    
    print("🚀 RAG System bereit!")

@cl.on_chat_start
async def start():
    """Initialisierung beim Chat-Start"""
    
    # Zeige Loading Message
    msg = cl.Message(content="🏒 Lade Unihockey-Reglement...")
    await msg.send()
    
    try:
        # Index laden/erstellen
        await create_index()
        
        # Success Message
        msg.content = "✅ Unihockey RAG System bereit!\n\n🤖 Stelle mir Fragen zum Schweizer Unihockey-Reglement:"
        await msg.update()
        
        # Beispielfragen zeigen
        await cl.Message(
            content="""💡 **Beispielfragen:**
            
📏 Wie groß ist das Spielfeld?
👥 Wie viele Spieler sind auf dem Feld?
⏱️ Wie lange dauert ein Spiel?
🚨 Was sind die Strafenarten?"""
        ).send()
        
    except Exception as e:
        msg.content = f"❌ Fehler beim Laden: {e}"
        await msg.update()

# Entferne Action Callbacks da wir keine Actions mehr haben

async def process_question(question: str):
    """Verarbeite eine Frage"""
    global QUERY_ENGINE
    
    if not QUERY_ENGINE:
        await cl.Message(content="❌ RAG System nicht bereit!").send()
        return
    
    # Loading message
    msg = cl.Message(content="🤔 Suche in den Regeln...")
    await msg.send()
    
    try:
        # Query ausführen
        response = await asyncio.to_thread(QUERY_ENGINE.query, question)
        
        # Antwort formatieren
        answer = f"**🏒 Antwort zum Unihockey-Reglement:**\n\n{response.response}"
        
        # Quellen hinzufügen falls verfügbar
        if hasattr(response, 'source_nodes') and response.source_nodes:
            answer += "\n\n**📚 Quellen:**\n"
            for i, node in enumerate(response.source_nodes[:3]):  # Max 3 Quellen
                source_text = node.text[:200] + "..." if len(node.text) > 200 else node.text
                answer += f"• *{source_text}*\n"
        
        msg.content = answer
        await msg.update()
        
    except Exception as e:
        msg.content = f"❌ Fehler bei der Suche: {e}"
        await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """Hauptfunktion für Nachrichten"""
    await process_question(message.content)

if __name__ == "__main__":
    # Teste die Funktionen
    import asyncio
    
    async def test():
        await create_index()
        print("Test erfolgreich!")
    
    asyncio.run(test())