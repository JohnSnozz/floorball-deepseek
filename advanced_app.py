# advanced_app.py - Chainlit mit detailliertem Train of Thought
import os
import asyncio
import time
import chainlit as cl
from pathlib import Path

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Ollama konfigurieren - 14b Model sollte mit 16GB funktionieren
Settings.llm = Ollama(
    model="deepseek-r1:14b",  # 14b Model mit 16GB VRAM möglich
    base_url="http://localhost:11434",
    request_timeout=300.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Globale Variablen
INDEX = None
QUERY_ENGINE = None

async def load_documents_step():
    """Lade Dokumente"""
    reglement_path = Path("data/unihockey_reglement.txt")
    
    if not reglement_path.exists():
        # Erstelle Demo-Reglement
        demo_content = """
# UNIHOCKEY REGLEMENT (Demo)

## Spielfeld
Das Spielfeld ist 40m x 20m groß.

## Spieleranzahl  
6 Spieler pro Team (5 Feldspieler + 1 Torhüter).

## Spielzeit
3 x 20 Minuten reine Spielzeit.

## Strafen
- Kleine Strafe: 2 Minuten
- Große Strafe: 5 Minuten
- Matchstrafe: Spielausschluss
        """
        
        os.makedirs('data', exist_ok=True)
        with open(reglement_path, 'w', encoding='utf-8') as f:
            f.write(demo_content)
    
    with open(reglement_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    document = Document(
        text=content,
        metadata={"source": "Unihockey Reglement", "type": "regulation"}
    )
    
    return [document]

async def create_index_step(documents):
    """Erstelle Vector Index"""
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
        show_progress=False
    )
    
    return index

async def retrieve_step(question):
    """Retrieve relevante Dokumente mit Quellenausgabe"""
    global QUERY_ENGINE
    
    retriever = QUERY_ENGINE._retriever
    nodes = await asyncio.to_thread(retriever.retrieve, question)
    
    # Kompakte Quellenausgabe
    if nodes:
        sources_preview = f"**📚 {len(nodes)} Quellen gefunden:**\n\n"
        for i, node in enumerate(nodes):
            # Nur erste 80 Zeichen + Score
            score = getattr(node, 'score', 'N/A')
            preview = node.text[:80].replace('\n', ' ') + "..."
            sources_preview += f"**[{i+1}]** (Score: {score:.3f})\n*{preview}*\n\n"
        
        # Zeige kompakte Quellen
        await cl.Message(content=sources_preview).send()
    
    return nodes

async def generate_answer_step(question, nodes):
    """Generiere Antwort mit LLM und Error Handling"""
    global QUERY_ENGINE
    
    try:
        response = await asyncio.to_thread(QUERY_ENGINE.query, question)
        return response
    except Exception as e:
        # Fallback bei Memory-Problemen
        if "terminated" in str(e) or "killed" in str(e):
            # Versuche mit kleineren Chunks
            simplified_query = question[:100]  # Kürze die Frage
            
            # Erstelle einfache Antwort aus gefundenen Nodes
            if nodes:
                context = "\n".join([node.text[:200] for node in nodes[:2]])
                
                # Erstelle Mock-Response
                class MockResponse:
                    def __init__(self, text, source_nodes):
                        self.response = f"Basierend auf dem Reglement:\n\n{context[:500]}..."
                        self.source_nodes = source_nodes
                
                return MockResponse(context, nodes)
            else:
                class MockResponse:
                    def __init__(self):
                        self.response = "❌ Entschuldigung, das LLM-Model hatte Memory-Probleme. Bitte verwende ein kleineres Model oder versuche es erneut."
                        self.source_nodes = []
                
                return MockResponse()
        else:
            raise e

async def create_index():
    """Erstelle oder lade Vector Index"""
    global INDEX, QUERY_ENGINE
    
    storage_dir = "storage"
    
    if os.path.exists(storage_dir):
        try:
            msg = cl.Message(content="📂 Lade bestehenden Index...")
            await msg.send()
            
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            INDEX = load_index_from_storage(storage_context)
            
            msg.content = "✅ Index erfolgreich geladen"
            await msg.update()
        except:
            INDEX = None
    
    if INDEX is None:
        msg = cl.Message(content="🔄 Erstelle neuen Index...")
        await msg.send()
        
        documents = await load_documents_step()
        INDEX = await create_index_step(documents)
        INDEX.storage_context.persist(persist_dir=storage_dir)
        
        msg.content = f"✅ Index erstellt mit {len(documents)} Dokumenten"
        await msg.update()
    
    QUERY_ENGINE = INDEX.as_query_engine(similarity_top_k=3)

@cl.on_chat_start
async def start():
    """Initialisierung beim Chat-Start"""
    
    start_time = time.time()
    
    await create_index()
    
    end_time = time.time()
    
    await cl.Message(
        content=f"""🏒 **Unihockey RAG System bereit!** (⏱️ {end_time - start_time:.1f}s)

🤖 **Model:** DeepSeek-R1:14b
🔍 **Features:** Train of Thought + Quellenangaben

💡 **Beispiele:**
- Wie groß ist das Spielfeld?
- Welche Strafen gibt es?
- Wie viele Spieler sind erlaubt?"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Hauptfunktion mit detailliertem Train of Thought und Quellenangabe"""
    
    start_time = time.time()
    question = message.content
    
    # Schritt 1: Status anzeigen
    status_msg = cl.Message(content="🔍 **Schritt 1:** Suche relevante Dokumente...")
    await status_msg.send()
    
    # Dokumente durchsuchen
    nodes = await retrieve_step(question)
    
    status_msg.content = f"✅ **Schritt 1:** {len(nodes)} relevante Abschnitte gefunden"
    await status_msg.update()
    
    # Schritt 2: Antwort generieren
    status_msg2 = cl.Message(content="🤖 **Schritt 2:** DeepSeek-R1:14b generiert Antwort...")
    await status_msg2.send()
    
    response = await generate_answer_step(question, nodes)
    
    status_msg2.content = f"✅ **Schritt 2:** Antwort generiert ({len(response.response)} Zeichen)"
    await status_msg2.update()
    
    # Kompakte finale Antwort
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Kurze Quellenangabe
    source_refs = ""
    if hasattr(response, 'source_nodes') and response.source_nodes:
        source_refs = "\n\n**📋 Verwendete Quellen:**\n"
        for i, node in enumerate(response.source_nodes[:3]):
            ref = node.text[:60].replace('\n', ' ') + "..."
            source_refs += f"[{i+1}] *{ref}*\n"
    
    final_answer = f"""**🏒 Antwort:**

{response.response}
{source_refs}
---
⏱️ {processing_time:.1f}s | 🤖 DeepSeek-R1:14b | 📊 {len(nodes)} Quellen"""
    
    await cl.Message(content=final_answer).send()

if __name__ == "__main__":
    import asyncio
    
    async def test():
        await create_index()
        print("✅ Test erfolgreich!")
    
    asyncio.run(test())