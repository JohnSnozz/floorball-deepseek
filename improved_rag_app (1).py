# improved_rag_app.py - Verbesserte RAG Pipeline
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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Bessere Ollama Konfiguration
Settings.llm = Ollama(
    model="deepseek-r1:14b",
    base_url="http://localhost:11434",
    request_timeout=300.0,
    temperature=0.1,  # Weniger kreativ, mehr faktisch
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

# Globale Variablen
INDEX = None
QUERY_ENGINE = None
DATA_SOURCE = "unknown"

def create_realistic_reglement():
    """Erstelle ein realistisches Unihockey-Reglement basierend auf echten Regeln"""
    
    reglement_content = """
# SCHWEIZER UNIHOCKEY REGLEMENT - SPIELREGELN (SRS 3-4)

## 1. SPIELFELD UND AUSRÜSTUNG

### 1.1 Spielfeld
Das Spielfeld ist rechteckig und von mindestens 50 cm hohen Banden umgeben.
- Standardgröße: 40m x 20m (Länge x Breite)
- Minimalgröße: 36m x 18m
- Die Ecken müssen abgerundet sein (Radius 1m)
- Torraum: Halbkreis mit 4m Radius vor jedem Tor

### 1.2 Tore
- Größe: 1,60m x 1,15m (Breite x Höhe)
- Tiefe: mindestens 0,60m
- Torlinien: 3,5m von der Stirnwand entfernt

### 1.3 Markierungen
- Mittellinie teilt das Spielfeld in zwei Hälften
- Anstoßpunkt in der Spielfeldmitte
- Penalty-Punkt: 7m vor jedem Tor
- Schiedsrichter-Eckraum: 1,5m x 1m in jeder Ecke

## 2. SPIELER UND POSITIONEN

### 2.1 Mannschaftsstärke
- Jede Mannschaft darf maximal 6 Spieler gleichzeitig auf dem Feld haben
- Davon 5 Feldspieler und 1 Torhüter
- Kader: maximal 20 Spieler pro Spiel
- Mindestens 8 Spieler müssen eingetragen sein

### 2.2 Torhüter
- Darf den Ball im Torraum mit allen Körperteilen spielen
- Außerhalb des Torraums gelten normale Feldspielregeln
- Muss sich farblich von Feldspielern unterscheiden
- Spezielle Schutzausrüstung erlaubt

### 2.3 Wechsel
- Beliebig viele Wechsel während des Spiels möglich
- Wechsel nur über die eigene Auswechselbank
- Spieler muss das Feld verlassen haben, bevor Ersatzspieler kommt

## 3. SPIELZEIT UND UNTERBRECHUNGEN

### 3.1 Spielzeit
- 3 Perioden à 20 Minuten reine Spielzeit
- Pause zwischen 1. und 2. Periode: 10 Minuten
- Pause zwischen 2. und 3. Periode: 15 Minuten
- Bei Jugendspielern kürzere Spielzeiten möglich

### 3.2 Verlängerung
- Bei Gleichstand in K.O.-Spielen: 2 x 10 Minuten Verlängerung
- Falls weiterhin unentschieden: Penalty-Schießen
- 3 Penalty pro Mannschaft, dann 1:1 bis Entscheidung

### 3.3 Time-out
- Jede Mannschaft hat pro Spiel ein Time-out (30 Sekunden)
- Nur bei eigenem Ballbesitz möglich

## 4. AUSRÜSTUNG

### 4.1 Stock
- Maximale Länge: 114 cm
- Schaufel: maximal 30 cm breit, 16 cm hoch
- Material: zugelassene Kunststoffe und Komposite
- Keine scharfen Kanten oder Beschädigungen

### 4.2 Ball
- Material: Kunststoff
- Durchmesser: 72 mm
- Gewicht: 23 Gramm
- Farbe: weiß (Ausnahmen in Absprache möglich)

### 4.3 Spielerausrüstung
Obligatorisch:
- Unihockey-Schuhe (keine Metallstollen)
- Stutzen (mindestens bis zum Knie)
- Kurze Sporthosen
- Trikot mit gut sichtbarer Nummer (1-99)

Zusätzlich für Torhüter:
- Helm mit Gesichtsschutz
- Brustschutz
- Beinschützer
- Spezielle Handschuhe

## 5. STRAFEN UND DISZIPLIN

### 5.1 Kleine Strafen (2 Minuten)
- Behinderung eines Gegenspielers
- Stockschlag (auch unabsichtlich)
- Gefährliches Spiel
- Bandencheck
- Unsportliches Verhalten
- Verzögerung des Spiels

### 5.2 Große Strafen (5 Minuten)
- Grobes Foulspiel
- Check gegen den Kopf-/Nackenbereich
- Stockschlag mit Verletzungsfolge
- Wiederholtes unsportliches Verhalten

### 5.3 Matchstrafen
- Schwere Verletzung des Gegners
- Tätlichkeiten
- Schweres unsportliches Verhalten
- Ausschluss für restliches Spiel

### 5.4 Powerplay
- Bei kleinen und großen Strafen spielt das gestrafte Team in Unterzahl
- Bei Tor gegen Unterzahl-Team endet kleine Strafe vorzeitig
- Große Strafen müssen vollständig abgesessen werden

## 6. SPIELREGELN

### 6.1 Spielbeginn
- Anspiel in der Spielfeldmitte
- Ball muss berührt werden, bevor andere Spieler eingreifen
- Nach jedem Tor: Anspiel durch die Mannschaft, die das Tor kassiert hat

### 6.2 Offside
- Ein Spieler steht im Offside, wenn er sich bei Ballabgabe eines Mitspielers vor dem Ball in der gegnerischen Spielhälfte befindet
- Ausnahmen: Der Spieler läuft den Ball ein, erhält einen Pass in der eigenen Hälfte

### 6.3 Penalty
Ein Penalty wird verhängt bei:
- Foul an einem Spieler mit klarer Torchance
- Absichtliches Wegschlagen des Balles mit der Hand im Torraum
- Zu viele Spieler auf dem Feld bei Torchance

Ausführung:
- Schuss vom Penalty-Punkt (7m)
- Nur Schütze und Torhüter im Torraum
- Ball muss vorwärts gespielt werden

### 6.4 Freistöße
- Bei Regelverstößen ohne Strafzeit: direkter Freistoß
- Gegner müssen 3m Abstand halten
- Indirekter Freistoß bei Offside und anderen technischen Vergehen

## 7. SCHIEDSRICHTER

### 7.1 Schiedsrichterteam
- 2 Hauptschiedsrichter mit gleichen Rechten
- Entscheidungen sind endgültig
- Können alle Strafen verhängen

### 7.2 Aufgaben
- Spielleitung und Regelüberwachung
- Zeitmessung und Spielprotokoll
- Sicherheit aller Beteiligten
- Fairplay-Überwachung

## 8. FAIR PLAY

### 8.1 Grundsätze
- Unihockey ist ein kontaktloser Sport
- Körperkontakt ist grundsätzlich nicht erlaubt
- Respekt gegenüber Gegnern, Mitspielern und Schiedsrichtern
- Spieler sind für eigenes Verhalten verantwortlich

### 8.2 Verboten
- Jeglicher Körperkontakt (Schlagen, Stoßen, Festhalten)
- Stockcheck (Schlagen gegen den Gegnerstock)
- Beinstellen oder zu Fall bringen
- Behinderung ohne Ballbesitz

## 9. SPEZIALREGELN

### 9.1 Torhüter-Spiel
- Torhüter darf Ball maximal 3 Sekunden in den Händen halten
- Nach Ballkontrolle muss Ball sofort gespielt werden
- Verzögerungstaktik wird bestraft

### 9.2 Bankstrafen
- Bei grobem Vergehen kann ganze Bank bestraft werden
- Trainer und Betreuer können des Feldes verwiesen werden

### 9.3 Protest und Einsprachen
- Proteste nur bei Regelmissverständnissen möglich
- Schiedsrichterentscheidungen sind nicht protestierbar
- Offizielle Einsprachen über Verband möglich
    """
    
    return reglement_content

async def load_documents_step():
    """Lade Dokumente - priorisiere Online vor PDF"""
    reglement_path = Path("data/unihockey_reglement.txt")
    pdf_path = Path("data/unihockey_reglement.pdf")
    
    data_source = "demo"  # Default
    content = None
    
    # 1. ERSTE PRIORITÄT: Online-Scraping (immer aktuell)
    print("🌐 Versuche Online-Scraping (höchste Priorität)...")
    try:
        from scrape_reglement import scrape_unihockey_reglement
        content = scrape_unihockey_reglement()
        if content and len(content) > 1000:  # Mindestlänge für valides Reglement
            print(f"✅ Online-Reglement geladen: {len(content)} Zeichen")
            data_source = "online"
        else:
            print("⚠️ Online-Scraping unvollständig")
            content = None
    except Exception as e:
        print(f"❌ Online-Scraping-Fehler: {e}")
        content = None
    
    # 2. ZWEITE PRIORITÄT: PDF (falls Online fehlschlägt)
    if not content and pdf_path.exists():
        print("📄 Online fehlgeschlagen - verwende PDF als Fallback")
        try:
            from pdf_loader import load_reglement_from_pdf
            content = load_reglement_from_pdf()
            if content:
                print(f"✅ PDF erfolgreich geladen: {len(content)} Zeichen")
                data_source = "pdf"
        except Exception as e:
            print(f"❌ PDF-Fehler: {e}")
            content = None
    
    # 3. LETZTE PRIORITÄT: Demo-Reglement
    if not content:
        print("📝 Verwende Demo-Reglement als letzten Fallback")
        content = create_realistic_reglement()
        data_source = "demo"
        
        # Speichere für zukünftige Verwendung
        os.makedirs('data', exist_ok=True)
        with open(reglement_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # Erstelle strukturierte Dokumente
    documents = []
    
    # Teile Reglement in Kapitel auf
    sections = content.split('## ')
    
    for i, section in enumerate(sections[1:], 1):  # Skip erste leere Sektion
        lines = section.strip().split('\n')
        title = lines[0] if lines else f"Kapitel {i}"
        section_content = '\n'.join(lines[1:]) if len(lines) > 1 else ""
        
        if section_content.strip():
            doc = Document(
                text=f"# {title}\n\n{section_content}",
                metadata={
                    "source": "Swiss Unihockey Reglement",
                    "chapter": title,
                    "chapter_number": i,
                    "type": "regulation",
                    "data_source": data_source  # Merke dir die Datenquelle
                }
            )
            documents.append(doc)
    
    print(f"📚 {len(documents)} Kapitel erstellt (Quelle: {data_source})")
    return documents, data_source

async def create_index_step(documents):
    """Erstelle optimierten Vector Index"""
    
    # Besserer Node Parser
    node_parser = SentenceSplitter(
        chunk_size=256,      # Kleinere Chunks für bessere Präzision
        chunk_overlap=20,    # Weniger Überlappung
        paragraph_separator="\n\n",
        secondary_chunking_regex="[.!?]\\s+"  # Teile bei Satzende
    )
    
    # Index mit besseren Einstellungen
    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser,
        show_progress=False
    )
    
    return index

async def retrieve_step(question):
    """Verbesserte Retrieval mit Scoring"""
    global INDEX
    
    # Erstelle besseren Retriever
    retriever = VectorIndexRetriever(
        index=INDEX,
        similarity_top_k=5,  # Mehr Kandidaten
    )
    
    # Post-Processor für bessere Filterung
    postprocessor = SimilarityPostprocessor(
        similarity_cutoff=0.6  # Nur relevante Ergebnisse
    )
    
    # Retrieve Nodes
    nodes = await asyncio.to_thread(retriever.retrieve, question)
    
    # Filter mit Post-Processor
    filtered_nodes = postprocessor.postprocess_nodes(nodes)
    
    # Zeige detaillierte Retrieval-Info
    if filtered_nodes:
        sources_preview = f"**📚 {len(filtered_nodes)} relevante Quellen (von {len(nodes)} gefunden):**\n\n"
        for i, node in enumerate(filtered_nodes):
            score = getattr(node, 'score', 0)
            chapter = node.metadata.get('chapter', 'Unbekannt')
            preview = node.text[:100].replace('\n', ' ') + "..."
            sources_preview += f"**[{i+1}]** {chapter} (Score: {score:.3f})\n*{preview}*\n\n"
        
        await cl.Message(content=sources_preview).send()
    else:
        await cl.Message(content="⚠️ Keine relevanten Quellen gefunden - versuche eine spezifischere Frage").send()
    
    return filtered_nodes

async def generate_answer_step(question, nodes):
    """Generiere eine klare, eindeutige Antwort"""
    global QUERY_ENGINE
    
    if not nodes:
        return type('MockResponse', (), {
            'response': "❌ Keine relevanten Informationen im Reglement gefunden. Bitte stelle eine spezifischere Frage zu Unihockey-Regeln.",
            'source_nodes': []
        })()
    
    try:
        # Verbesserter Prompt für eindeutige Antworten
        context_prompt = f"""
Du bist ein Unihockey-Regelexperte. Beantworte die Frage mit EINER klaren, eindeutigen Antwort basierend auf dem Schweizer Unihockey-Reglement.

WICHTIGE REGELN:
1. Gib NUR EINE eindeutige, direkte Antwort
2. Keine Aufzählungen oder mehrere Optionen
3. Beginne direkt mit der Antwort, ohne "Das Reglement besagt..." 
4. Sei präzise und faktisch
5. Falls mehrere Werte existieren (z.B. Standard- und Minimalgröße), nenne den Standardwert als Hauptantwort
6. Wenn die Information nicht eindeutig im Reglement steht, sage: "Diese Information ist im verfügbaren Reglement nicht eindeutig definiert."

Beispiele für gute Antworten:
- "Das Spielfeld ist 40m x 20m groß."
- "Eine kleine Strafe dauert 2 Minuten."
- "6 Spieler sind gleichzeitig auf dem Feld erlaubt."

Frage: {question}

Antwort (eine klare, eindeutige Aussage):"""
        
        response = await asyncio.to_thread(QUERY_ENGINE.query, context_prompt)
        
        # Nachbearbeitung der Antwort um sicherzustellen, dass sie eindeutig ist
        answer = response.response.strip()
        
        # Entferne typische "Aufzählungs-Indikatoren"
        cleanup_patterns = [
            "Das Reglement besagt:",
            "Laut Reglement:",
            "Die Antwort lautet:",
            "Es gibt mehrere Möglichkeiten:",
            "Die Optionen sind:",
        ]
        
        for pattern in cleanup_patterns:
            if answer.startswith(pattern):
                answer = answer[len(pattern):].strip()
        
        # Falls die Antwort immer noch Aufzählungen enthält, nimm nur den ersten Punkt
        if answer.startswith(("- ", "• ", "1. ", "* ")):
            lines = answer.split('\n')
            first_line = lines[0]
            # Entferne Aufzählungszeichen
            for prefix in ["- ", "• ", "1. ", "* ", "2. ", "3. "]:
                if first_line.startswith(prefix):
                    first_line = first_line[len(prefix):].strip()
                    break
            answer = first_line
        
        # Prüfe ob die Antwort unvollständig ist (abgeschnitten)
        if answer.endswith(("...", ".", "#", "1.", "2.")) and len(answer) < 20:
            # Fallback: Verwende direkt die beste Node-Information
            if nodes:
                best_node = nodes[0]
                text = best_node.text
                
                # Extrahiere spezifische Informationen basierend auf der Frage
                if "groß" in question.lower() or "größe" in question.lower():
                    # Suche nach Größenangaben
                    import re
                    size_matches = re.findall(r'(\d+m?\s*x?\s*\d+m?)', text)
                    if size_matches:
                        answer = f"Das Spielfeld ist {size_matches[0]} groß."
                elif "spieler" in question.lower():
                    player_matches = re.findall(r'(\d+\s*Spieler)', text)
                    if player_matches:
                        answer = f"{player_matches[0]} sind gleichzeitig auf dem Feld erlaubt."
                else:
                    # Nimm den ersten vollständigen Satz
                    sentences = text.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 10:
                            answer = sentence.strip() + "."
                            break
        
        # Erstelle neue Response mit bereinigter Antwort
        return type('CleanResponse', (), {
            'response': answer,
            'source_nodes': response.source_nodes if hasattr(response, 'source_nodes') else nodes
        })()
        
    except Exception as e:
        if "terminated" in str(e) or "killed" in str(e):
            # Fallback: Nimm direkt die relevanteste Information
            if nodes:
                best_node = nodes[0]  # Bester Match
                context = best_node.text[:200]
                
                # Versuche eine direkte Antwort aus dem Context zu extrahieren
                answer = f"Laut Reglement: {context.split('.')[0]}."
                
                return type('MockResponse', (), {
                    'response': answer,
                    'source_nodes': nodes
                })()
            else:
                return type('MockResponse', (), {
                    'response': "❌ LLM-Fehler und keine Reglement-Informationen verfügbar.",
                    'source_nodes': []
                })()
        else:
            raise e

async def create_index():
    """Erstelle oder lade optimierten Vector Index"""
    global INDEX, QUERY_ENGINE, DATA_SOURCE
    
    storage_dir = "storage_improved"
    
    if os.path.exists(storage_dir):
        try:
            msg = cl.Message(content="📂 Lade optimierten Index...")
            await msg.send()
            
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            INDEX = load_index_from_storage(storage_context)
            
            # Versuche Datenquelle aus Index-Metadaten zu lesen
            if INDEX.docstore.docs:
                first_doc = list(INDEX.docstore.docs.values())[0]
                DATA_SOURCE = first_doc.metadata.get('data_source', 'unknown')
            
            msg.content = "✅ Optimierter Index geladen"
            await msg.update()
        except:
            INDEX = None
    
    if INDEX is None:
        msg = cl.Message(content="🔄 Erstelle optimierten Index...")
        await msg.send()
        
        documents, data_source = await load_documents_step()
        DATA_SOURCE = data_source
        INDEX = await create_index_step(documents)
        INDEX.storage_context.persist(persist_dir=storage_dir)
        
        msg.content = f"✅ Optimierter Index erstellt ({len(documents)} Kapitel)"
        await msg.update()
    
    # Bessere Query Engine
    QUERY_ENGINE = INDEX.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.6)]
    )

@cl.on_chat_start
async def start():
    """Initialisierung"""
    start_time = time.time()
    await create_index()
    end_time = time.time()
    
    # Datenquelle-Info
    source_info = {
        "online": "🌐 **Datenquelle:** Aktuelles Online-Reglement (immer aktuell)",
        "pdf": "📄 **Datenquelle:** PDF-Reglement (möglicherweise veraltet)",
        "demo": "📝 **Datenquelle:** Demo-Reglement (Beispieldaten)"
    }
    
    source_text = source_info.get(DATA_SOURCE, "❓ **Datenquelle:** Unbekannt")
    
    await cl.Message(
        content=f"""🏒 **Verbessertes Unihockey RAG System** (⏱️ {end_time - start_time:.1f}s)

🤖 **Model:** DeepSeek-R1:14b
{source_text}
🔍 **Features:** Eindeutige Antworten + Vollständige Extraktion

💡 **Teste spezifische Fragen:**
- "Wie groß ist das Spielfeld?"
- "Wie lange dauert eine kleine Strafe?"
- "Wie viele Spieler sind auf dem Feld?"
- "Wo wird ein Penalty ausgeführt?"
- "Wie hoch sind die Banden?"

✅ **System optimiert für klare, vollständige Antworten!**"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Hauptfunktion mit verbesserter RAG Pipeline und separaten Quellen"""
    
    start_time = time.time()
    question = message.content
    
    # Schritt 1: Retrieval
    status_msg = cl.Message(content="🔍 **Schritt 1:** Suche in Reglement-Kapiteln...")
    await status_msg.send()
    
    nodes = await retrieve_step(question)
    
    status_msg.content = f"✅ **Schritt 1:** {len(nodes)} relevante Abschnitte gefunden"
    await status_msg.update()
    
    # Schritt 2: LLM Generation
    status_msg2 = cl.Message(content="🤖 **Schritt 2:** Generiere regelkonforme Antwort...")
    await status_msg2.send()
    
    response = await generate_answer_step(question, nodes)
    
    status_msg2.content = f"✅ **Schritt 2:** Antwort basierend auf {len(nodes)} Quellen generiert"
    await status_msg2.update()
    
    # Finale Antwort OHNE Quellen
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Zeige Datenquelle in der Antwort
    source_emoji = {"online": "🌐", "pdf": "📄", "demo": "📝"}.get(DATA_SOURCE, "❓")
    source_text = {"online": "Online", "pdf": "PDF", "demo": "Demo"}.get(DATA_SOURCE, "Unbekannt")
    
    main_answer = f"""**🎯 Eindeutige Antwort:**

{response.response}

---
⏱️ {processing_time:.1f}s | 🤖 DeepSeek-R1:14b | {source_emoji} {source_text}-Reglement | 📚 {len(nodes)} Kapitel"""
    
    await cl.Message(content=main_answer).send()
    
    # Separate Quellenangabe rechts davon (als separate Message)
    if hasattr(response, 'source_nodes') and response.source_nodes:
        source_details = "**📋 Verwendete Regelabschnitte:**\n\n"
        for i, node in enumerate(response.source_nodes[:3]):
            chapter = node.metadata.get('chapter', 'Regelabschnitt')
            ref = node.text[:80].replace('\n', ' ') + "..."
            score = getattr(node, 'score', 0)
            source_details += f"**[{i+1}] {chapter}** (Score: {score:.3f})\n*{ref}*\n\n"
        
        # Sende Quellen als separate Message mit Indikator dass es Quellen sind
        await cl.Message(
            content=source_details,
            author="Quellen"  # Zeigt es als separate "Quellen"-Message
        ).send()

if __name__ == "__main__":
    import asyncio
    
    async def test():
        await create_index()
        print("✅ Verbesserte RAG Pipeline bereit!")
    
    asyncio.run(test())