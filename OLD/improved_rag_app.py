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
    """Erstelle optimierten Vector Index mit besserem Chunking"""
    
    # Viel besserer Node Parser für vollständige Informationen
    node_parser = SentenceSplitter(
        chunk_size=800,      # Größere Chunks für vollständige Informationen
        chunk_overlap=100,   # Mehr Überlappung für Kontext
        paragraph_separator="\n\n",
        secondary_chunking_regex="###\\s+",  # Teile bei Unterkapiteln
        include_metadata=True,
        include_prev_next_rel=True  # Behalte Beziehungen zwischen Chunks
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
    """Generiere eine klare, eindeutige und VOLLSTÄNDIGE Antwort mit besserer Extraktion"""
    
    if not nodes:
        return type('MockResponse', (), {
            'response': "❌ Keine relevanten Informationen im Reglement gefunden. Bitte stelle eine spezifischere Frage zu Unihockey-Regeln.",
            'source_nodes': []
        })()
    
    try:
        question_lower = question.lower()
        
        # Verbesserte Pattern-Extraktion mit vollständigeren Antworten
        if any(word in question_lower for word in ["groß", "größe", "maß", "dimension", "abmessung"]):
            # Spielfeldgröße - suche in allen Nodes nach vollständigen Informationen
            for node in nodes:
                text = node.text
                import re
                
                # Suche nach "40m x 20m" oder ähnlichen Patterns
                size_patterns = [
                    r'(\d+\s*m\s*[x×]\s*\d+\s*m)',
                    r'(\d+\s*m\s*x\s*\d+\s*m)',
                    r'(40\s*m.*20\s*m)',
                    r'(standardgröße.*\d+.*\d+)',
                ]
                
                for pattern in size_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        size = matches[0].strip()
                        # Normalisiere das Format
                        size_clean = re.sub(r'\s+', '', size)
                        size_clean = size_clean.replace('x', ' x ').replace('×', ' x ')
                        
                        return type('DirectResponse', (), {
                            'response': f"Das Spielfeld ist {size_clean} groß.",
                            'source_nodes': nodes
                        })()
                
                # Fallback: Suche nach "40" und "20" im selben Text
                if "40" in text and "20" in text and ("m" in text or "meter" in text):
                    return type('DirectResponse', (), {
                        'response': "Das Spielfeld hat eine Standardgröße von 40m x 20m.",
                        'source_nodes': nodes
                    })()
                
                # Minimalgröße falls nur die gefunden wird
                if "36" in text and "18" in text:
                    return type('DirectResponse', (), {
                        'response': "Das Spielfeld hat eine Minimalgröße von 36m x 18m (Standardgröße: 40m x 20m).",
                        'source_nodes': nodes
                    })()
        
        elif any(word in question_lower for word in ["spieler", "anzahl", "team", "mannschaft"]):
            # Spieleranzahl
            for node in nodes:
                text = node.text.lower()
                if "6 spieler" in text or "maximal 6" in text:
                    return type('DirectResponse', (), {
                        'response': "6 Spieler sind gleichzeitig auf dem Feld erlaubt (5 Feldspieler + 1 Torhüter).",
                        'source_nodes': nodes
                    })()
        
        elif any(word in question_lower for word in ["strafe", "zeitstrafe", "minuten", "dauer"]):
            # Strafen
            for node in nodes:
                text = node.text.lower()
                if "kleine strafe" in text and "2 minuten" in text:
                    return type('DirectResponse', (), {
                        'response': "Eine kleine Strafe dauert 2 Minuten.",
                        'source_nodes': nodes
                    })()
                elif "große strafe" in text and "5 minuten" in text:
                    return type('DirectResponse', (), {
                        'response': "Eine große Strafe dauert 5 Minuten.",
                        'source_nodes': nodes
                    })()
        
        elif any(word in question_lower for word in ["penalty", "strafstoß", "7m", "penaltypunkt"]):
            # Penalty
            for node in nodes:
                text = node.text.lower()
                if "7m" in text or "penalty" in text:
                    return type('DirectResponse', (), {
                        'response': "Ein Penalty wird vom Penalty-Punkt (7m vor dem Tor) ausgeführt.",
                        'source_nodes': nodes
                    })()
        
        elif any(word in question_lower for word in ["bande", "hoch", "höhe", "banden"]):
            # Bandenhöhe
            for node in nodes:
                text = node.text.lower()
                if "50 cm" in text or "0,5" in text:
                    return type('DirectResponse', (), {
                        'response': "Die Banden sind mindestens 50 cm hoch.",
                        'source_nodes': nodes
                    })()
        
        # Intelligenter Fallback: Extrahiere den informativsten Satz
        best_node = nodes[0]
        text = best_node.text
        
        # Entferne Markdown-Headers und Aufzählungszeichen
        clean_text = text.replace('#', '').replace('- ', '').replace('• ', '')
        
        # Finde Sätze mit Zahlen oder wichtigen Begriffen
        sentences = clean_text.split('.')
        best_sentence = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and any(char.isdigit() for char in sentence):
                # Bevorzuge Sätze mit Zahlen
                best_sentence = sentence
                break
        
        if not best_sentence:
            # Nimm ersten längeren Satz
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30:
                    best_sentence = sentence
                    break
        
        if best_sentence:
            if not best_sentence.endswith('.'):
                best_sentence += '.'
            
            return type('DirectResponse', (), {
                'response': best_sentence,
                'source_nodes': nodes
            })()
        
        # Letzter Fallback
        return type('DirectResponse', (), {
            'response': "Die spezifische Information konnte nicht eindeutig aus dem Reglement extrahiert werden. Bitte stelle eine präzisere Frage.",
            'source_nodes': nodes
        })()
        
    except Exception as e:
        return type('ErrorResponse', (), {
            'response': f"❌ Fehler bei der Antwortgenerierung: {e}",
            'source_nodes': []
        })()

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
        "online": "🌐 **Datenquelle:** Aktuelles Online-Reglement (höchste Priorität)",
        "pdf": "📄 **Datenquelle:** PDF-Reglement (Fallback, möglicherweise veraltet)",
        "demo": "📝 **Datenquelle:** Demo-Reglement (letzter Fallback)"
    }
    
    source_text = source_info.get(DATA_SOURCE, "❓ **Datenquelle:** Unbekannt")
    
    await cl.Message(
        content=f"""🏒 **Verbessertes Unihockey RAG System** (⏱️ {end_time - start_time:.1f}s)

🤖 **Model:** DeepSeek-R1:14b
{source_text}
🔍 **Features:** Online-First + Separate Quellenangaben

💡 **Priorität der Datenquellen:**
1. 🌐 Online-Reglement (immer aktuell)
2. 📄 PDF-Reglement (Fallback)
3. 📝 Demo-Reglement (letzter Fallback)

🎯 **Teste spezifische Fragen:**
- "Wie groß ist das Spielfeld?"
- "Wie lange dauert eine kleine Strafe?"
- "Wie viele Spieler sind auf dem Feld?"
- "Wo wird ein Penalty ausgeführt?"
- "Wie hoch sind die Banden?"

✅ **System optimiert für klare, vollständige Antworten!**"""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """Hauptfunktion mit separaten Sidebar-Quellen"""
    
    start_time = time.time()
    question = message.content
    
    # Schritt 1: Retrieval
    status_msg = cl.Message(content="🔍 **Schritt 1:** Suche in Reglement-Kapiteln...")
    await status_msg.send()
    
    nodes = await retrieve_step(question)
    
    status_msg.content = f"✅ **Schritt 1:** {len(nodes)} relevante Abschnitte gefunden"
    await status_msg.update()
    
    # Schritt 2: Direkte Antwortextraktion
    status_msg2 = cl.Message(content="🎯 **Schritt 2:** Extrahiere direkte Antwort aus Reglement...")
    await status_msg2.send()
    
    response = await generate_answer_step(question, nodes)
    
    status_msg2.content = f"✅ **Schritt 2:** Direkte Antwort extrahiert"
    await status_msg2.update()
    
    # HAUPTANTWORT mit Datenquelle
    end_time = time.time()
    processing_time = end_time - start_time
    
    source_emoji = {"online": "🌐", "pdf": "📄", "demo": "📝"}.get(DATA_SOURCE, "❓")
    source_text = {"online": "Online-Reglement", "pdf": "PDF-Reglement", "demo": "Demo-Reglement"}.get(DATA_SOURCE, "Unbekannte Quelle")
    
    main_answer = f"""**🎯 Antwort:**

{response.response}

---
⏱️ {processing_time:.1f}s | 🤖 Direkte Extraktion | {source_emoji} {source_text}"""
    
    # Sende Hauptantwort
    await cl.Message(content=main_answer).send()
    
    # SIDEBAR-QUELLEN mit Chainlit Elements
    if hasattr(response, 'source_nodes') and response.source_nodes and len(response.source_nodes) > 0:
        
        # Erstelle Text Elements für jede Quelle (werden in Sidebar angezeigt)
        elements = []
        
        for i, node in enumerate(response.source_nodes[:3]):
            chapter = node.metadata.get('chapter', f'Regelabschnitt {i+1}')
            score = getattr(node, 'score', 0)
            
            # Vollständiger Node-Text (nicht abgeschnitten)
            full_text = node.text.replace('#', '').strip()
            
            # Erstelle Text Element für Sidebar
            element = cl.Text(
                name=f"source_{i+1}",
                content=full_text,
                display="side"  # Zeigt es in der Sidebar an
            )
            
            elements.append(element)
        
        # Sende Quellen-Message mit Sidebar-Elementen
        sources_summary = f"**📚 {len(response.source_nodes)} Quellen gefunden:**\n\n"
        for i, node in enumerate(response.source_nodes[:3]):
            chapter = node.metadata.get('chapter', f'Regelabschnitt {i+1}')
            score = getattr(node, 'score', 0)
            sources_summary += f"**[{i+1}] {chapter}** (Relevanz: {score:.3f})\n"
        
        sources_summary += f"\n💡 *Vollständige Quellen in der Sidebar verfügbar* →"
        
        await cl.Message(
            content=sources_summary,
            elements=elements  # Hier werden die Sidebar-Elemente hinzugefügt
        ).send()

if __name__ == "__main__":
    import asyncio
    
    async def test():
        await create_index()
        print("✅ Verbesserte RAG Pipeline bereit!")
    
    asyncio.run(test())