# -----------------------------------------------------------------------------------------
# BESCHREIBUNG:
# Dieses Skript ist eine Streamlit-Webanwendung, die einen RAG (Retrieval-Augmented Generation) 
# Chatbot bereitstellt. Der Chatbot dient als Finanzanalyst für den Nasdaq 100.
#
# FUNKTIONSWEISE:
# 1. UI & Setup: Der Nutzer gibt seinen OpenAI API-Key ein.
# 2. Data Ingestion: Die CSV-Datei mit Finanzdaten wird geladen, bereinigt und in Dokumente umgewandelt.
# 3. Vektorisierung: Die Dokumente werden in Chunks (Schnipsel) zerteilt, in Vektoren (Embeddings) 
#    umgewandelt und in einer ChromaDB gespeichert (In-Memory).
# 4. RAG-Workflow (LangGraph):
#    - Reformulate: Die Nutzerfrage wird basierend auf dem Chatverlauf präzisiert.
#    - Retrieve: Relevante Dokumente werden aus der Datenbank gesucht.
#    - Generate: Ein LLM (GPT-3.5) generiert eine Antwort basierend auf den gefundenen Finanzdaten.
# -----------------------------------------------------------------------------------------

# Importiert Streamlit für die Web-Oberfläche
import streamlit as st
# Importiert Pandas für die Datenverarbeitung (CSV laden)
import pandas as pd
# Importiert OS für Umgebungsvariablen (API Key)
import os
# Importiert Typisierungen für saubereren Code
from typing import TypedDict, List, Any
# Importiert das Embedding-Modell von OpenAI
from langchain_openai import OpenAIEmbeddings
# Importiert das Chat-Modell von OpenAI (z.B. GPT-3.5/4)
from langchain_openai import ChatOpenAI
# Importiert die Vektor-Datenbank Chroma
from langchain_community.vectorstores import Chroma
# Importiert die Dokument-Klasse für LangChain
from langchain_core.documents import Document
# Importiert Nachrichten-Typen für den Chat-Verlauf
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Importiert den Text-Splitter zum Zerteilen langer Texte
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Importiert LangGraph für den RAG-Workflow (Knoten & Kanten)
from langgraph.graph import StateGraph, END

# Konfiguriert die Seite (Titel, Icon, Layout)
st.set_page_config(page_title="Der Nasdaq Experte", page_icon="📈", layout="wide")

# --- Sidebar & Setup ---
with st.sidebar:
    # Titel in der Seitenleiste
    st.title("⚙️ Einstellungen")
    
    # Eingabefeld für den API Key (Maskiert als Passwort)
    api_key = st.text_input("OpenAI API Key", type="password", help="Gib hier deinen OpenAI API Key ein.")
    
    # Wenn kein Key eingegeben wurde, wird eine Warnung gezeigt und das Skript stoppt hier
    if not api_key:
        st.warning("Bitte gib einen API Key ein, um fortzufahren.")
        st.stop()
    
    # Setzt den API Key als Umgebungsvariable für LangChain
    os.environ["OPENAI_API_KEY"] = api_key

    # Fügt eine Trennlinie hinzu
    st.divider()
    
    # Button zum Neustarten des Chats
    if st.button("Neuer Chat", type="primary", use_container_width=True):
        st.session_state.messages = [] # Leert den Nachrichtenverlauf
        st.rerun() # Lädt die App neu
    
    # Info-Box für den Nutzer
    st.info("Vollständiges Chat-Memory: Das LLM kennt jetzt den Verlauf.")

# --- Initialisierung & Caching (RAG Logik) ---
# @st.cache_resource sorgt dafür, dass die Datenbank nicht bei jedem Klick neu gebaut wird, sondern im Cache bleibt.
@st.cache_resource(show_spinner="Lade Daten und erstelle Vektor-Datenbank...")
def initialize_rag_system():
    
    # 1. CSV Laden & Bereinigen
    # Lädt die CSV-Datei direkt (wir gehen davon aus, dass sie da ist)
    df = pd.read_csv("nasdaq_100_final_for_RAG.csv")

    # Entfernt die Spalte 'PEG Ratio', falls sie existiert (wie im Notebook)
    if "PEG Ratio" in df.columns:
        df = df.drop(columns=["PEG Ratio"])
    
    # Füllt leere Zellen (NaN) mit leeren Strings, damit es keine Fehler gibt
    df = df.fillna("") 
    # Wandelt alle Daten in Strings um, da Embeddings Text brauchen
    df = df.astype(str)

    # 2. Dokumente erstellen (Funktion aus dem IPYNB übernommen)
    def row_to_document(row):
        # Erstellt den Haupttext für das Embedding (Suche)
        # Hier wurde News Summary ergänzt!
        text = f"""
        Company: {row["Company"]}  
        Ticker: {row["Ticker"]}

        Business Summary: {row["Long Business Summary"]}

        Sector: {row["Sector (Yahoo)"]}
        Industry: {row["Industry (Yahoo)"]}
        Country: {row["Country"]}

        Latest News:
        Title: {row["Latest_News_Title"]}
        Summary: {row["News Summary"]}
        Sentiment: {row["Sentiment"]}
        """
    
        # Erstellt das Metadaten-Dictionary (Datenpaket für den Rucksack)
        # Hier wurden 52_week_high, avg_monthly_return und news_summary ergänzt
        metadata = {
            "ticker": row["Ticker"],
            "company": row["Company"],
            "sector": row["Sector (Yahoo)"],
            "industry": row["Industry (Yahoo)"],
            "country": row["Country"],
            "market_cap": row["Market Cap"],
            "current_price": row["Current Price"],
            "previous_close": row["Previous Close"],
            "52_week_high": row["52 Week High"],             # NEU
            "avg_monthly_return": row["Average Monthly Return"], # NEU
            "dividend_yield": row["Dividend Yield"],
            "pe_ratio": row["PE Ratio"],
            "forward_pe": row["Forward PE"],
            "price_to_book": row["Price to Book"],
            "total_revenue": row["Total Revenue"],
            "debt_to_equity": row["Debt to Equity"],
            "roe": row["ROE"],
            "return_1y": row["1y Return"],
            "volatility": row["Volatility"],
            "sentiment": row["Sentiment"],
            "confidence": row["Confidence"],
            "news_link": row["Latest_News_Link"],
            "latest_news_title": row["Latest_News_Title"],
            "news_summary": row["News Summary"],             # NEU
            "website": row["Website"], 
        }
        # Gibt ein Document-Objekt zurück
        return Document(page_content=text, metadata=metadata)

    # Wendet die Funktion auf jede Zeile im DataFrame an --> Liste von Dokumenten
    docs = [row_to_document(row) for i, row in df.iterrows()]

    # 3. Splitting (Text zerteilen)
    # Konfiguriert den Splitter (500 Zeichen pro Chunk, 50 Zeichen Überlappung)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    chunked_docs = [] # Liste für die Schnipsel
    for doc in docs:
        # Zerteilt den Haupttext des Dokuments
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            # Erstellt für jeden Schnipsel ein neues Dokument mit den ORIGINALEN Metadaten
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    # 4. Vector Store (Datenbank erstellen)
    # Lädt das Embedding-Modell
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Erstellt die ChromaDB im Arbeitsspeicher aus den Dokumenten
    db = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, collection_name="nasdaq_docs_memory_full")
    # Erstellt einen Retriever (Suchmaschine), der die 5 ähnlichsten Chunks sucht (k=5 wie besprochen/optimiert)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 5. LLM & Graph Definition
    # Initialisiert das LLM (GPT-3.5 Turbo) mit Temperatur 0 für faktische Antworten
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Definiert die Struktur des Zustands (State) für LangGraph
    class RAGState(TypedDict):
        input: str              # Die aktuelle Frage
        chat_history: List[Any] # Der Chatverlauf
        context: List[Document] # Die gefundenen Dokumente
        answer: str             # Die generierte Antwort

    # --- Knoten 1: Query Reformulation (Frage verbessern) ---
    def reformulate_query(state):
        # Holt den Verlauf
        history = state.get("chat_history", [])
        original_query = state["input"]

        # Wenn kein Verlauf da ist, muss nichts umformuliert werden
        if not history:
            return state

        # Prompt für das Umschreiben der Frage
        system_prompt = "Formuliere die Frage so um, dass sie ohne Vorwissen verständlich ist. Ersetze Pronomen durch Namen. Gib NUR die Frage zurück."
        
        # Nimmt die letzten 3 Nachrichten für den Kontext
        history_messages = history[-3:] if len(history) > 3 else history
        
        # Baut die Nachrichtenliste für das LLM
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=original_query)]
        
        # Ruft das LLM auf
        response = llm.invoke(messages)
        # Überschreibt den Input mit der verbesserten Frage
        state["input"] = response.content
        return state

    # --- Knoten 2: Retrieve (Suchen) ---
    def retrieve(state):
        # Sucht in der Datenbank nach der (umformulierten) Frage
        docs = retriever.invoke(state["input"])
        # Speichert die Ergebnisse im State
        state["context"] = docs
        return state

    # --- Knoten 3: Generate (Antworten) ---
    # Hier wurde der Code exakt an das IPYNB angepasst und korrigiert
    def generate(state):

        docs = state["context"]  # abgerufene Dokumente aus dem Zustand holen
        history = state.get("chat_history", []) #

        # Liste für die Kontext-Blöcke, die an das LLM übergeben werden
        prompt_blocks = []

        for doc in docs:  # Für jedes Dokument einen eigenen Block bauen
            block = "" # leerer String 

            # Textinhalt des Dokuments 
            block += doc.page_content + "\n\n"

            # Alle relevanten Metadaten (Kennzahlen) strukturiert hinzufügen
            block += "Finanzkennzahlen & Metadaten\n"
            block += f"Ticker: {doc.metadata.get('ticker', '')}, Unternehmen: {doc.metadata.get('company', '')}\n"
            block += f"Sektor: {doc.metadata.get('sector', '')}, Industrie: {doc.metadata.get('industry', '')}\n"
            block += f"Marktkapitalisierung: {doc.metadata.get('market_cap', '')}\n"
            block += f"Aktueller Kurs: {doc.metadata.get('current_price', '')}, Vortagesschluss: {doc.metadata.get('previous_close', '')}, 52-Wochen-Hoch: {doc.metadata.get('52_week_high', '')}\n"
            block += f"KGV (PE Ratio): {doc.metadata.get('pe_ratio', '')}, KGV (Forward PE): {doc.metadata.get('forward_pe', '')}\n"
            block += f"Dividendenrendite: {doc.metadata.get('dividend_yield', '')}, Kurs-Buchwert-Verhältnis (PB): {doc.metadata.get('price_to_book', '')}\n"
            block += f"Gesamtumsatz: {doc.metadata.get('total_revenue', '')}, Verschuldungsgrad (Debt/Equity): {doc.metadata.get('debt_to_equity', '')}\n"
            block += f"Eigenkapitalrendite (ROE): {doc.metadata.get('roe', '')}, 1-Jahres-Rendite: {doc.metadata.get('return_1y', '')}, durchschnittliche monatl. Rendite der letzten 5 jahre: {doc.metadata.get('avg_monthly_return', '')}\n"
            block += f"Volatilität: {doc.metadata.get('volatility', '')}, Link zur Webseite: {doc.metadata.get('website', '')}\n\n"
            block += f"Nachrichten Titel: {doc.metadata.get('latest_news_title', '')}, Bewertung der News: {doc.metadata.get('sentiment', '')}, Link zur News: {doc.metadata.get('news_link', '')}\n"
            block += f"Inhalt: {doc.metadata.get('news_summary', '')}\n"

            prompt_blocks.append(block)  # kompletten Block in die Liste packen

        # finalen Promptkontext bauen
        context = "\n\n".join(prompt_blocks) # Liste von Strings verbinden

        # System Prompt definieren
        system_msg = SystemMessage(content="You are a helpful financial assistant who answers questions based on the given context. Analyze all financial data and metadata carefully. Bei den Kennzahlen handelt es sich um EURO")
        
        # Aktueller Input mit dem gefundenen Kontext zusammenbauen
        current_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['input']}")
        
        # Nachrichtenliste für den finalen Aufruf: System -> Verlauf -> Frage mit Kontext
        messages = [system_msg] + history + [current_msg]

        # LLM Aufruf
        answer = llm.invoke(messages)
        
        # Antwort im State speichern
        state["answer"] = answer.content
        return state

    # --- Graph Definition (Zusammenbau) ---
    rag_graph = (
        StateGraph(RAGState)
        .add_node("reformulate", reformulate_query) # Knoten hinzufügen
        .add_node("retrieve", retrieve)             # Knoten hinzufügen
        .add_node("generate", generate)             # Knoten hinzufügen
        .set_entry_point("reformulate")             # Startpunkt festlegen
        .add_edge("reformulate", "retrieve")        # Verbindung: Erst umschreiben, dann suchen
        .add_edge("retrieve", "generate")           # Verbindung: Erst suchen, dann antworten
        .add_edge("generate", END)                  # Ende
        .compile()                                  # Graph kompilieren
    )
    
    return rag_graph # Gibt den fertigen Graphen zurück

# --- Main UI (Hauptprogramm) ---
st.title("📈 NASDAQ Financial Analyst AI (Full Memory)")

# Initialisiert das RAG System (wird gecached)
rag_app = initialize_rag_system()

# Initialisiert den Session State für Nachrichten, falls noch nicht vorhanden
if "messages" not in st.session_state:
    st.session_state.messages = []

# Zeigt den bisherigen Nachrichtenverlauf im Chat-Fenster an
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Eingabefeld für den Nutzer (unten am Bildschirm)
if prompt := st.chat_input("Stelle ein Frage z.B. analysiere Apple"):
    
    # Fügt die Nutzernachricht zum Verlauf hinzu & zeigt sie an
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Bereitet den Verlauf für LangChain vor (wandelt Dictionaries in Message-Objekte um)
    history_langchain = []
    # Nimmt alle Nachrichten außer der allerletzten (die kommt ja gleich in den Prompt)
    for msg in st.session_state.messages[:-1]: 
        if msg["role"] == "user":
            history_langchain.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain.append(AIMessage(content=msg["content"]))

    # Führt die KI-Logik aus
    with st.chat_message("assistant"):
        with st.spinner("Analysiere..."): # Zeigt Lade-Animation
            # Baut den Input für den Graphen
            inputs = {"input": prompt, "chat_history": history_langchain}
            
            # Ruft den LangGraph auf (.invoke startet den Workflow)
            result = rag_app.invoke(inputs)
            
            # Holt die Antwort aus dem Ergebnis
            answer_text = result["answer"]
            
            # Zeigt die Antwort an
            st.markdown(answer_text)
            
            # Speichert die Antwort im Session State Verlauf
            st.session_state.messages.append({"role": "assistant", "content": answer_text})
