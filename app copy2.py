import streamlit as st
import pandas as pd
import os
from typing import TypedDict, List, Any
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# Überschriften
st.set_page_config(page_title="Der Nasdaq Experte", page_icon="📈", layout="wide")

# Sidebar & Setup
with st.sidebar:
    st.title("⚙️ Einstellungen")
    
    api_key = st.text_input("OpenAI API Key", type="password", help="Gib hier deinen OpenAI API Key ein.")
    if not api_key:
        st.warning("Bitte gib einen API Key ein, um fortzufahren.")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    
    if st.button("Neuer Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.info("Vollständiges Chat-Memory: Das LLM kennt jetzt den Verlauf.")

# Initialisierung & Caching (RAG Logik)
@st.cache_resource(show_spinner="Lade Daten und erstelle Vektor-Datenbank...")
def initialize_rag_system():
    
    # 1. CSV Laden & Bereinigen (Notebook Logik)
    try:
        df = pd.read_csv("nasdaq_100_final_for_RAG.csv")
    except FileNotFoundError:
        st.error("Die Datei 'nasdaq_100_final_for_RAG.csv' wurde nicht gefunden.")
        st.stop()

    if "PEG Ratio" in df.columns:
        df = df.drop(columns=["PEG Ratio"])
    df = df.fillna("") 
    df = df.astype(str)

    # 2. Dokumente erstellen (Logik aus meinem IPYNB)
    def row_to_document(row):
        text = f"""
        Company: {row["Company"]}  
        Ticker: {row["Ticker"]}

        Business Summary: {row["Long Business Summary"]}

        Sector: {row["Sector (Yahoo)"]}
        Industry: {row["Industry (Yahoo)"]}
        Country: {row["Country"]}

        Latest News:
        Title: {row["Latest_News_Title"]}
        Sentiment: {row["Sentiment"]}
        """
    
        metadata = {
            "ticker": row["Ticker"],
            "company": row["Company"],
            "sector": row["Sector (Yahoo)"],
            "industry": row["Industry (Yahoo)"],
            "country": row["Country"],
            "market_cap": row["Market Cap"],
            "current_price": row["Current Price"],
            "previous_close": row["Previous Close"],
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
            "website": row["Website"], 
        }
        return Document(page_content=text, metadata=metadata)

    docs = [row_to_document(row) for i, row in df.iterrows()]

    # 3. Splitting (IPYNB Logik)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    for doc in docs:
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))

    # 4. Vector Store (IPYNB Logik)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(documents=chunked_docs, embedding=embeddings, collection_name="nasdaq_docs_memory_full")
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # 5. LLM & Graph Definition
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    class RAGState(TypedDict):
        input: str
        chat_history: List[Any] # Liste von Messages für das Gedächtnis
        context: List[Document]
        answer: str

    # --- Query Reformulation (Damit die Suche weiß, worum es geht) ---
    def reformulate_query(state):
        history = state.get("chat_history", [])
        original_query = state["input"]

        if not history:
            return state

        # Kurzer Check mit LLM, um "es", "sie", "das" aufzulösen für die Suche
        system_prompt = "Formuliere die Frage so um, dass sie ohne Vorwissen verständlich ist. Ersetze Pronomen durch Namen. Gib NUR die Frage zurück."
        
        # Letzte paar Nachrichten für den Kontext der Umformulierung
        history_messages = history[-3:] if len(history) > 3 else history
        
        messages = [SystemMessage(content=system_prompt)] + history_messages + [HumanMessage(content=original_query)]
        
        response = llm.invoke(messages)
        state["input"] = response.content
        return state

    def retrieve(state):
        docs = retriever.invoke(state["input"])
        state["context"] = docs
        return state

    def generate(state):
        docs = state["context"]
        prompt_blocks = []

        # Kontext bauen wie im Notebook
        for doc in docs:
            block = ""
            block += doc.page_content + "\n\n"
            block += "Finanzkennzahlen & Metadaten\n"
            block += f"Ticker: {doc.metadata.get('ticker', '')}, Unternehmen: {doc.metadata.get('company', '')}\n"
            block += f"Sektor: {doc.metadata.get('sector', '')}, Industrie: {doc.metadata.get('industry', '')}\n"
            block += f"Marktkapitalisierung: {doc.metadata.get('market_cap', '')}\n"
            block += f"Aktueller Kurs: {doc.metadata.get('current_price', '')}, Vortagesschluss: {doc.metadata.get('previous_close', '')}\n"
            block += f"KGV (PE Ratio): {doc.metadata.get('pe_ratio', '')}, KGV (Forward PE): {doc.metadata.get('forward_pe', '')}\n"
            block += f"Dividendenrendite: {doc.metadata.get('dividend_yield', '')}, Kurs-Buchwert-Verhältnis (PB): {doc.metadata.get('price_to_book', '')}\n"
            block += f"Gesamtumsatz: {doc.metadata.get('total_revenue', '')}, Verschuldungsgrad (Debt/Equity): {doc.metadata.get('debt_to_equity', '')}\n"
            block += f"Eigenkapitalrendite (ROE): {doc.metadata.get('roe', '')}, 1-Jahres-Rendite: {doc.metadata.get('return_1y', '')}\n"
            block += f"Volatilität: {doc.metadata.get('volatility', '')}, Link zur Webseite: {doc.metadata.get('website', '')}\n\n"
            block += f"Nachrichten Titel: {doc.metadata.get('latest_news_title', '')}, Medienstimmung diesbezüglich: {doc.metadata.get('sentiment', '')}, Link zur News: {doc.metadata.get('news_link', '')}\n"
            prompt_blocks.append(block)

        context_text = "\n\n".join(prompt_blocks)

        # --- HIER IST DIE ÄNDERUNG FÜR DAS GEDÄCHTNIS ---
        # Wir bauen die Message-Liste: System -> Chat Verlauf -> Aktuelle Frage mit Kontext
        
        system_msg = SystemMessage(content="You are a helpful financial assistant who answers questions based on the given context. Analyze all financial data and metadata carefully. Bei den Kennzahlen handelt es sich um EURO")
        
        # Verlauf holen
        history = state.get("chat_history", [])
        
        # Aktueller Input mit dem gefundenen Kontext (Notebook Logik für den Prompt)
        current_msg = HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {state['input']}")
        
        # Alles zusammenfügen
        messages = [system_msg] + history + [current_msg]

        # LLM aufrufen
        answer = llm.invoke(messages)
        state["answer"] = answer.content
        return state

    # Graph Definition
    rag_graph = (
        StateGraph(RAGState)
        .add_node("reformulate", reformulate_query)
        .add_node("retrieve", retrieve)
        .add_node("generate", generate)
        .set_entry_point("reformulate")
        .add_edge("reformulate", "retrieve")
        .add_edge("retrieve", "generate")
        .add_edge("generate", END)
        .compile()
    )
    return rag_graph

# --- Main UI ---
st.title("📈 NASDAQ Financial Analyst AI (Full Memory)")

# Initialisiere RAG
rag_app = initialize_rag_system()

# Session State für Nachrichten
if "messages" not in st.session_state:
    st.session_state.messages = []

# Verlauf anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Stelle ein Frage z.B. analysiere Apple"):
    
    # User Message UI & State
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # History für LangChain vorbereiten (Liste von Message Objekten)
    history_langchain = []
    for msg in st.session_state.messages[:-1]: # Alle außer der aktuellen (die kommt mit Context in den Prompt)
        if msg["role"] == "user":
            history_langchain.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history_langchain.append(AIMessage(content=msg["content"]))

    # App ausführen
    with st.chat_message("assistant"):
        with st.spinner("Analysiere..."):
            try:
                # History übergeben
                inputs = {"input": prompt, "chat_history": history_langchain}
                result = rag_app.invoke(inputs)
                
                answer_text = result["answer"]
                st.markdown(answer_text)
                
                st.session_state.messages.append({"role": "assistant", "content": answer_text})
                
            except Exception as e:
                st.error(f"Fehler: {e}")