# 📈 Der Nasdaq100 Experte: RAG-Chatbot für Finanzanalysen

Dieses Projekt ist eine interaktive **Streamlit-Webanwendung**, die einen hochentwickelten **RAG (Retrieval-Augmented Generation) Chatbot** bereitstellt. Der Bot agiert als spezialisierter Finanzanalyst und beantwortet Fragen basierend auf aktuellen und historischen Finanzdaten der Unternehmen des **Nasdaq 100 Index**.

Link: https://schelki.streamlit.app/

Die gesamte Logik des Chatbots wird durch den **LangGraph**-State-Machine-Ansatz gesteuert, um einen präzisen, mehrstufigen Analyseprozess zu gewährleisten.

## 🌟 Funktionen und Anwendungsfall

* **Faktengestützte Analyse:** Der Chatbot nutzt eine interne Datenbank (ChromaDB) mit Finanzkennzahlen, Geschäftsberichten und aktuellen Nachrichten der Nasdaq 100-Unternehmen.
* **Historie & Kontext:** Durch die Verwendung von LangGraph kann der Bot den Chatverlauf berücksichtigen und Folgefragen ("Und was ist mit dieser Firma?") korrekt im Kontext beantworten (**Query Reformulation**).
* **Einfache Bedienung:** Intuitive Streamlit-Oberfläche für die Eingabe des OpenAI API-Keys und die direkte Interaktion.
* **Datengrundlage:** Die Analyse basiert auf der Datei `nasdaq_100_final_for_RAG.csv`.

---

## 💻 Technischer Überblick: Die RAG-Pipeline mit LangGraph

Die Anwendung folgt einem mehrstufigen LangGraph-Workflow, um jede Nutzeranfrage zu verarbeiten. Der Prozess ist darauf ausgelegt, die Genauigkeit und Relevanz der generierten Antworten zu maximieren. 

### 1. Initialisierung und Datenaufnahme (Caching)

Beim Start der Anwendung wird die Datenbank aufgebaut und dank `st.cache_resource` im Speicher gehalten, um schnelle Folgeanfragen zu ermöglichen:

1.  **Datenbereinigung:** Die CSV-Datei wird geladen, leere Zellen (`NaN`) werden gefüllt, und die Daten werden in den String-Typ umgewandelt.
2.  **Dokumenterstellung:** Jede Zeile der CSV wird in ein LangChain-`Document`-Objekt umgewandelt. Der Haupttext enthält die Unternehmenszusammenfassung und News, während **alle Finanzkennzahlen** in den **Metadaten** gespeichert werden.
3.  **Vektorisierung:** Die Dokumente werden in kleinere Chunks (Schnipsel) zerteilt (`RecursiveCharacterTextSplitter`), mithilfe von `OpenAIEmbeddings` in Vektoren umgewandelt und in einer **In-Memory ChromaDB** gespeichert.

### 2. LangGraph Workflow (Der Analyseprozess)

Jede Chat-Nachricht durchläuft diesen Graph:

| Knoten | Beschreibung |
| :--- | :--- |
| **`reformulate`** | **Präzisierung der Frage:** Wenn ein Chatverlauf existiert, wird die aktuelle Nutzerfrage unter Berücksichtigung des Verlaufs umgeschrieben (z.B. "Wie ist deren KGV?" $\rightarrow$ "Wie ist das KGV von Apple?"). |
| **`retrieve`** | **Datensuche:** Der LangChain `Retriever` sucht in der ChromaDB nach den **Top 5** relevantesten Dokument-Chunks, die zur umformulierten Frage passen. |
| **`generate`** | **Antwortgenerierung:** Die gefundenen Dokumente (inklusive aller **Finanzkennzahlen aus den Metadaten**) werden zusammen mit der ursprünglichen Frage und dem System-Prompt an das **LLM (GPT-3.5-Turbo)** übergeben. Das LLM generiert die finale, faktenbasierte Antwort. |
| **`END`** | Der Workflow ist abgeschlossen. |

---

## 🛠️ Lokale Installation und Start

### Voraussetzungen

Sie benötigen:
* Python 3.9+
* Einen gültigen **OpenAI API Key** (da die Modelle `text-embedding-3-small` und `gpt-3.5-turbo` verwendet werden).

### Einrichtung

1.  **Repository klonen:**
    ```bash
    git clone [IHR_REPO_LINK]
    cd [PROJEKT-ORDNER]
    ```

2.  **Abhängigkeiten installieren:**
    ```bash
    pip install streamlit pandas openai langchain langchain-openai langchain-community langchain-core langgraph
    ```

3.  **Datenbankdatei:**
    Stellen Sie sicher, dass die CSV-Datei mit den Finanzdaten im Hauptverzeichnis des Projekts liegt:
    ```
    nasdaq_100_final_for_RAG.csv
    ```

4.  **Anwendung starten:**
    ```bash
    streamlit run [DATEINAME_DES_SKRIPTS].py
    ```

Nach dem Start wird die Anwendung im Browser geöffnet und fordert Sie in der Seitenleiste zur Eingabe Ihres **OpenAI API Keys** auf.
