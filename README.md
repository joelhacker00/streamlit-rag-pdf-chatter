# Chatta med dina PDFer - Streamlit RAG App

Detta är en Streamlit-applikation som implementerar en RAG (Retrieval-Augmented Generation)-pipeline. Den låter dig ladda upp PDF-dokument, bearbeta dem, och sedan ställa frågor mot innehållet via ett chattgränssnitt. Appen kan använda antingen Google Gemini (via API) eller en lokal Ollama-modell för att generera svar baserade på informationen i dina dokument.

## Funktioner

*   **PDF-uppladdning:** Ladda upp en eller flera PDF-filer via gränssnittet.
*   **Textextraktion & Chunking:** Extraherar automatiskt text från PDF:er och delar upp den i hanterbara text-chunks.
*   **Embedding & Vektorlagring:** Skapar text-embeddings (med `intfloat/multilingual-e5-large-instruct`) och lagrar dem i en lokal ChromaDB-vektordatabas.
*   **RAG-baserad Chatt:** Ställ frågor på naturligt språk. Appen hämtar relevanta text-chunks från databasen och använder dem tillsammans med din fråga för att generera ett svar med en vald LLM.
*   **Valbar LLM:** Växla mellan att använda Google Gemini (kräver API-nyckel) eller en lokalt körd Ollama-modell (t.ex. `gemma3:4b`). Appen känner av om internetanslutning finns och anpassar valen.
*   **Databashantering:** Möjlighet att se antal dokument i databasen och rensa hela databasen via gränssnittet.

## Installation

1.  **Klona Repositoryt:**
    ```bash
    git clone <URL-till-ditt-github-repo>
    cd <repo-namn>/streamlit_app 
    ```

2.  **Skapa och Aktivera Virtuell Miljö (Rekommenderas):**
    ```bash
    # Exempel med venv
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate 
    ```

3.  **Installera Beroenden:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Notera: Om du får problem relaterade till `tiktoken`, kan du behöva installera det separat: `pip install tiktoken`)*

## Konfiguration

1.  **Google API Nyckel:**
    *   För att använda Google Gemini behöver du en API-nyckel.
    *   Gå till [Google AI Studio](https://aistudio.google.com/app/apikey) för att skapa en nyckel.
    *   Skapa en fil som heter `.env` i roten av `streamlit_app`-mappen (du kan kopiera och döpa om `.env.example` som finns i mappen).
    *   Lägg till din API-nyckel i `.env`-filen så här:
        ```env
        GOOGLE_API_KEY=DIN_GOOGLE_API_NYCKEL_HÄR
        ```
    *   **VIKTIGT:** Lägg **aldrig** till `.env`-filen i Git! `.gitignore`-filen bör redan förhindra detta.

2.  **Ollama (Valfritt):**
    *   Om du vill använda en lokal modell måste du ha [Ollama](https://ollama.com/) installerat och köra det på din dator.
    *   Se till att du har laddat ner modellen som anges i `app.py` (för närvarande `gemma3:4b`) genom att köra följande kommando i din terminal:
        ```bash
        ollama pull gemma3:4b
        ```
    *   Om du vill använda en annan Ollama-modell, ändra `OLLAMA_MODEL_NAME` i `app.py` och ladda ner den modellen med `ollama pull <modellnamn>`.

## Kör Applikationen

När installation och konfiguration är klar, starta Streamlit-appen från `streamlit_app`-mappen:

```bash
streamlit run app.py
```

Appen bör nu öppnas i din webbläsare.

## Användning

1.  **Ladda upp PDFer:** Använd filuppladdaren i sidofältet för att välja en eller flera PDF-filer.
2.  **Bearbeta Filer:** Klicka på knappen "Bearbeta uppladdade PDFer". Appen kommer att extrahera text, skapa chunks, generera embeddings och lagra dem i ChromaDB. Detta kan ta en stund beroende på filernas storlek och antal.
3.  **Välj LLM:** Om du är online och har angett en API-nyckel, kan du välja mellan Gemini och Ollama i sidofältet. Om du är offline eller saknar nyckel, används Ollama automatiskt (om den körs).
4.  **Chatta:** Skriv dina frågor om dokumentens innehåll i chattfönstret längst ner och tryck Enter. Appen använder RAG-processen för att generera ett svar.
5.  **Rensa Databas (Vid behov):** Använd knappen i sidofältet för att ta bort all data från den lokala ChromaDB-instansen.

## Teknologier

*   **Streamlit:** För webbgränssnittet.
*   **ChromaDB:** För lokal vektordatabaslagring.
*   **Sentence Transformers:** För att skapa text-embeddings ([intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct) används i detta projekt).
*   **Langchain:** För att integrera med LLMs och hantera delar av RAG-flödet.
*   **Google Generative AI:** För att använda Gemini LLM.
*   **Ollama:** För att köra lokala LLMs.
*   **PyPDF:** För att extrahera text från PDF-filer. 