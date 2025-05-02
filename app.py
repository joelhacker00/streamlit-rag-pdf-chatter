import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import chromadb
# Ta bort onödiga importer här
# from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import time
# NYTT: För Ollama
from langchain_community.llms import Ollama
from langchain_core.language_models.llms import BaseLLM
from typing import Union, Any
# NYTT: För online-check
import socket

# Importera funktioner från våra *nya* utils-filer
import pdf_extractor
import chunker
import chroma_handler
import rag_utils # Behövs fortfarande för generate_rag_response

# --- Konfiguration och Initialisering ---

# Ladda API-nyckel från .env-fil
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Streamlit sidkonfiguration
st.set_page_config(page_title="Chatta med dina PDFer", layout="wide")
st.title("📄 Chatta med dina PDFer")

# Konstanter
CHROMA_DB_PATH = "./chroma_db_streamlit"
COLLECTION_NAME = "pdf_rag_collection"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
# Modellnamn för val
GEMINI_MODEL_NAME = 'gemini-2.0-flash'
OLLAMA_MODEL_NAME = 'gemma3:latest' # Byt om du vill använda annan Ollama-modell

# --- Funktioner för Streamlit-appen ---

# NYTT: Funktion för att kolla internetstatus
def check_online_status(host="8.8.8.8", port=53, timeout=3):
    """Försöker ansluta till en extern server för att kolla internetstatus."""
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        print("Internetanslutning hittad.")
        return True
    except socket.error as ex:
        print(f"Ingen internetanslutning hittad: {ex}")
        return False

# Kör online-check tidigt
is_online = check_online_status()

@st.cache_resource
def load_embedding_model():
    """Laddar embedding-modellen."""
    print("Laddar embedding-modellen...")
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding-modell laddad.")
        return model
    except Exception as e:
        st.error(f"Kunde inte ladda embedding-modellen: {e}")
        return None

# Modifierad: Laddar Gemini om API-nyckel finns
@st.cache_resource
def load_gemini_model():
    """Initierar Gemini LLM-modellen."""
    if not api_key:
        print("API-nyckel saknas för Gemini.") # Mindre synligt i sidebar
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"{GEMINI_MODEL_NAME} LLM-modell initierad.")
        _ = model.generate_content("Test", generation_config=genai.types.GenerationConfig(candidate_count=1))
        print("Gemini-modell verifierad.")
        return model
    except Exception as e:
        # Visa inte error i UI om vi är offline och inte ens försökte
        # if is_online: 
        #     st.error(f"Kunde inte initiera Gemini LLM-modellen: {e}")
        print(f"Fel vid initiering av Gemini: {e}")
        return None

# NYTT: Laddar Ollama-modell
@st.cache_resource
def load_ollama_model():
    """Initierar Ollama LLM-modellen."""
    try:
        ollama_llm = Ollama(model=OLLAMA_MODEL_NAME)
        # Verifiera genom att göra ett kort anrop
        print(f"Verifierar Ollama-modell ({OLLAMA_MODEL_NAME})...")
        _ = ollama_llm.invoke("Hej")
        print(f"Ollama-modell ({OLLAMA_MODEL_NAME}) initierad och verifierad.")
        return ollama_llm
    except ImportError:
        st.error(f"Fel: `langchain-community` inte installerat? Kör `pip install langchain-community`")
        print("Fel: langchain-community ej installerat för Ollama.")
        return None
    except Exception as e:
        # Vanligt fel: Ollama-servern körs inte
        st.error(f"Kunde inte ansluta till Ollama. Kör Ollama-servern? (Modell: {OLLAMA_MODEL_NAME}) Fel: {e}")
        print(f"Fel vid initiering av Ollama: {e}")
        return None

# Funktion för att processa uppladdade filer (ANVÄNDER NYA IMPORTER)
def process_uploaded_files(uploaded_files, collection):
    """Bearbetar uppladdade PDF-filer och lägger till dem i ChromaDB."""
    if not uploaded_files:
        st.warning("Inga filer valda.")
        return
    if not collection:
        st.error("ChromaDB-samlingen är inte tillgänglig.")
        return

    total_chunks_added = 0
    start_time = time.time()
    progress_bar = st.progress(0, text="Startar bearbetning...")
    files_processed = 0

    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        progress_text = f"Bearbetar {file_name}..."
        progress_bar.progress(files_processed / len(uploaded_files), text=progress_text)

        # Spara temporär fil
        temp_file_path = os.path.join(".", file_name)
        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write(f"Extraherar text från {file_name}...")
            # Använd pdf_extractor
            pages_data = pdf_extractor.extract_text_from_pdf_pages(temp_file_path)
            if not pages_data:
                st.warning(f"Kunde inte extrahera text från {file_name}. Hoppar över.")
                continue

            st.write(f"Skapar chunks för {file_name}...")
            # Använd chunker
            chunks_with_metadata = chunker.create_chunks_with_metadata(pages_data, temp_file_path)
            if not chunks_with_metadata:
                st.warning(f"Kunde inte skapa chunks från {file_name}. Hoppar över.")
                continue

            st.write(f"Lägger till {len(chunks_with_metadata)} chunks från {file_name} i databasen...")
            # Använd chroma_handler
            added_count = chroma_handler.add_chunks_to_chroma(collection, chunks_with_metadata)
            total_chunks_added += added_count
        
        except Exception as e:
            st.error(f"Ett fel uppstod vid bearbetning av {file_name}: {e}")
        finally:
            # Ta bort temporär fil
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            files_processed += 1
            progress_bar.progress(files_processed / len(uploaded_files), text=progress_text)

    end_time = time.time()
    progress_bar.progress(1.0, text=f"Bearbetning klar! {total_chunks_added} chunks lades till på {end_time - start_time:.2f} sekunder.")
    time.sleep(3)
    progress_bar.empty()
    
    if total_chunks_added > 0:
        st.success(f"Klar! {total_chunks_added} text-chunks har lagts till i vektordatabasen.")
    else:
        st.warning("Inga nya chunks lades till i databasen.")


# --- Streamlit UI --- 

# Ladda modeller och databas
embedding_model = load_embedding_model()
chroma_client, collection = chroma_handler.get_chroma_collection(
    chroma_db_path=CHROMA_DB_PATH, 
    collection_name=COLLECTION_NAME, 
    embedding_model_name=EMBEDDING_MODEL_NAME
)

# --- Sidebar --- 

# NYTT: Visa Online/Offline status
st.sidebar.header("Status")
if is_online:
    st.sidebar.success("🟢 Online", icon="✅")
else:
    st.sidebar.warning("🟠 Offline", icon="⚠️")

st.sidebar.header("LLM-Val")

# NYTT: Villkorlig modellväljare och LLM-laddning
active_llm: Union[genai.GenerativeModel, BaseLLM, None] = None
model_option = None # Initiera

if is_online:
    # Visa radioknapp om online
    model_option = st.sidebar.radio(
        "Välj LLM för svar:",
        ("Gemini 2.0 Flash (Extern)", f"Ollama ({OLLAMA_MODEL_NAME} - Lokal)"),
        index=0 # Förvälj Gemini
    )
    if model_option.startswith("Gemini"):
        active_llm = load_gemini_model()
        if not api_key and active_llm is None:
             st.warning("Google API-nyckel saknas. Kan inte använda Gemini.", icon="⚠️")
    else: # Ollama valt
        active_llm = load_ollama_model()
else:
    # Visa bara meddelande om offline och ladda Ollama
    st.sidebar.info(f"Offline-läge: Använder lokal Ollama ({OLLAMA_MODEL_NAME}).")
    active_llm = load_ollama_model()

st.sidebar.markdown("---")
st.sidebar.header("Ladda upp PDFer")
uploaded_files = st.sidebar.file_uploader(
    "Välj en eller flera PDF-filer", 
    accept_multiple_files=True, 
    type="pdf"
)
process_button = st.sidebar.button("Bearbeta uppladdade PDFer", disabled=(not uploaded_files))

if process_button:
    with st.spinner("Bearbetar PDFer..."):
        process_uploaded_files(uploaded_files, collection)

st.sidebar.markdown("---")
st.sidebar.header("Databasinfo")
if collection:
    try:
        db_count = collection.count()
        st.sidebar.write(f"Antal dokument i databasen: {db_count}")
    except Exception as e:
        st.sidebar.error(f"Kunde inte hämta antal dokument: {e}")
else:
    st.sidebar.write("Databas ej tillgänglig.")

if st.sidebar.button("⚠️ Rensa Hela Databasen", type="secondary"):
    if chroma_client and COLLECTION_NAME:
        try:
            with st.spinner("Rensar databasen..."):
                chroma_client.delete_collection(name=COLLECTION_NAME)
            st.sidebar.success(f"Databasen '{COLLECTION_NAME}' har rensats!")
            time.sleep(2)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Kunde inte rensa databasen: {e}")
    else:
        st.sidebar.warning("Kan inte rensa, databasklient ej tillgänglig.")

# --- Chatt-gränssnitt --- 
st.header("Chatt")

# Initiera chatthistorik
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visa gamla meddelanden
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ta emot ny input
if prompt := st.chat_input("Ställ en fråga om dina dokument..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generera och visa assistentens svar (ANVÄNDER active_llm)
    if not active_llm:
        st.warning("Ingen LLM är tillgänglig (kontrollera val, API-nyckel eller Ollama-status). Kan inte generera svar.")
    elif not collection:
         st.warning("Databasen är inte tillgänglig. Kan inte generera svar.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Tänker..."):
                # Anropa RAG med den *aktiva* LLM:en
                response = rag_utils.generate_rag_response(prompt, collection, active_llm)
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response}) 