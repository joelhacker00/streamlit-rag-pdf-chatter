import google.generativeai as genai
import os
# NYTT: Type hinting och Langchain LLM
from typing import Union, Any
from langchain_core.language_models.llms import BaseLLM 

# NYTT: Importer för direkt körning
import tkinter as tk
from tkinter import filedialog
from dotenv import load_dotenv
import pdf_extractor
import chunker
import chroma_handler

# Endast RAG-specifika funktioner kvar här

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Formatera en sökfråga med instruktion enligt embedding-modellens format."""
    return f'Instruct: {task_description}\nQuery: {query}'

# UPPDATERAD: Hanterar källhänvisning olika för olika modeller
def generate_rag_response(query: str, collection, llm_model: Union[genai.GenerativeModel, BaseLLM, Any]):
    """
    Genererar ett svar från LLM baserat på hämtad kontext från ChromaDB.
    Hanterar källhänvisning olika beroende på modelltyp.
    """
    if not collection:
        return "Fel: ChromaDB-samling är inte tillgänglig."
    if not llm_model:
        return "Fel: LLM-modell är inte tillgänglig."

    # 1. Förbered sökfrågan
    search_task = "Given a document, retrieve relevant passages that address the following information need"
    instructed_query = get_detailed_instruct(search_task, query)

    # 2. Hämta kontext
    print(f"\nHämtar relevanta chunks för frågan: '{query}'...")
    try:
        results = collection.query(
            query_texts=[instructed_query],
            n_results=5,
            include=['documents', 'metadatas']
        )
    except Exception as e:
        print(f"Fel vid query mot ChromaDB: {e}")
        return "Ett fel uppstod vid hämtning av kontext från databasen."

    retrieved_documents = results.get('documents', [[]])[0]
    retrieved_metadatas = results.get('metadatas', [[]])[0]

    # NYTT: Extrahera unika källor direkt för eventuell manuell tilläggning
    unique_sources = []
    if retrieved_metadatas:
        temp_sources = set()
        for meta in retrieved_metadatas:
            source_file = meta.get('source', 'N/A')
            page_num = meta.get('page_number', 'N/A')
            if source_file != 'N/A' and page_num != 'N/A':
                 temp_sources.add((source_file, page_num))
        # Sortera för konsekvent ordning
        unique_sources = sorted(list(temp_sources))

    # Bygg kontext för LLM
    if not retrieved_documents:
        print("Kunde inte hitta relevanta chunks.")
        context = "Ingen specifik kontext hittades i dokumenten för denna fråga."
    else:
        print(f"\nHittade {len(retrieved_documents)} relevanta chunks:")
        context_chunks_for_llm = []
        for i, (doc, meta) in enumerate(zip(retrieved_documents, retrieved_metadatas)):
            page_num = meta.get('page_number', 'N/A')
            source_file = meta.get('source', 'N/A')
            print(f"  - Chunk {i+1} (Från: {source_file}, Sida: {page_num})")
            chunk_with_source = f"--- Chunk {i+1} (Källa: '{source_file}', Sida: {page_num}) ---\n{doc}"
            context_chunks_for_llm.append(chunk_with_source)
        context = "\n\n".join(context_chunks_for_llm)

    # 4. Skapa prompten - TVÅ versioner
    prompt_template_base = """**Instruktioner:**
1.  Svara på frågan nedan ENDAST baserat på den givna kontexten. Om ingen relevant kontext finns, säg det.
2.  Svara ALLTID på **samma språk** som frågan är ställd på.
3.  Citera relevanta delar från kontexten om det är lämpligt.
4.  Om kontexten inte innehåller tillräcklig information för att svara, skriv endast: 'Kontexten innehåller inte tillräcklig information för att svara på frågan.'
{source_instruction}
**Kontext:**
{context}

**Fråga:**
{query}

**Svar:**
"""

    gemini_source_instruction = "5.  **VIKTIGT:** Avsluta ALLTID ditt svar med en källhänvisning. Titta på källinformationen (filnamn och sidnummer) som anges för varje chunk i kontexten ovan. Baserat på de chunks du använde för att formulera svaret, skapa en mening som liknar: 'Informationen baseras på [Filnamn] (sida [nummer]) och [Annat filnamn] (sida [nummer]).' Lista endast de källor (fil och sida) som var relevanta för ditt svar. Om ingen specifik chunk användes eller kunde användas, skriv: 'Generell information baserad på tillgängliga dokument.'" 
    ollama_source_instruction = "" # Ingen instruktion för Ollama att lägga till källa

    # Välj rätt prompt baserat på modelltyp
    if isinstance(llm_model, genai.GenerativeModel):
        prompt = prompt_template_base.format(
            source_instruction=gemini_source_instruction, 
            context=context, 
            query=query
        )
    elif isinstance(llm_model, BaseLLM):
        prompt = prompt_template_base.format(
            source_instruction=ollama_source_instruction, 
            context=context, 
            query=query
        )
    else: # Fallback för okänd typ
        print(f"VARNING: Okänd modelltyp ({type(llm_model)}), använder Gemini-prompt.")
        prompt = prompt_template_base.format(
            source_instruction=gemini_source_instruction, 
            context=context, 
            query=query
        )

    # 5. Generera svar
    print("\nGenererar svar...")
    llm_answer = "Fel: Kunde inte generera svar."
    is_ollama = isinstance(llm_model, BaseLLM) # Kom ihåg om det var Ollama
    try:
        if isinstance(llm_model, genai.GenerativeModel):
            # ... (Google-anrop och felhantering som tidigare) ...
            print(f"Använder Google GenerativeModel ({llm_model.model_name})...")
            response = llm_model.generate_content(prompt)
            if hasattr(response, 'text'):
                 llm_answer = response.text.strip()
            elif hasattr(response, 'parts') and response.parts:
                 llm_answer = " ".join(part.text for part in response.parts if hasattr(part, 'text')).strip()
            elif response.prompt_feedback.block_reason:
                 llm_answer = f"Svaret blockerades: {response.prompt_feedback.block_reason}"
                 print(f"VARNING: Google blockerade svaret - Anledning: {response.prompt_feedback.block_reason}")
            else:
                 llm_answer = "Google LLM returnerade ett oväntat svarformat."
                 print(f"VARNING: Oväntat svarformat från Google LLM: {response}")
        
        elif is_ollama: # Använd is_ollama-flaggan
            # ... (Ollama/Langchain-anrop och felhantering som tidigare) ...
            print(f"Använder Langchain BaseLLM (t.ex. Ollama)...")
            response = llm_model.invoke(prompt)
            if isinstance(response, str):
                llm_answer = response.strip()
            elif hasattr(response, 'content') and isinstance(response.content, str):
                llm_answer = response.content.strip()
            else:
                llm_answer = "Langchain LLM returnerade ett oväntat svarformat."
                print(f"VARNING: Oväntat svarformat från Langchain LLM: {response}")
        
        else:
            llm_answer = "Fel: Okänd typ av LLM-modell för anrop."
            print(f"Fel: Okänd modelltyp för anrop: {type(llm_model)}")

    except Exception as e:
        print(f"Ett fel uppstod vid anrop till LLM: {e}")
        return f"Ett fel uppstod när svaret skulle genereras: {e}"

    # NYTT: Lägg till källhänvisning manuellt för Ollama
    if is_ollama and unique_sources:
        # Formatera källsträngen
        source_strings_list = []
        current_file = None
        page_list = []
        for file, page in unique_sources:
            if file != current_file and current_file is not None:
                source_strings_list.append(f"'{current_file}' (sida { ', '.join(map(str, sorted(list(set(page_list))))) })")
                page_list = []
            current_file = file
            page_list.append(page)
        # Lägg till sista filen
        if current_file is not None:
             source_strings_list.append(f"'{current_file}' (sida { ', '.join(map(str, sorted(list(set(page_list))))) })")
        
        if source_strings_list:
             source_ref_text = f"\n\n*Svaret är baserat på information från { ' och '.join(source_strings_list) }.*"
             # Lägg bara till om svaret inte redan är ett felmeddelande
             if not llm_answer.startswith("Fel:") and not llm_answer.startswith("Langchain LLM returnerade"):
                  llm_answer += source_ref_text

    return llm_answer

# --- NYTT: Kod som körs endast när filen exekveras direkt --- 

if __name__ == "__main__":

    # Ladda API-nyckel
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    # Konstanter för testkörning
    script_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_CHROMA_DB_PATH = os.path.join(script_dir, "chroma_db_test_rag") # Ny mapp för denna test
    TEST_COLLECTION_NAME = "pdf_rag_test_collection"
    TEST_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
    TEST_GEMINI_MODEL_NAME = 'gemini-2.0-flash'
    # NYTT: Ollama-modell för test
    TEST_OLLAMA_MODEL_NAME = 'gemma3:latest'

    def select_pdf_file_for_test():
        """Öppnar ett filvalsfönster för testning."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Välj PDF för RAG-test",
            filetypes=[("PDF-filer", "*.pdf")]
        )
        root.destroy()
        return file_path

    print("--- Testkörning av rag_utils.py ---")
    selected_file = select_pdf_file_for_test()

    if not selected_file:
        print("Ingen fil vald. Avbryter.")
        exit()

    print(f"\nVald fil för test: {selected_file}")

    # Steg 1: Extrahera text
    print("\nSteg 1: Extraherar text...")
    extracted_data = pdf_extractor.extract_text_from_pdf_pages(selected_file)
    if not extracted_data:
        print("Kunde inte extrahera text. Avbryter.")
        exit()
    print("Text extraherad.")

    # Steg 2: Skapa chunks
    print("\nSteg 2: Skapar chunks...")
    chunks_data = chunker.create_chunks_with_metadata(extracted_data, selected_file)
    if not chunks_data:
        print("Kunde inte skapa chunks. Avbryter.")
        exit()
    print(f"{len(chunks_data)} chunks skapade.")

    # Steg 3: Initiera ChromaDB
    print("\nSteg 3: Initierar ChromaDB...")
    client, collection = chroma_handler.get_chroma_collection(
        chroma_db_path=TEST_CHROMA_DB_PATH, 
        collection_name=TEST_COLLECTION_NAME, 
        embedding_model_name=TEST_EMBEDDING_MODEL_NAME
    )
    if not collection:
        print("Kunde inte initiera ChromaDB-samling. Avbryter.")
        exit()
    print(f"Samling '{TEST_COLLECTION_NAME}' redo.")

    # Steg 4: Lägg till chunks (om de inte redan finns? För enkelhet, lägg till ändå)
    print("\nSteg 4: Lägger till chunks i ChromaDB...")
    added_count = chroma_handler.add_chunks_to_chroma(collection, chunks_data)
    print(f"{added_count} chunks bearbetades.")

    # Steg 5: Initiera BÅDA LLM-modellerna (om möjligt) för test
    print("\nSteg 5: Initierar LLM-modeller...")
    gemini_llm = None
    ollama_llm = None

    # Initiera Gemini
    if not api_key:
        print("VARNING: Google API-nyckel saknas. Kan inte initiera Gemini.")
    else:
        try:
            genai.configure(api_key=api_key)
            gemini_llm = genai.GenerativeModel(TEST_GEMINI_MODEL_NAME)
            _ = gemini_llm.generate_content("Test", generation_config=genai.types.GenerationConfig(candidate_count=1))
            print(f"Gemini-modell ({TEST_GEMINI_MODEL_NAME}) initierad och verifierad.")
        except Exception as e:
            print(f"Kunde inte initiera Gemini: {e}")
    
    # Initiera Ollama (förutsätter att Ollama körs)
    try:
        # NYTT: Import och initiering av Ollama LLM
        from langchain_community.llms import Ollama
        ollama_llm = Ollama(model=TEST_OLLAMA_MODEL_NAME)
        # Snabb verifiering (kan ta en stund första gången)
        _ = ollama_llm.invoke("Berätta en kort historia")
        print(f"Ollama-modell ({TEST_OLLAMA_MODEL_NAME}) initierad och verifierad.")
    except ImportError:
        print(f"VARNING: langchain-community inte installerat? Kunde inte importera Ollama.")
    except Exception as e:
        print(f"Kunde inte initiera Ollama ({TEST_OLLAMA_MODEL_NAME}). Kör Ollama-servern? Fel: {e}")

    # Steg 6: Ställ fråga
    if collection and (gemini_llm or ollama_llm):
        while True:
            user_query = input("\nStäll en fråga (eller skriv 'avsluta'): ")
            if user_query.lower() == 'avsluta':
                break

            # NYTT: Välj vilken modell som ska användas för test
            selected_llm = None
            if gemini_llm and ollama_llm:
                model_choice = input("Använd [G]emini eller [O]llama? (G/O): ").lower()
                if model_choice == 'o':
                    selected_llm = ollama_llm
                else:
                    selected_llm = gemini_llm # Default till Gemini
            elif gemini_llm:
                selected_llm = gemini_llm
            elif ollama_llm:
                selected_llm = ollama_llm
            
            if selected_llm:
                print("\nGenererar svar...")
                final_answer = generate_rag_response(user_query, collection, selected_llm)
                print("\n--- Svar från RAG --- ")
                print(final_answer)
                print("---------------------")
            else:
                print("Ingen LLM tillgänglig för att generera svar.")
                break # Avsluta loopen om ingen modell finns
    else:
        print("\nKan inte ställa frågor eftersom DB eller ingen LLM kunde initieras.")

    print("--- Testkörning klar ---") 