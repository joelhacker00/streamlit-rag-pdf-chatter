import chromadb
import uuid
# Ta bort onödig streamlit-import
# import streamlit as st 
from chromadb.utils import embedding_functions
import os # Behövs för att definiera sökväg

# Importera count_tokens för att beräkna metadata
from chunker import count_tokens

# NYTT: Importer för direkt körning
import tkinter as tk
from tkinter import filedialog
from pdf_extractor import extract_text_from_pdf_pages
from chunker import create_chunks_with_metadata

# --- ChromaDB Funktioner ---

def add_chunks_to_chroma(collection, chunks_with_metadata: list[tuple[str, dict]]):
    """
    Lägger till chunks och deras metadata i en befintlig ChromaDB-samling.
    Skapar embeddings automatiskt via samlingens embedding function.

    Args:
        collection: ChromaDB collection-objektet.
        chunks_with_metadata: Lista av (chunk_text, metadata_dict).

    Returns:
        Antal chunks som lades till.
    """
    if not chunks_with_metadata:
        print("Inga chunks att lägga till.")
        return 0

    chunk_texts = [text for text, _ in chunks_with_metadata]
    metadatas = [meta for _, meta in chunks_with_metadata]

    # Lägg till ytterligare metadata automatiskt
    for i, meta in enumerate(metadatas):
        meta["chunk_index_in_batch"] = i
        meta["token_count"] = count_tokens(chunk_texts[i])

    # Skapa unika ID:n
    ids = [str(uuid.uuid4()) for _ in range(len(chunk_texts))]

    print(f"Lägger till {len(chunk_texts)} chunks i ChromaDB-samlingen '{collection.name}'...")
    try:
        collection.add(
            documents=chunk_texts,
            ids=ids,
            metadatas=metadatas
        )
        print(f"{len(chunk_texts)} chunks lades till framgångsrikt.")
        return len(chunk_texts)
    except Exception as e:
        print(f"Fel vid tillägg av chunks till ChromaDB: {e}")
        return 0

# Flyttad från app.py - kräver EMBEDDING_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME som argument
# @st.cache_resource - Kan inte cachas här om den ska användas i olika kontexter.
# Vi tar bort cachning för nu, app.py kan välja att cacha anropet.
def get_chroma_collection(chroma_db_path: str, collection_name: str, embedding_model_name: str):
    """Skapar/Hämtar ChromaDB-klient och samling."""
    print(f"Försöker skapa/ladda ChromaDB på: {chroma_db_path}")
    try:
        # Se till att embedding-funktionen använder rätt modellnamn
        embedding_function_chroma = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name
        )
        chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function_chroma
        )
        print(f"ChromaDB-samling '{collection_name}' laddad/skapad.")
        return chroma_client, collection
    except Exception as e:
        print(f"Fel med ChromaDB ({collection_name} på {chroma_db_path}): {e}")
        # Stoppa inte appen helt, returnera None
        # st.error(f"Kunde inte skapa/ladda ChromaDB-samling: {e}") # Ta bort Streamlit här
        return None, None 

# --- NYTT: Kod som körs endast när filen exekveras direkt --- 

if __name__ == "__main__":

    # Konstanter för testkörning
    # NYTT: Skapa sökväg i samma mapp som skriptet
    script_dir = os.path.dirname(os.path.abspath(__file__))
    TEST_CHROMA_DB_PATH = os.path.join(script_dir, "chroma_db_test")
    TEST_COLLECTION_NAME = "pdf_test_collection"
    TEST_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

    def select_pdf_file_for_test():
        """Öppnar ett filvalsfönster för testning."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Välj PDF för ChromaDB-test",
            filetypes=[("PDF-filer", "*.pdf")]
        )
        root.destroy()
        return file_path

    print("--- Testkörning av chroma_handler.py ---")
    selected_file = select_pdf_file_for_test()

    if selected_file:
        print(f"\nVald fil för test: {selected_file}")

        # Steg 1: Extrahera text
        print("\nSteg 1: Extraherar text...")
        extracted_data = extract_text_from_pdf_pages(selected_file)
        if not extracted_data:
            print("Kunde inte extrahera text. Avbryter.")
            exit()
        print("Text extraherad.")

        # Steg 2: Skapa chunks
        print("\nSteg 2: Skapar chunks...")
        chunks_data = create_chunks_with_metadata(extracted_data, selected_file)
        if not chunks_data:
            print("Kunde inte skapa chunks. Avbryter.")
            exit()
        print(f"{len(chunks_data)} chunks skapade.")

        # Steg 3: Initiera ChromaDB
        print("\nSteg 3: Initierar ChromaDB...")
        client, collection = get_chroma_collection(
            chroma_db_path=TEST_CHROMA_DB_PATH, 
            collection_name=TEST_COLLECTION_NAME, 
            embedding_model_name=TEST_EMBEDDING_MODEL_NAME
        )
        
        if not collection:
            print("Kunde inte initiera ChromaDB-samling. Avbryter.")
            exit()
        
        initial_count = collection.count()
        print(f"Samling '{TEST_COLLECTION_NAME}' redo. Antal dokument före: {initial_count}")

        # Steg 4: Lägg till chunks
        print("\nSteg 4: Lägger till chunks i ChromaDB...")
        added_count = add_chunks_to_chroma(collection, chunks_data)

        if added_count > 0:
            final_count = collection.count()
            print(f"\nKlart! {added_count} chunks lades till.")
            print(f"Totalt antal dokument i samlingen nu: {final_count}")
        else:
            print("\nInga chunks verkar ha lagts till.")

    else:
        print("\nIngen fil vald för test.")

    print("--- Testkörning klar ---") 