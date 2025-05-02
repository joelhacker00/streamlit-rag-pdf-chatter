import os
import tiktoken
from langchain.text_splitter import TokenTextSplitter
import tkinter as tk
from tkinter import filedialog
from pdf_extractor import extract_text_from_pdf_pages

def count_tokens(text, encoding_name="cl100k_base"):
    """Räkna antalet tokens i en text med hjälp av tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Fel vid räkning av tokens: {e}")
        return 0

def create_chunks_with_metadata(pages_data: list[tuple[int, str]], pdf_path: str, chunk_size: int = 250, chunk_overlap: int = 50) -> list[tuple[str, dict]]:
    """
    Skapar chunks från siddata och lägger till metadata (källa, sidnummer).

    Args:
        pages_data: Lista med (sidnummer, sidtext).
        pdf_path: Sökvägen till den ursprungliga PDF-filen (för metadata).
        chunk_size: Max antal tokens per chunk.
        chunk_overlap: Antal tokens som överlappar mellan chunks.

    Returns:
        En lista av tuples, där varje tuple innehåller (chunk_text, metadata_dict).
    """
    token_splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        encoding_name="cl100k_base",
    )

    all_chunks_with_metadata = []
    pdf_filename = os.path.basename(pdf_path)

    print(f"Chunkar text från {len(pages_data)} sidor för {pdf_filename}...")
    for page_num, page_text in pages_data:
        if not page_text:
            continue

        page_chunks = token_splitter.split_text(page_text)
        for chunk_text in page_chunks:
            chunk_metadata = {
                "source": pdf_filename,
                "page_number": page_num
            }
            all_chunks_with_metadata.append((chunk_text, chunk_metadata))

    print(f"Totalt {len(all_chunks_with_metadata)} chunks skapade från {pdf_filename}.")
    return all_chunks_with_metadata

if __name__ == "__main__":
    
    def select_pdf_file_for_test():
        """Öppnar ett filvalsfönster för testning."""
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Välj PDF för chunking-test",
            filetypes=[("PDF-filer", "*.pdf")]
        )
        root.destroy()
        return file_path

    print("--- Testkörning av chunker.py ---")
    selected_file = select_pdf_file_for_test()

    if selected_file:
        print(f"\nVald fil för test: {selected_file}")
        
        # Steg 1: Extrahera text
        print("\nSteg 1: Extraherar text...")
        extracted_data = extract_text_from_pdf_pages(selected_file)
        
        if not extracted_data:
            print("Kunde inte extrahera text från filen. Avbryter chunking-test.")
        else:
            print(f"Text extraherad från {len(extracted_data)} sidor.")
            
            # NYTT: Beräkna totala tokens för hela texten
            full_text = "\n".join([page_text for _, page_text in extracted_data if page_text])
            total_original_tokens = count_tokens(full_text)
            print(f"Totalt antal tokens i originaltexten: {total_original_tokens}")

            # Steg 2: Skapa chunks
            print("\nSteg 2: Skapar chunks...")
            chunks_data = create_chunks_with_metadata(extracted_data, selected_file)

            if chunks_data:
                # NYTT: Beräkna totala tokens i chunks
                total_chunk_tokens = sum(count_tokens(chunk_text) for chunk_text, _ in chunks_data)
                print(f"Totalt antal tokens i {len(chunks_data)} chunks: {total_chunk_tokens}")
                
                print("\n--- Skapade Chunks (med metadata) ---")
                for i, (chunk_text, metadata) in enumerate(chunks_data):
                    print(f"\n=== Chunk {i+1} Metadata: {metadata} ===")
                    print(chunk_text)
                    print("-" * 40)
                print("\n--- Slut på chunks ---")
            else:
                print("\nInga chunks kunde skapas från texten.")
    else:
        print("\nIngen fil vald för test.")

    print("--- Testkörning klar ---") 