import pypdf
# Ta bort oanvänd import
# import os
# NYTT: Importera för filväljare vid direkt körning
import tkinter as tk
from tkinter import filedialog

def extract_text_from_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extraherar text från en PDF-fil, sida för sida.

    Args:
        pdf_path: Sökväg till PDF-filen

    Returns:
        En lista av tuples, där varje tuple innehåller (sidnummer, sidtext).
        Returnerar en tom lista vid fel.
    """
    pages_data = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        print(f"Läser {len(reader.pages)} sidor från {pdf_path}...")
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                pages_data.append((i + 1, page_text.strip()))
            else:
                pages_data.append((i + 1, ""))
        return pages_data
    except FileNotFoundError:
        print(f"Fel: Filen hittades inte på sökvägen: {pdf_path}")
        return []
    except Exception as e:
        print(f"Ett fel uppstod vid läsning av PDF '{pdf_path}': {e}")
        return []

# --- NYTT: Kod som körs endast när filen exekveras direkt --- 

if __name__ == "__main__":
    
    def select_pdf_file_for_test():
        """Öppnar ett filvalsfönster för testning."""
        root = tk.Tk()
        root.withdraw() # Dölj huvudfönstret
        root.attributes('-topmost', True)
        
        file_path = filedialog.askopenfilename(
            title="Välj PDF för extraktionstest",
            filetypes=[("PDF-filer", "*.pdf")]
        )
        root.destroy()
        return file_path

    print("--- Testkörning av pdf_extractor.py ---")
    selected_file = select_pdf_file_for_test()

    if selected_file:
        print(f"\nVald fil för test: {selected_file}")
        extracted_data = extract_text_from_pdf_pages(selected_file)
        
        if extracted_data:
            print("\n--- Extraherad text ---")
            for page_num, page_text in extracted_data:
                print(f"\n=== Sida {page_num} ===")
                if page_text:
                    print(page_text)
                else:
                    print("(Ingen text extraherad från denna sida)")
            print("\n--- Slut på extraherad text ---")
        else:
            print("\nKunde inte extrahera någon data från filen.")
    else:
        print("\nIngen fil vald för test.")

    print("--- Testkörning klar ---") 