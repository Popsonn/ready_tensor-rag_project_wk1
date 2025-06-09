import fitz

def pdf_to_text(pdf_path, txt_path):
    doc = fitz.open(pdf_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        for page in doc:
            text = page.get_text()
            f.write(text)
    print(f"Text extracted and saved to {txt_path}")
    

pdf_path = r"C:\Users\hp\Downloads\Ready Tensor Project\module1 Project\Lehninger - RAG.pdf"
output_txt_path = r"C:\Users\hp\Downloads\Ready Tensor Project\module1 Project\Data\cleaned_data.txt"
    
pdf_to_text(pdf_path, output_txt_path)
