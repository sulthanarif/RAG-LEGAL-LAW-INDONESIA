import os
import re
import shutil
import fitz
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

BASE_DIR = os.path.join("data", "pdf")
FOLDERS = {
    "undang-undang": os.path.join(BASE_DIR, "undang-undang"),
    "peraturan pemerintah": os.path.join(BASE_DIR, "peraturan-pemerintah"),
    "peraturan presiden": os.path.join(BASE_DIR, "peraturan-presiden"),
    "peraturan menteri": os.path.join(BASE_DIR, "peraturan-menteri")
}

for folder_path in FOLDERS.values():
    os.makedirs(folder_path, exist_ok=True)

def extract_text_from_pdf(filepath):
    try:
        doc = fitz.open(filepath)
        if len(doc) == 0:
            return ""
        
        page = doc[0]
        text = page.get_text().strip()
        
        if len(text) < 50:
            print(f"Memerlukan pemindaian visual untuk fail {os.path.basename(filepath)}...")
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="ind+eng")
            
        doc.close()
        return text
    except Exception as e:
        print(f"Gagal membaca fail {filepath}: {e}")
        return ""

def identify_and_rename(text):
    text_clean = re.sub(r'\s+', ' ', text).lower()
    
    doc_type = None
    prefix = ""
    
    if "undang-undang" in text_clean and "republik indonesia" in text_clean:
        doc_type = "undang-undang"
        prefix = "UU"
    elif "peraturan pemerintah" in text_clean:
        doc_type = "peraturan pemerintah"
        prefix = "PP"
    elif "peraturan presiden" in text_clean:
        doc_type = "peraturan presiden"
        prefix = "PERPRES"
    elif "peraturan menteri" in text_clean or "keputusan menteri" in text_clean:
        doc_type = "peraturan menteri"
        prefix = "PERMEN"
        
    if not doc_type:
        return None, None

    match = re.search(r'(?:nomor|no\.?)\s*([0-9a-z./-]+)\s*(?:tahun|thn|th\.?)\s*(\d{4})', text_clean)
    
    if match:
        nomor = match.group(1).strip('./-').upper().replace(' ', '_').replace('/', '_')
        tahun = match.group(2)
        
        new_filename = f"{prefix}_Nomor_{nomor}_Tahun_{tahun}.pdf"
        return doc_type, new_filename
        
    return doc_type, None

def process_pdfs():
    for root, _, files in os.walk(BASE_DIR):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
                
            filepath = os.path.join(root, file)
            
            is_in_target_folder = any(root == target for target in FOLDERS.values())
            if is_in_target_folder and re.match(r'^(UU|PP|PERPRES|PERMEN)_Nomor_.*_Tahun_\d{4}\.pdf$', file):
                continue

            print(f"Memproses fail {file}...")
            text = extract_text_from_pdf(filepath)
            
            if not text.strip():
                print(f"Teks tidak ditemukan pada fail {file}")
                continue
                
            doc_type, new_filename = identify_and_rename(text)
            
            if new_filename and doc_type:
                target_folder = FOLDERS[doc_type]
                target_path = os.path.join(target_folder, new_filename)
                
                if filepath == target_path:
                    continue
                    
                counter = 1
                base_name, ext = os.path.splitext(new_filename)
                while os.path.exists(target_path):
                    if filepath == target_path:
                        break
                    target_path = os.path.join(target_folder, f"{base_name}_{counter}{ext}")
                    counter += 1
                
                if filepath != target_path:
                    try:
                        shutil.move(filepath, target_path)
                        print(f"Berhasil merapikan fail ke {target_path}")
                    except Exception as e:
                        print(f"Gagal memindahkan fail {filepath}: {e}")
            else:
                print(f"Pola penamaan gagal terdeteksi pada fail {file}")

if __name__ == "__main__":
    print("Memulai penataan direktori PDF...")
    process_pdfs()
    print("Operasi selesai.")