import requests
from bs4 import BeautifulSoup
import os
import re
import urllib.parse
from time import sleep

# Base URLs and Configuration
BASE_URL = "https://jdih.kemnaker.go.id"
# Menggunakan URL Filter dari user (terlama, berlaku, UU/PP/Perpres)
SEARCH_URL = "https://jdih.kemnaker.go.id/peraturan?keyword=&nomor=&tahun=&jenis%5B0%5D=2&jenis%5B1%5D=4&jenis%5B2%5D=5&jenis%5B3%5D=8&status=berlaku&terjemahan=&sort=terlama&hal={page}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Mapping document types to your folder structure
FOLDER_MAPPING = {
    "Undang-undang": "data/pdf/undang-undang",
    "Peraturan Pemerintah": "data/pdf/peraturan-pemerintah",
    "Peraturan Presiden": "data/pdf/peraturan-presiden"
}

def ensure_folders():
    for folder in FOLDER_MAPPING.values():
        os.makedirs(folder, exist_ok=True)

def download_file(url, folder, filename):
    filepath = os.path.join(folder, filename)
    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return True
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_total_pages():
    # If standard pagination is hard to parse, you can run until no more items are found or set a fixed number
    # Currently setting to scrape until an empty list is encountered
    pass

def scrape_pages(start_page=1, max_pages=50):
    ensure_folders()
    
    for page in range(start_page, max_pages + 1):
        print(f"\n--- Scraping Page {page} ---")
        url = SEARCH_URL.format(page=page)
        
        try:
            res = requests.get(url, headers=HEADERS)
            res.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            break
            
        soup = BeautifulSoup(res.text, "html.parser")
        
        # Searching for card/items that contain the regulation info.
        # Note: You may need to inspect the HTML and adjust the selectors here
        # E.g. find all links starting with /peraturan/
        item_links = []
        for a_tag in soup.find_all('a', href=True):
            if '/katalog/' in a_tag['href'] or '/peraturan/' in a_tag['href']:
                if a_tag['href'] not in item_links and 'sort=' not in a_tag['href']:
                    item_links.append(a_tag['href'])
                    
        if not item_links:
            print("No items found on this page or end of pages. Exiting.")
            break
            
        print(f"Found {len(item_links)} regulation links on page {page}.")
        
        for link in item_links:
            detail_url = urllib.parse.urljoin(BASE_URL, link)
            scrape_detail_page(detail_url)
            sleep(1) # Polite scraping delay

def scrape_detail_page(detail_url):
    try:
        res = requests.get(detail_url, headers=HEADERS)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching detail {detail_url}: {e}")
        return
        
    soup = BeautifulSoup(res.text, "html.parser")
    
    # Try to find document type to map to the correct folder
    # This usually appears as breadcrumbs or metadata (e.g., "Undang-undang")
    page_text = soup.get_text()
    folder_path = None
    
    title = soup.find('h1') or soup.find('h2')
    if title:
        title_text = title.get_text().strip().lower()
        # FILTER JUDUL (Mencegah Amandemen/Perubahan/Pembentukan dll)
        exclusions = [
            "perubahan atas",
            "perubahan undang",
            "perubahan pertama",
            "perubahan kedua",
            "penetapan peraturan pemerintah pengganti",
            "pembentukan provinsi",
            "pembentukan daerah"
        ]
        
        if any(ex in title_text for ex in exclusions):
            print(f"Skipping (Filter Excluded): {title_text}")
            return

    for doc_type, path in FOLDER_MAPPING.items():
        if doc_type.lower() in page_text.lower():
            folder_path = path
            break
            
    if not folder_path:
        print(f"Could not determine document type for {detail_url}. Skipping.")
        return
        
    # Find PDF download link from <embed> tag or data-pdf-url attribute
    pdf_link = None
    embed_tag = soup.find('embed', type='application/pdf')
    if embed_tag and 'src' in embed_tag.attrs:
        pdf_link = embed_tag['src']
    else:
        # Fallback to div data-pdf-url
        div_viewer = soup.find('div', id='pdf-viewer')
        if div_viewer and 'data-pdf-url' in div_viewer.attrs:
            pdf_link = div_viewer['data-pdf-url']
            
    if pdf_link and not pdf_link.startswith('http'):
        pdf_link = urllib.parse.urljoin(BASE_URL, pdf_link)
            
    if pdf_link:
        # Extract filename (last part of URL or a cleaned version of the title)
        filename = pdf_link.split('/')[-1]
        if not filename.endswith('.pdf'):
            # Fallback to a sanitized url if it's dynamic
            # In jdih, download links might look like /download/1234
            title = soup.find('h1') or soup.find('h2')
            if title:
                safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title.get_text().strip())
                filename = f"{safe_title}.pdf"
            else:
                filename = f"document_{abs(hash(pdf_link))}.pdf"
                
        download_file(pdf_link, folder_path, filename)
    else:
        print(f"No PDF link found on {detail_url}")

if __name__ == "__main__":
    print("Starting JDIH Kemnaker Scraper with Filters...")
    # Asumsikan ada sekitar 100 halaman jika ada ratusan/ribuan peraturan
    scrape_pages(start_page=1, max_pages=100)
    print("Done!")