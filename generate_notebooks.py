import json
import os

def create_notebook(filename, cells_content):
    cells = []
    for cell_type, content in cells_content:
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": [line + "\n" for line in content.split("\n")]
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
        
    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Generated {filename}")

# ==========================================
# BATCH 1: Phase 4, 5, 6
# ==========================================
batch1_cells = [
    ("markdown", "# Batch 1: Phase 4, 5, 6\n## Phase 4: Skema Metadata Final & Validasi\n## Phase 5: Embedding Pipeline (Dense + BM25)\n## Phase 6: Vector Database (ChromaDB + BM25 Local Index)"),
    ("code", """import logging
import json
import os
import pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)"""),
    ("markdown", "### Phase 4: Skema Metadata Final & Validasi"),
    ("code", """def validate_chunk_metadata(meta: dict) -> List[str]:
    errors = []
    # Primary metadata (MUST exist in JSON)
    required_str = ['regulation_type', 'nomor', 'tentang', 'source_file', 'pasal_id']
    required_int = ['year', 'chunk_index']

    for field in required_str:
        if not meta.get(field):
            errors.append(f"Field '{field}' kosong atau None")

    for field in required_int:
        val = meta.get(field)
        if val is None:
            errors.append(f"Field '{field}' is None")

    if meta.get('active_status') not in ('Berlaku', 'Tidak Berlaku', 'Sebagian Berlaku'):
        meta['active_status'] = 'Berlaku' # Auto-fix

    return errors
"""),
    ("markdown", "### Phase 5: Embedding Pipeline (Dense)"),
    ("code", """import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

class LegalEmbeddingPipeline:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_texts(self, texts: List[str], prefix: str = "", batch_size: int = 16) -> List[List[float]]:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = [prefix + t for t in texts[i:i+batch_size]]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
        return all_embeddings

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 16) -> List[Dict]:
        texts = [chunk.get('embedding_text', chunk['text']) for chunk in chunks]
        embeddings = self.embed_texts(texts, prefix="passage: ", batch_size=batch_size)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk['dense_embedding'] = emb
        return chunks

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query], prefix="query: ", batch_size=1)[0]"""),
    ("markdown", "### Phase 6: Vector Database (ChromaDB + Local BM25 Index)"),
    ("code", """import chromadb
import uuid

def setup_chromadb(db_path: str = "./chroma_db", collection_name: str = "hukum_ketenagakerjaan"):
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    logger.info(f"ChromaDB collection '{collection_name}' ready at {db_path}")
    return collection

class BM25Index:
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_ids = []
        
    def build_index(self, chunks: List[Dict], ids: List[str]):
        logger.info("Building BM25 Local Index for Hybrid Search...")
        corpus = [chunk.get('embedding_text', chunk['text']).lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.corpus_ids = ids
        
        with open(self.index_path, 'wb') as f:
            pickle.dump((self.bm25, self.corpus_ids), f)
        logger.info("BM25 Index saved successfully.")
        
    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.bm25, self.corpus_ids = pickle.load(f)
            logger.info("BM25 Index loaded successfully.")
        else:
            logger.warning("BM25 Index not found!")
            
    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        if not self.bm25: return []
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append({"id": self.corpus_ids[idx], "score": float(doc_scores[idx])})
        return results

def store_chunks_hybrid(collection, bm25_index: BM25Index, chunks: List[Dict], batch_size: int = 100):
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    # Hierarchy mapping
    hierarchy_map = {
        'Undang-Undang': 1,
        'Peraturan Pemerintah Pengganti Undang-Undang': 1,
        'Peraturan Pemerintah': 2,
        'Peraturan Presiden': 3,
        'Peraturan Menteri Ketenagakerjaan': 4,
        'Peraturan Menteri': 4,
        'Keputusan Menteri': 5
    }
    
    for idx, chunk in enumerate(chunks):
        chunk_id = f"chunk_{idx}_{str(uuid.uuid4())[:8]}"
        ids.append(chunk_id)
        documents.append(chunk["text"])
        embeddings.append(chunk["dense_embedding"])
        
        meta = chunk.get("metadata", {})
        
        # --- AUTO GENERATE MISSING FIELDS ---
        reg_type = meta.get('regulation_type', 'Unknown')
        nomor = meta.get('nomor', '')
        year = meta.get('year', '')
        pasal = meta.get('pasal_id', '')
        
        if not meta.get('citation_text'):
            meta['citation_text'] = f"{reg_type} No. {nomor} Tahun {year} {pasal}"
            
        if not meta.get('regulation_hierarchy'):
            meta['regulation_hierarchy'] = hierarchy_map.get(reg_type, 5)
            
        if not meta.get('section_type'):
            meta['section_type'] = 'batang_tubuh'
            
        # Ensure year is int
        try: meta['year'] = int(year)
        except: pass
        
        clean_meta = {}
        for k, v in meta.items():
            if v is None:
                clean_meta[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)
                
        clean_meta["display_text"] = chunk.get("display_text", chunk["text"])
        metadatas.append(clean_meta)
        
    for i in tqdm(range(0, len(ids), batch_size), desc="Uploading to ChromaDB"):
        collection.add(
            ids=ids[i:i+batch_size],
            documents=documents[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
    
    bm25_index.build_index(chunks, ids)
    logger.info(f"Successfully stored {len(ids)} chunks in ChromaDB & BM25 Index.")"""),
    ("markdown", "## EKSEKUSI BATCH 1\nJalankan cell di bawah ini untuk memulai proses ke data Anda."),
    ("code", """# --- EXECUTION BLOCK ---
data_path = "../data/processed_chunks_1b.json" # Match your backup script output

if os.path.exists(data_path):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")
    
    # Init Pipeline
    pipeline = LegalEmbeddingPipeline()
    embedded_chunks = pipeline.embed_chunks(chunks, batch_size=32)
    
    # Store
    collection = setup_chromadb()
    bm25 = BM25Index()
    store_chunks_hybrid(collection, bm25, embedded_chunks)
    
    print("Batch 1 Selesai! Data per-pasal berhasil di-embed dan disimpan ke ChromaDB.")
else:
    print(f"File {data_path} tidak ditemukan. Pastikan path benar.")
""")
]

# ==========================================
# BATCH 2: Phase 7, 8, 9
# ==========================================
batch2_cells = [
    ("markdown", "# Batch 2: Phase 7, 8, 9\n## Phase 7: Hybrid Retrieval (ChromaDB + BM25 + RRF)\n## Phase 8: Reranker (BGE-M3 CrossEncoder)\n## Phase 9: Context Assembler"),
    ("code", """import logging
from typing import List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)"""),
    ("markdown", "### Mendefinisikan ulang dependency dari Batch 1 untuk kemudahan eksekusi"),
    ("code", """# Mendefinisikan class yang dibutuhkan dari Batch 1 agar notebook ini bisa berjalan mandiri
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
import os
import pickle
from rank_bm25 import BM25Okapi

class LegalEmbeddingPipeline:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_texts(self, texts: List[str], prefix: str = "") -> List[List[float]]:
        batch_texts = [prefix + t for t in texts]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query], prefix="query: ")[0]

class BM25Index:
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_ids = []
        
    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.bm25, self.corpus_ids = pickle.load(f)
        else:
            print("Warning: BM25 Index not found!")
            
    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        if not self.bm25: return []
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append({"id": self.corpus_ids[idx], "score": float(doc_scores[idx])})
        return results
"""),
    ("markdown", "### Phase 7: Hybrid Retrieval dengan RRF"),
    ("code", """class HybridRetriever:
    def __init__(self, chroma_collection, bm25_index, embedding_pipeline):
        self.collection = chroma_collection
        self.bm25_index = bm25_index
        self.embedding_pipeline = embedding_pipeline
        
    def _rrf_fusion(self, dense_hits: List[Dict], sparse_hits: List[Dict], k: int = 60, rrf_k: int = 60) -> List[Dict]:
        scores = {}
        def _compute_rrf(hits, weight=1.0):
            for rank, hit in enumerate(hits, 1):
                doc_id = hit['id']
                if doc_id not in scores:
                    scores[doc_id] = {"score": 0.0, "hit_data": hit}
                scores[doc_id]["score"] += weight * (1.0 / (rrf_k + rank))
                
        _compute_rrf(dense_hits, weight=1.0)
        _compute_rrf(sparse_hits, weight=0.8)
        
        fused = [v["hit_data"] for k, v in sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)]
        missing_ids = [doc['id'] for doc in fused if 'metadata' not in doc]
        if missing_ids:
            missing_docs = self.collection.get(ids=missing_ids, include=["documents", "metadatas"])
            if missing_docs and missing_docs['ids']:
                lookup = {mid: {"text": mdoc, "metadata": mmeta} for mid, mdoc, mmeta in zip(missing_docs['ids'], missing_docs['documents'], missing_docs['metadatas'])}
                for doc in fused:
                    if doc['id'] in lookup and 'metadata' not in doc:
                        doc['text'] = lookup[doc['id']]['text']
                        doc['metadata'] = lookup[doc['id']]['metadata']
        return fused[:k]

    def retrieve(self, query: str, k: int = 5, fetch_k: int = 30, filter_active_only: bool = True) -> List[Dict]:
        query_vector = self.embedding_pipeline.embed_query(query)
        
        # Build filter dynamically
        filters = [{"section_type": "batang_tubuh"}]
        if filter_active_only:
            filters.append({"active_status": "Berlaku"})
            
        where_filter = filters[0] if len(filters) == 1 else {"$and": filters}
            
        dense_results = self.collection.query(
            query_embeddings=[query_vector], n_results=fetch_k, where=where_filter, include=["documents", "metadatas", "distances"]
        )
        
        dense_hits = []
        if dense_results["ids"]:
            for i in range(len(dense_results["ids"][0])):
                dense_hits.append({"id": dense_results["ids"][0][i], "dense_score": dense_results["distances"][0][i], "text": dense_results["documents"][0][i], "metadata": dense_results["metadatas"][0][i]})
                
        sparse_raw_hits = self.bm25_index.search(query, top_k=fetch_k)
        fused_hits = self._rrf_fusion(dense_hits, sparse_raw_hits, k=fetch_k)
        return fused_hits"""),
    ("markdown", "### Phase 8: Reranker Neural CrossEncoder + Legal Pre-Score"),
    ("code", """from transformers import AutoModelForSequenceClassification

class LegalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.use_neural = True
        except Exception as e:
            self.use_neural = False

    def legal_pre_score(self, query: str, doc: Dict) -> float:
        penalty = 1.0
        meta = doc.get("metadata", {})
        if str(meta.get("active_status", "")).lower() != "berlaku": penalty += 5.0
        pasal_id = str(meta.get("pasal_id", "")).lower()
        if pasal_id and pasal_id in query.lower(): penalty -= 0.5
        try: year = int(meta.get("year", 0))
        except: year = 0
        if year > 2020: penalty -= 0.2
        elif year > 2010: penalty -= 0.1
        try: hierarchy = int(meta.get("regulation_hierarchy", 5))
        except: hierarchy = 5
        if hierarchy <= 2: penalty -= 0.3
        return max(0.1, penalty)
        
    def _neural_score(self, query: str, docs: List[Dict]) -> List[float]:
        if not self.use_neural or not docs: return [0.0] * len(docs)
        pairs = [[query, doc.get("text", "")] for doc in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.cpu().numpy().tolist()

    def rerank(self, query: str, results: List[Dict], k: int = 5) -> List[Dict]:
        filtered = []
        for res in results:
            pre_score = self.legal_pre_score(query, res)
            if pre_score <= 5.0:
                res["pre_score"] = pre_score
                filtered.append(res)
                
        if not filtered: return []
        neural_scores = self._neural_score(query, filtered)
        for res, n_score in zip(filtered, neural_scores):
            res["final_score"] = n_score - (res["pre_score"] * 2.0)
            
        filtered.sort(key=lambda x: x["final_score"], reverse=True)
        seen = set()
        deduped = []
        for res in filtered:
            meta = res.get("metadata", {})
            key = (str(meta.get("regulation_type", "")), str(meta.get("year", "")), str(meta.get("pasal_id", "")), str(meta.get("ayat_no", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(res)
                if len(deduped) == k: break
        return deduped"""),
    ("markdown", "### Phase 9: Context Assembler Advanced"),
    ("code", """class ContextAssembler:
    def __init__(self, chroma_collection):
        self.collection = chroma_collection
        
    def _fetch_siblings(self, doc_metadata: Dict) -> List[Dict]:
        pasal_id = str(doc_metadata.get("pasal_id", ""))
        source_file = str(doc_metadata.get("source_file", ""))
        if not pasal_id or not source_file: return []
            
        results = self.collection.get(
            where={"$and": [{"pasal_id": pasal_id}, {"source_file": source_file}]},
            include=["documents", "metadatas"]
        )
        siblings = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                siblings.append({"text": results["documents"][i], "metadata": results["metadatas"][i]})
        siblings.sort(key=lambda x: int(x.get("metadata", {}).get("chunk_index", 0)))
        return siblings

    def assemble(self, retrieved_docs: List[Dict]) -> str:
        context_parts = []
        for idx, doc in enumerate(retrieved_docs, 1):
            meta = doc.get("metadata", {})
            siblings = self._fetch_siblings(meta)
            reg_identity = f"Dokumen {idx}: {meta.get('regulation_type', '')} No. {meta.get('nomor', '')} Tahun {meta.get('year', '')}"
            hierarchy_str = f"Hierarki: {meta.get('bab', '')} - {meta.get('bagian', '')} - {meta.get('pasal_id', '')}"
            
            assembled_text = ""
            main_chunk_index = str(meta.get("chunk_index", ""))
            
            if not siblings:
                assembled_text = f">>> {doc.get('text', '')} <<<"
            else:
                for sib in siblings:
                    sib_text = sib.get("text", "")
                    sib_chunk_index = str(sib.get("metadata", {}).get("chunk_index", ""))
                    if sib_chunk_index == main_chunk_index:
                        assembled_text += f"\\n>>> {sib_text} <<<"
                    else:
                        assembled_text += f"\\n{sib_text}"
            
            part = f"{reg_identity}\\n{hierarchy_str}\\nIsi:{assembled_text}\\n"
            context_parts.append(part)
        return "\\n---\\n".join(context_parts)"""),
    ("markdown", "## EKSEKUSI BATCH 2\nJalankan cell di bawah ini untuk mencari dokumen dengan Pipeline Anda."),
    ("code", """# --- EXECUTION BLOCK ---
# Load DB & Index
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="hukum_ketenagakerjaan", metadata={"hnsw:space": "cosine"})
    bm25 = BM25Index()
    bm25.load_index()
    
    # Init Pipeline
    pipeline = LegalEmbeddingPipeline()
    retriever = HybridRetriever(collection, bm25, pipeline)
    reranker = LegalReranker()
    assembler = ContextAssembler(collection)
    
    # Test Query
    query = "PHK sepihak"
    print(f"\\n--- MENCARI QUERY: '{query}' ---\\n")
    
    retrieved = retriever.retrieve(query, fetch_k=10)
    reranked = reranker.rerank(query, retrieved, k=3)
    
    print(f"Top 3 Dokumen setelah Rerank:")
    for i, d in enumerate(reranked):
        print(f"{i+1}. {d['metadata'].get('citation_text')} (Score: {d.get('final_score', 0):.2f})")
        
    context = assembler.assemble(reranked)
    print("\\n--- HASIL CONTEXT ASSEMBLER ---")
    print(context)
except Exception as e:
    print(f"Gagal menjalankan eksekusi: {e}\\nPastikan Anda sudah menjalankan Batch 1 terlebih dahulu!")
""")
]

# ==========================================
# BATCH 3: Phase 10, 11
# ==========================================
batch3_cells = [
    ("markdown", "# Batch 3: Phase 10, 11\n## Phase 10: Answer Generation & Citation\n## Phase 11: Evaluasi Pipeline"),
    ("code", """import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

warnings.filterwarnings('ignore', category=FutureWarning)

# Kuantisasi 4-bit secara on-the-fly (Sangat aman untuk GPU 24GB)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Memakai Qwen3.5-27B resmi dan dikuantisasi sendiri ke 4-bit nf4
LLM_MODEL_ID = "Qwen/Qwen3.5-27B" 
print(f"Memuat model LLM resmi: {LLM_MODEL_ID}...")

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
print("LLM Siap!")"""),
    ("code", """def extract_keywords(query: str) -> str:
    messages = [
        {
            "role": "system", 
            "content": "Ekstrak kata kunci hukum penting dari pertanyaan user. Berikan HANYA kata kuncinya saja yang relevan untuk pencarian database hukum, pisahkan dengan koma. Tanpa penjelasan apapun."
        },
        {"role": "user", "content": query}
    ]
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = llm_tokenizer(prompt, return_tensors="pt", padding=True).to(llm_model.device)
    outputs = llm_model.generate(**inputs, max_new_tokens=30, temperature=0.1)
    
    keywords = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    print(f"2. Kata Kunci: {keywords}")
    return keywords

def sanitize_output(text: str) -> str:
    emoji_pat = re.compile(r'[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF\\U0001F1E0-\\U0001F1FF]+', flags=re.UNICODE)
    text = emoji_pat.sub('', text)
    text = re.sub(r'\\*\\*(.*?)\\*\\*', r'\\1', text)
    text = re.sub(r'\\*(.*?)\\*', r'\\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\n\\s*[-=•*]{3,}\\s*\\n', '\\n', text)
    return re.sub(r'\\n{3,}', '\\n\\n', text).strip()"""),
    ("markdown", "### Phase 9.5: Inisialisasi Retriever & Reranker\nMenyiapkan komponen pencarian agar notebook ini bisa berjalan mandiri tanpa harus Run Notebook 2 terlebih dahulu."),
    ("code", """# --- DEPENDENSI RETRIEVAL ---
import torch
import chromadb
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
import os
from typing import List, Dict

# Copy dari Batch 1 & 2
class LegalEmbeddingPipeline:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def embed_texts(self, texts: List[str], prefix: str = "") -> List[List[float]]:
        batch_texts = [prefix + t for t in texts]
        inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().tolist()
    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query], prefix="query: ")[0]

class BM25Index:
    def __init__(self, index_path: str = "./bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_ids = []
    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.bm25, self.corpus_ids = pickle.load(f)
    def search(self, query: str, top_k: int = 50) -> List[Dict]:
        if not self.bm25: return []
        tokenized_query = query.lower().split()
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                results.append({"id": self.corpus_ids[idx], "score": float(doc_scores[idx])})
        return results

class HybridRetriever:
    def __init__(self, chroma_collection, bm25_index, embedding_pipeline):
        self.collection = chroma_collection
        self.bm25_index = bm25_index
        self.embedding_pipeline = embedding_pipeline
    def _rrf_fusion(self, dense_hits, sparse_hits, k=60, rrf_k=60):
        scores = {}
        def _compute_rrf(hits, weight=1.0):
            for rank, hit in enumerate(hits, 1):
                doc_id = hit['id']
                if doc_id not in scores: scores[doc_id] = {"score": 0.0, "hit_data": hit}
                scores[doc_id]["score"] += weight * (1.0 / (rrf_k + rank))
        _compute_rrf(dense_hits, weight=1.0)
        _compute_rrf(sparse_hits, weight=0.8)
        fused = [v["hit_data"] for k, v in sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)]
        missing_ids = [doc['id'] for doc in fused if 'metadata' not in doc]
        if missing_ids:
            missing_docs = self.collection.get(ids=missing_ids, include=["documents", "metadatas"])
            if missing_docs and missing_docs['ids']:
                lookup = {mid: {"text": mdoc, "metadata": mmeta} for mid, mdoc, mmeta in zip(missing_docs['ids'], missing_docs['documents'], missing_docs['metadatas'])}
                for doc in fused:
                    if doc['id'] in lookup and 'metadata' not in doc:
                        doc['text'] = lookup[doc['id']]['text']
                        doc['metadata'] = lookup[doc['id']]['metadata']
        return fused[:k]
    def retrieve(self, query: str, k: int = 5, fetch_k: int = 30):
        query_vector = self.embedding_pipeline.embed_query(query)
        where_filter = {"$and": [{"section_type": "batang_tubuh"}, {"active_status": "Berlaku"}]}
        dense_results = self.collection.query(query_embeddings=[query_vector], n_results=fetch_k, where=where_filter, include=["documents", "metadatas", "distances"])
        dense_hits = []
        if dense_results["ids"]:
            for i in range(len(dense_results["ids"][0])):
                dense_hits.append({"id": dense_results["ids"][0][i], "dense_score": dense_results["distances"][0][i], "text": dense_results["documents"][0][i], "metadata": dense_results["metadatas"][0][i]})
        sparse_raw_hits = self.bm25_index.search(query, top_k=fetch_k)
        return self._rrf_fusion(dense_hits, sparse_raw_hits, k=fetch_k)

class LegalReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.use_neural = True
        except Exception:
            self.use_neural = False
    def legal_pre_score(self, query: str, doc: Dict) -> float:
        penalty = 1.0
        meta = doc.get("metadata", {})
        if str(meta.get("active_status", "")).lower() != "berlaku": penalty += 5.0
        pasal_id = str(meta.get("pasal_id", "")).lower()
        if pasal_id and pasal_id in query.lower(): penalty -= 0.5
        try: year = int(meta.get("year", 0))
        except: year = 0
        if year > 2020: penalty -= 0.2
        elif year > 2010: penalty -= 0.1
        try: hierarchy = int(meta.get("regulation_hierarchy", 5))
        except: hierarchy = 5
        if hierarchy <= 2: penalty -= 0.3
        return max(0.1, penalty)
    def _neural_score(self, query: str, docs: List[Dict]) -> List[float]:
        if not self.use_neural or not docs: return [0.0] * len(docs)
        pairs = [[query, doc.get("text", "")] for doc in docs]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.cpu().numpy().tolist()
    def rerank(self, query: str, results: List[Dict], k: int = 5) -> List[Dict]:
        filtered = []
        for res in results:
            pre_score = self.legal_pre_score(query, res)
            if pre_score <= 5.0:
                res["pre_score"] = pre_score
                filtered.append(res)
        if not filtered: return []
        neural_scores = self._neural_score(query, filtered)
        for res, n_score in zip(filtered, neural_scores):
            res["final_score"] = n_score - (res["pre_score"] * 2.0)
        filtered.sort(key=lambda x: x["final_score"], reverse=True)
        seen = set()
        deduped = []
        for res in filtered:
            meta = res.get("metadata", {})
            key = (str(meta.get("regulation_type", "")), str(meta.get("year", "")), str(meta.get("pasal_id", "")), str(meta.get("ayat_no", "")))
            if key not in seen:
                seen.add(key)
                deduped.append(res)
                if len(deduped) == k: break
        return deduped

print("Menghubungkan ke ChromaDB & BM25...")
try:
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="hukum_ketenagakerjaan", metadata={"hnsw:space": "cosine"})
    bm25 = BM25Index()
    bm25.load_index()
    pipeline = LegalEmbeddingPipeline()
    retriever = HybridRetriever(collection, bm25, pipeline)
    reranker = LegalReranker()
    print("Retriever & Reranker siap!")
except Exception as e:
    print(f"Gagal memuat DB: {e}")
"""),
    ("code", """# --- LEGAL VALIDATION LAYER ---
def validate_citations(answer_text: str, context_docs: List[Dict]) -> Dict:
    \"\"\"Pastikan setiap klaim hukum dalam answer bisa dilacak ke context.\"\"\"
    import re
    # Extract semua sitasi dari answer (format: UU/PP/Peraturan No. X Tahun Y)
    cited = re.findall(r'(Undang-Undang|Peraturan Pemerintah|Peraturan Presiden|Peraturan Menteri|UU|PP)\\s+(?:No\\.\\s*)?(\\d+)\\s+(?:Tahun\\s*)?(\\d+)', answer_text, re.IGNORECASE)
    
    valid_citations = []
    for reg_type, number, year in cited:
        # Normalisasi sebutan UU/PP
        reg_search = "Undang-Undang" if reg_type.upper() == "UU" else "Peraturan Pemerintah" if reg_type.upper() == "PP" else reg_type
        
        found = any(
            (reg_search.lower() in str(doc['metadata'].get('regulation_type', '')).lower()) and
            (str(doc['metadata'].get('nomor', '')) == str(number)) and
            (str(doc['metadata'].get('year', '')) == str(year))
            for doc in context_docs
        )
        valid_citations.append((f"{reg_type} No. {number} Tahun {year}", found))
    
    return {
        "all_citations_valid": all(v for _, v in valid_citations),
        "invalid_citations": [c for c, v in valid_citations if not v]
    }

def sanitize_output(text: str) -> str:
    emoji_pat = re.compile(r'[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF\\U0001F1E0-\\U0001F1FF]+', flags=re.UNICODE)
    text = emoji_pat.sub('', text)
    text = re.sub(r'\\*\\*(.*?)\\*\\*', r'\\1', text)
    text = re.sub(r'\\*(.*?)\\*', r'\\1', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'^#{1,6}\\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\\n\\s*[-=•*]{3,}\\s*\\n', '\\n', text)
    return re.sub(r'\\n{3,}', '\\n\\n', text).strip()

def generate_answer(query: str):
    optimized_query = extract_keywords(query)
    
    print("3. Mencari Referensi...")
    docs = retriever.retrieve(optimized_query, k=5) # Ambil lebih banyak referensi
    print(f"4. Ketemu {len(docs)} Referensi")
    
    context_str = ""
    for doc in docs:
        meta = doc['metadata']
        clean_text = re.sub(r'[\\U0001F600-\\U0001F64F\\U0001F300-\\U0001F5FF\\U0001F680-\\U0001F6FF]+', '', doc['text'])
        context_str += f"[SUMBER HUKUM: {meta.get('regulation_type', 'Aturan')} No. {meta.get('nomor','')} Tahun {meta.get('year', '-')} Pasal {meta.get('pasal_id', '')}]\\n{clean_text.strip()}\\n\\n"

    system_prompt = (
        "Anda adalah Pakar Hukum Ketenagakerjaan Indonesia.\\n"
        "ATURAN WAJIB (GROUNDING KETAT):\\n"
        "1. JAWAB HANYA berdasarkan [KONTEKS] di bawah. Jika tidak ada di konteks, katakan Anda tidak tahu.\\n"
        "2. DILARANG KERAS berhalusinasi atau mengarang 'Pedoman' atau 'Aturan' yang tidak ada di konteks.\\n"
        "3. WAJIB AWALI JAWABAN DENGAN MENYEBUTKAN PASAL DAN SUMBER HUKUMNYA. \\n"
        "4. DILARANG menggunakan kata 'Menurut Referensi' atau 'Berdasarkan Konteks'. Langsung sebut Pasalnya.\\n"
        "5. Jika menghitung PHK, perhatikan apakah ada PP 35 Tahun 2021 di konteks. Jika ada, gunakan rumus terbaru (misal: Pelanggaran berat = 0.5x pesangon).\\n"
        "6. JANGAN mengacu pada pengetahuan internal Anda jika bertentangan dengan [KONTEKS].\\n"
        "7. Gunakan teks polos saja, tanpa markdown bold/italic."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"[KONTEKS]:\\n{context_str}\\n\\n[PERTANYAAN]:\\n{query}"}
    ]
    
    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    
    print("\\n5. Jawaban AI:\\n")
    streamer = TextStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)

    outputs = llm_model.generate(
        **inputs, max_new_tokens=1500, temperature=0.01, do_sample=False, repetition_penalty=1.1
    )
    
    raw_answer = llm_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    clean_answer = sanitize_output(raw_answer)
    
    # Validation Layer
    val = validate_citations(clean_answer, docs)
    if not val["all_citations_valid"]:
        print(f"\\n⚠️ PERINGATAN: AI menyebutkan referensi fiktif/tidak ada di konteks: {val['invalid_citations']}")
        return f"Jawaban ini mungkin tidak akurat karena merujuk pada: {val['invalid_citations']}.\\n\\nOriginal: {clean_answer}", docs
    
    return clean_answer, docs

tes_kueri = "Jika pekerja di-PHK karena melakukan pelanggaran berat, berapa pesangonnya?"
jawaban, referensi = generate_answer(tes_kueri)
print("\\n--- OUTPUT FINAL ---")
print(jawaban)""")
]

if __name__ == "__main__":
    os.makedirs("d:/kuliah/llm/RAG/notebooks", exist_ok=True)
    create_notebook("d:/kuliah/llm/RAG/notebooks/01_Phase_4_5_6_Embedding_ChromaDB_BM25.ipynb", batch1_cells)
    create_notebook("d:/kuliah/llm/RAG/notebooks/02_Phase_7_8_9_HybridRetrieval_Reranker_Advanced.ipynb", batch2_cells)
    create_notebook("d:/kuliah/llm/RAG/notebooks/03_Phase_10_11_Generation_Eval.ipynb", batch3_cells)
