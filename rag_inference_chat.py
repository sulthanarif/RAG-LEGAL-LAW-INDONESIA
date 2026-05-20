import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DATA_PATH = Path("data/processed_chunks_ringan_pasal_chroma_ready.json")
CHROMA_DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "hukum_ketenagakerjaan"
BM25_PATH = Path("data/bm25_index.pkl")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_ID = os.getenv("RAG_LLM_MODEL_ID", "Qwen/Qwen3.5-27B")


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def _remove_emoji(text: str) -> str:
    emoji_pat = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        flags=re.UNICODE,
    )
    return emoji_pat.sub("", text or "")


def sanitize_output(text: str) -> str:
    text = _remove_emoji(text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\[(?:R\d+(?:\s*,\s*)?)+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:\n\s*Referensi\s*:\s*)[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n\s*[-=*]{3,}\s*\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ensure_unique_chunk_ids(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        base_id = str(chunk.get("id") or f"chunk-{idx}")
        count = seen.get(base_id, 0)
        seen[base_id] = count + 1
        if count:
            seed = "::".join(
                [
                    base_id,
                    str(idx),
                    chunk.get("metadata", {}).get("source_file", ""),
                    chunk.get("metadata", {}).get("pasal_id", ""),
                    chunk.get("text", "")[:200],
                ]
            )
            chunk["original_id"] = base_id
            chunk["id"] = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return chunks


def load_chunks(path: Path = DATA_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ditemukan. Jalankan finalisasi chunk dan vector pipeline dulu."
        )
    chunks = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "text", "display_text", "embedding_text", "citation_text", "metadata"}
    missing = [i for i, chunk in enumerate(chunks[:20]) if not required.issubset(chunk)]
    if missing:
        raise ValueError(f"Chunk belum memakai schema batch 3. Cek sample index: {missing}")
    return ensure_unique_chunk_ids(chunks)


def normalize_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(chunk.get("metadata", {}))
    meta["chunk_id"] = chunk.get("id", "")
    meta["citation_text"] = chunk.get("citation_text", "")
    meta["source_file"] = meta.get("source_file", "")
    meta["pasal_id"] = meta.get("pasal_id", "")
    meta["bab"] = meta.get("bab", "")
    meta["bab_title"] = meta.get("bab_title", "")
    meta["regulation_type"] = meta.get("regulation_type", "")
    meta["nomor"] = meta.get("nomor", "")
    meta["tentang"] = meta.get("tentang", "")
    meta["year"] = int(meta.get("year") or 0)
    meta["publication_year"] = int(meta.get("publication_year") or meta.get("year") or 0)
    safe: Dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    return safe


def compact_citation(meta: Dict[str, Any]) -> str:
    citation_text = str(meta.get("citation_text") or "").strip()
    if citation_text and citation_text.lower() != "unknown":
        return citation_text

    reg_type = str(meta.get("regulation_type") or "Aturan").strip()
    nomor = str(meta.get("nomor") or "").strip()
    year = str(meta.get("publication_year") or meta.get("year") or "").strip()
    pasal = str(meta.get("pasal_id") or "").strip()
    parts = [reg_type]
    if nomor and nomor.lower() != "unknown":
        parts.append(f"No. {nomor}")
    if year and year != "0":
        parts.append(f"Tahun {year}")
    citation = " ".join(parts).strip()
    if pasal:
        citation = f"{citation}, {pasal}"
    return citation


def normalize_reg_type(value: str) -> str:
    value = str(value or "").lower()
    if "undang" in value or value == "uu":
        return "uu"
    if "pemerintah" in value or value == "pp":
        return "pp"
    if "presiden" in value or "perpres" in value:
        return "perpres"
    if "menteri" in value or "permen" in value:
        return "permen"
    return value


@dataclass
class ShortTermMemory:
    max_turns: int = 4
    turns: List[Dict[str, str]] = field(default_factory=list)

    def add(self, user: str, assistant: str) -> None:
        self.turns.append({"user": sanitize_output(user), "assistant": sanitize_output(assistant)})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        self.turns.clear()

    def as_prompt(self) -> str:
        if not self.turns:
            return ""
        lines = []
        for i, turn in enumerate(self.turns, 1):
            lines.append(f"Turn {i} - Pengguna: {turn['user']}")
            lines.append(f"Turn {i} - Asisten: {turn['assistant'][:900]}")
        return "\n".join(lines)

    def retrieval_query(self, query: str, max_chars: int = 1400) -> str:
        if not self.turns:
            return query
        history = " ".join(turn["user"] for turn in self.turns[-self.max_turns :])
        combined = f"{history} {query}".strip()
        return combined[-max_chars:]


class LegalRAGInference:
    def __init__(
        self,
        data_path: Path = DATA_PATH,
        chroma_dir: Path = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        bm25_path: Path = BM25_PATH,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        reranker_model_name: str = RERANKER_MODEL_NAME,
        llm_model_id: str = LLM_MODEL_ID,
        use_bm25: bool = True,
        use_reranker: bool = True,
        load_llm: bool = True,
        auto_build_chroma: bool = False,
    ) -> None:
        self.device = _device()
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker
        self.llm_model_id = llm_model_id
        self.chunks = load_chunks(data_path)
        self.embedding_model = SentenceTransformer(embedding_model_name, device=self.device)
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self._get_or_build_collection(collection_name, auto_build=auto_build_chroma)
        self.bm25 = None
        self.bm25_ids: List[str] = []
        self.reranker = None
        self.llm_tokenizer = None
        self.llm_model = None

        if use_bm25 and bm25_path.exists():
            with bm25_path.open("rb") as f:
                payload = pickle.load(f)
            self.bm25 = payload["bm25"]
            self.bm25_ids = payload["ids"]

        if use_reranker:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(reranker_model_name, device=self.device)

        if load_llm:
            self.load_llm()

    def _get_or_build_collection(self, collection_name: str, auto_build: bool) -> Any:
        try:
            return self.chroma_client.get_collection(collection_name)
        except Exception as exc:
            if not auto_build:
                raise RuntimeError(
                    f"Collection Chroma '{collection_name}' tidak ditemukan. "
                    "Jalankan vector pipeline atau set auto_build_chroma=True."
                ) from exc
            print(f"Collection Chroma '{collection_name}' tidak ditemukan. Membuat ulang dari chunk batch 3.")
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._build_chroma_collection(collection)
            return collection

    def _build_chroma_collection(self, collection: Any, batch_size: int = 64) -> None:
        ids = [chunk["id"] for chunk in self.chunks]
        documents = [chunk.get("display_text") or chunk.get("text") or "" for chunk in self.chunks]
        metadatas = [normalize_metadata(chunk) for chunk in self.chunks]
        embedding_texts = [chunk.get("embedding_text") or chunk.get("display_text") or chunk.get("text") or "" for chunk in self.chunks]

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            embeddings = self.embedding_model.encode(
                ["passage: " + text for text in embedding_texts[start:end]],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings,
            )
            print(f"Chroma build: {min(end, len(ids))}/{len(ids)} chunks", end="\r")
        print(f"\nChroma collection ready: {collection.count()} chunks")

    def load_llm(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        use_4bit = torch.cuda.is_available()
        quantization_config = None
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_id, trust_remote_code=True)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

    def dense_search(self, query: str, fetch_k: int = 30) -> List[Dict[str, Any]]:
        q_emb = self.embedding_model.encode(
            ["query: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc_id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "dense_distance": float(dist),
                    "source": "dense",
                }
            )
        return hits

    def bm25_search(self, query: str, fetch_k: int = 30) -> List[Dict[str, Any]]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize_for_bm25(query))
        order = np.argsort(scores)[::-1][:fetch_k]
        ids = [self.bm25_ids[i] for i in order if scores[i] > 0]
        if not ids:
            return []
        got = self.collection.get(ids=ids, include=["documents", "metadatas"])
        lookup = {doc_id: (doc, meta) for doc_id, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])}
        hits = []
        for i in order:
            doc_id = self.bm25_ids[i]
            if scores[i] <= 0 or doc_id not in lookup:
                continue
            doc, meta = lookup[doc_id]
            hits.append(
                {
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta,
                    "bm25_score": float(scores[i]),
                    "source": "bm25",
                }
            )
        return hits

    @staticmethod
    def rrf_fuse(result_sets: List[List[Dict[str, Any]]], weights: Optional[List[float]] = None, rrf_k: int = 60) -> List[Dict[str, Any]]:
        weights = weights or [1.0] * len(result_sets)
        fused: Dict[str, Dict[str, Any]] = {}
        for hits, weight in zip(result_sets, weights):
            for rank, hit in enumerate(hits, 1):
                item = fused.setdefault(hit["id"], {"score": 0.0, "hit": dict(hit)})
                item["score"] += weight / (rrf_k + rank)
        out = []
        for item in sorted(fused.values(), key=lambda x: x["score"], reverse=True):
            hit = item["hit"]
            hit["rrf_score"] = item["score"]
            out.append(hit)
        return out

    @staticmethod
    def lex_posterior_score(hit: Dict[str, Any]) -> float:
        meta = hit.get("metadata", {})
        year = int(meta.get("publication_year") or meta.get("year") or 0)
        base = float(hit.get("rrf_score", 0.0))
        if year:
            base += min(max(year - 1945, 0), 100) / 10000
        return base

    def dedupe_legal_hits(self, hits: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        out = []
        seen: Set[tuple] = set()
        for hit in sorted(hits, key=self.lex_posterior_score, reverse=True):
            meta = hit.get("metadata", {})
            key = (
                meta.get("source_file", ""),
                meta.get("pasal_id", ""),
                meta.get("chunk_kind", ""),
                meta.get("chunk_index", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            hit["final_score"] = self.lex_posterior_score(hit)
            out.append(hit)
            if len(out) >= k:
                break
        return out

    def retrieve_documents(self, query: str, k: int = 6, fetch_k: int = 30) -> List[Dict[str, Any]]:
        dense_hits = self.dense_search(query, fetch_k=fetch_k)
        sparse_hits = self.bm25_search(query, fetch_k=fetch_k) if self.use_bm25 else []
        fused = self.rrf_fuse([dense_hits, sparse_hits], weights=[1.0, 0.7]) if sparse_hits else dense_hits
        return self.dedupe_legal_hits(fused, k=k)

    @staticmethod
    def query_terms(query: str) -> Set[str]:
        stopwords = {
            "yang", "dan", "atau", "karena", "dengan", "untuk", "pada", "dalam",
            "jika", "maka", "dari", "berapa", "apakah", "bagaimana", "dimana",
            "kapan", "siapa", "pekerja", "buruh", "pengusaha", "perusahaan",
            "hak", "nya", "itu", "ini", "ada", "dapat", "bisa", "oleh", "ke", "di",
        }
        return {t for t in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(t) > 2 and t not in stopwords}

    @staticmethod
    def expand_query_terms(query: str) -> str:
        q = query.lower()
        expansions = []
        rules = [
            (
                ["pelanggaran berat", "mendesak", "bersifat mendesak"],
                "pelanggaran bersifat mendesak uang pisah uang penggantian hak tidak mendapat pesangon",
            ),
            (
                ["pesangon", "phk", "pemutusan hubungan kerja"],
                "pemutusan hubungan kerja uang pesangon uang penghargaan masa kerja uang penggantian hak",
            ),
            (["pkwt", "kontrak"], "perjanjian kerja waktu tertentu kompensasi pkwt jangka waktu"),
            (["alih daya", "outsourcing"], "alih daya perusahaan alih daya hubungan kerja perlindungan upah"),
            (["upah", "gaji", "tunjangan"], "upah gaji tunjangan tetap upah minimum pembayaran upah"),
        ]
        for triggers, expansion in rules:
            if any(trigger in q for trigger in triggers):
                expansions.append(expansion)
        return " ".join([query] + expansions).strip()

    def rerank_documents(self, query: str, docs: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
        if self.reranker is None or not docs:
            return docs[:k]
        pairs = [(query, doc["text"]) for doc in docs]
        scores = self.reranker.predict(pairs)
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)
        return sorted(docs, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:k]

    def retrieve_context(self, query: str, k: int = 4, fetch_k: int = 50) -> List[Dict[str, Any]]:
        expanded = self.expand_query_terms(query)
        queries = [query] if expanded == query else [query, expanded]
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for q in queries:
            hits = self.retrieve_documents(q, k=max(k * 6, 18), fetch_k=fetch_k)
            for hit in hits:
                existing = candidate_map.get(hit["id"])
                if existing is None or hit.get("final_score", 0.0) > existing.get("final_score", 0.0):
                    candidate_map[hit["id"]] = hit
        candidates = list(candidate_map.values())
        reranked = self.rerank_documents(expanded, candidates, k=max(k * 4, 12))
        return self.filter_relevant_context(query, reranked, max_docs=k)

    def filter_relevant_context(self, query: str, docs: List[Dict[str, Any]], max_docs: int = 4) -> List[Dict[str, Any]]:
        terms = self.query_terms(self.expand_query_terms(query))
        if not terms:
            return docs[:max_docs]
        scored = []
        for doc in docs:
            meta = doc.get("metadata", {})
            haystack = " ".join(
                [
                    str(doc.get("text", "")),
                    str(meta.get("citation_text", "")),
                    str(meta.get("regulation_type", "")),
                    str(meta.get("nomor", "")),
                    str(meta.get("pasal_id", "")),
                    str(meta.get("bab_title", "")),
                ]
            ).lower()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap > 0:
                doc["context_relevance_score"] = overlap / max(len(terms), 1)
                scored.append(doc)
        if not scored:
            return docs[:1]
        return sorted(scored, key=lambda d: d.get("context_relevance_score", 0.0), reverse=True)[:max_docs]

    @staticmethod
    def build_context(docs: List[Dict[str, Any]]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            citation = compact_citation(doc["metadata"])
            text = str(doc["text"]).strip()
            parts.append(f"SUMBER HUKUM {i}: {citation}\n{text}")
        return "\n\n".join(parts)

    @staticmethod
    def validate_named_citations(answer: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        pattern = re.compile(
            r"(Undang-Undang|Peraturan Pemerintah|Peraturan Presiden|Peraturan Menteri(?:\s+Ketenagakerjaan)?|UU|PP|Perpres|Permen)"
            r"\s+(?:No\.?|Nomor)?\s*([A-Za-z0-9./-]+)\s+Tahun\s+(\d{4})",
            re.IGNORECASE,
        )
        cited: List[str] = []
        invalid: List[str] = []
        for reg_type, nomor, year in pattern.findall(answer):
            label = f"{reg_type} No. {nomor} Tahun {year}"
            cited.append(label)
            target_type = normalize_reg_type(reg_type)
            found = False
            for doc in docs:
                meta = doc.get("metadata", {})
                doc_type = normalize_reg_type(meta.get("regulation_type", ""))
                doc_nomor = str(meta.get("nomor", "")).strip()
                doc_year = str(meta.get("publication_year") or meta.get("year") or "").strip()
                if doc_type == target_type and doc_nomor == str(nomor).strip() and doc_year == str(year):
                    found = True
                    break
            if not found:
                invalid.append(label)
        out_of_scope = "tidak tersedia dalam database" in answer.lower()
        ok = not invalid and (bool(cited) or out_of_scope)
        return {"cited": sorted(set(cited)), "invalid": sorted(set(invalid)), "ok": ok}

    def build_messages(self, query: str, docs: List[Dict[str, Any]], memory: Optional[ShortTermMemory] = None) -> List[Dict[str, str]]:
        context = self.build_context(docs)
        memory_text = memory.as_prompt() if memory else ""
        memory_block = f"\nMEMORI PERCAKAPAN SINGKAT:\n{memory_text}\n" if memory_text else ""
        system_prompt = (
            "Kamu adalah asisten hukum ketenagakerjaan Indonesia yang teliti.\n"
            "Jawab hanya berdasarkan KONTEKS REFERENSI HUKUM. Jangan memakai pengetahuan luar.\n"
            "Sebutkan sumber hukum secara natural, misalnya Peraturan Pemerintah No. 35 Tahun 2021, Pasal 52 ayat (2).\n"
            "Jangan memakai kode [R1], [R2], SUMBER HUKUM 1, atau ID internal di jawaban.\n"
            "Jangan menulis daftar referensi terpisah di dalam jawaban.\n"
            "Jika konteks tidak cukup, katakan bahwa informasi tersebut tidak tersedia dalam database hukum ketenagakerjaan yang dimiliki.\n"
            "Jika ada beberapa ayat dengan kondisi berbeda, pilih ayat yang sesuai dengan kasus pengguna dan jangan campur konsekuensi antar ayat.\n"
            "Gunakan bahasa Indonesia yang rapi, langsung, tanpa markdown, tanpa emoji, dan tanpa pengulangan.\n"
            "Memori percakapan hanya boleh dipakai untuk memahami rujukan seperti 'itu', 'kasus tadi', atau pertanyaan lanjutan; sumber hukum tetap harus dari konteks."
        )
        user_prompt = (
            f"{memory_block}"
            f"KONTEKS REFERENSI HUKUM:\n{context}\n\n"
            f"PERTANYAAN PENGGUNA:\n{query}"
        )
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    def generate_chat_text(self, messages: List[Dict[str, str]], max_new_tokens: int = 800) -> str:
        if self.llm_model is None or self.llm_tokenizer is None:
            self.load_llm()
        tokenizer = self.llm_tokenizer
        model = self.llm_model
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=12000).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.04,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def extractive_fallback_answer(self, query: str, docs: List[Dict[str, Any]], max_sentences: int = 6) -> str:
        terms = self.query_terms(query)
        if not docs:
            return "Maaf, informasi tersebut tidak tersedia dalam database hukum ketenagakerjaan yang saya miliki."
        selected = []
        for doc in docs:
            citation = compact_citation(doc.get("metadata", {}))
            text = re.sub(r"\s+", " ", doc.get("text", "")).strip()
            sentences = re.split(r"(?<=[.!?])\s+|(?=\([0-9]+\)\s)", text)
            scored = []
            for sentence in sentences:
                if len(sentence.strip()) < 25:
                    continue
                score = sum(1 for term in terms if term in sentence.lower())
                if score:
                    scored.append((score, sentence.strip()))
            for _, sentence in sorted(scored, key=lambda x: x[0], reverse=True)[:2]:
                selected.append(f"{citation} menyatakan bahwa {sentence}")
                if len(selected) >= max_sentences:
                    return "\n\n".join(selected)
        citation = compact_citation(docs[0].get("metadata", {}))
        excerpt = re.sub(r"\s+", " ", docs[0].get("text", "")).strip()[:900]
        return f"{citation} memuat ketentuan berikut: {excerpt}"

    def answer(
        self,
        query: str,
        memory: Optional[ShortTermMemory] = None,
        k: int = 4,
        max_new_tokens: int = 800,
    ) -> Dict[str, Any]:
        retrieval_query = memory.retrieval_query(query) if memory else query
        docs = self.retrieve_context(retrieval_query, k=k)
        messages = self.build_messages(query, docs, memory=memory)
        answer = sanitize_output(self.generate_chat_text(messages, max_new_tokens=max_new_tokens))
        validation = self.validate_named_citations(answer, docs)
        if not validation["ok"]:
            answer = self.extractive_fallback_answer(query, docs)
            validation = self.validate_named_citations(answer, docs)
            validation["fallback"] = "extractive"
        if memory is not None:
            memory.add(query, answer)
        return {
            "query": query,
            "retrieval_query": retrieval_query,
            "answer": answer,
            "references": docs,
            "validation": validation,
        }


def print_references(docs: List[Dict[str, Any]]) -> None:
    seen: Set[str] = set()
    for doc in docs:
        citation = compact_citation(doc.get("metadata", {}))
        if citation in seen:
            continue
        seen.add(citation)
        print(f"- {citation}")


def chat_loop(engine: LegalRAGInference, memory: Optional[ShortTermMemory] = None) -> None:
    memory = memory or ShortTermMemory()
    print("RAG chat siap. Ketik 'exit' untuk keluar, 'clear' untuk menghapus memori.")
    while True:
        query = input("\nPertanyaan: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "keluar"}:
            print("Selesai.")
            break
        if query.lower() in {"clear", "reset"}:
            memory.clear()
            print("Memori percakapan sudah dikosongkan.")
            continue
        result = engine.answer(query, memory=memory)
        print("\nJawaban:")
        print(result["answer"])
        print("\nReferensi:")
        print_references(result["references"])
