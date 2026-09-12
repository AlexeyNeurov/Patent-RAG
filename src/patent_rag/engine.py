# engine.py — единый модуль поиска, импортируется в app.py
# ─────────────────────────────────────────────────────────────────

import os
import re
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ── Конфигурация ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QDRANT_PATH = Path(
    os.getenv("PATENT_RAG_QDRANT_PATH", PROJECT_ROOT / "data" / "qdrant_local_db")
)
COLLECTION_NAME = os.getenv("PATENT_RAG_COLLECTION", "patents_ru")
MODEL_NAME = os.getenv(
    "PATENT_RAG_MODEL",
    "DeepPavlov/rubert-base-cased-sentence",
)

SYNONYMS = {
    "нефть": ["петролеум", "нефтяное сырьё", "углеводородное сырьё", "crude", "нефтяной конденсат", "нефтяная эмульсия", "углеводороды", "пластовая жидкость"],
    "дизель": ["дизельное топливо", "дизельная фракция", "ДТ", "газойль", "среднедистиллятная фракция ", "дизельное горючее"],
    "катализатор": ["катализ", "каталитический", "каталитическая композиция", "активный слой"],
    "депарафинизация": ["изодепарафинизация", "гидроизодепарафинизация"],
    "цеолит": ["молекулярное сито", "MFI", "MTT", "MEL", "USY", "ЦВН", "микропористый алюмосиликат", "цеолитсодержащий адсорбент", "BEA", "фожазит", "морденит"],
    "гидроочистка": ["гидрогенизация", "гидрофинишинг", "гидрообессеривание", "гидроденитрогенизация", "гидродеароматизация", "гидрокрекинг мягкий"],
    "бензин": ["бензиновая фракция", "автобензин", "нафта"],
    "алюмооксид": ["оксид алюминия", "Al2O3", "гамма-оксид алюминия", "псевдобемит", "гамма-глинозем", "бемит", "алюминиевый гидроксид"],
}

STOPWORDS = {
    "и","в","на","с","по","для","из","от","до","при","а","или","не","но",
    "к","о","об","то","что","как","является","может","быть","также","при",
    "этом","которые","которая","который","изобретение","относится","области",
    "содержащий","включающий","характеризующийся",
}

# ── Инициализация (вызывается один раз при старте) ─────────────────────────────

_client      = None
_model       = None
_bm25        = None
_all_patents = []
_number_to_idx = {}
_citation_graph = None


def init_engine():
    """Загружаем всё один раз. Streamlit кэширует через @st.cache_resource."""
    global _client, _model, _bm25, _all_patents, _number_to_idx, _citation_graph

    if not QDRANT_PATH.exists():
        raise FileNotFoundError(
            "Локальный индекс Qdrant не найден. "
            f"Ожидаемый путь: {QDRANT_PATH}. "
            "Инструкции по подготовке данных приведены в data/README.md."
        )

    _client = QdrantClient(path=str(QDRANT_PATH))
    _model  = SentenceTransformer(MODEL_NAME)

    # Загружаем патенты
    _all_patents   = _load_all_patents()
    _number_to_idx = {p["number"]: i for i, p in enumerate(_all_patents)}

    # BM25 индекс
    corpus = [_tokenize(_bm25_text(p)) for p in _all_patents]
    _bm25  = BM25Okapi(corpus)

    # Граф цитирований
    _citation_graph = _build_graph()

    return _client, _model, _bm25, _all_patents, _number_to_idx, _citation_graph


# ── Вспомогательные функции ───────────────────────────────────────────────────

def _load_all_patents():
    patents = []
    offset  = None
    while True:
        batch, offset = _client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        if not batch:
            break
        for p in batch:
            patents.append({
                "id":       p.id,
                "number":   p.payload.get("number", ""),
                "title":    p.payload.get("title", ""),
                "abstract": p.payload.get("abstract", ""),
                "date":     p.payload.get("publication_date", ""),
                "citations":p.payload.get("citations", []),
            })
        if offset is None:
            break
    return patents


def _build_graph():
    G = nx.DiGraph()
    for p in _all_patents:
        G.add_node(p["number"])
        for cited in p["citations"]:
            G.add_edge(p["number"], cited)
    return G


def _tokenize(text: str) -> list:
    if not text:
        return []
    text   = text.lower()
    tokens = re.findall(r"[а-яёa-z][а-яёa-z0-9\-]*|[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def _bm25_text(p: dict) -> str:
    return f"{p['title']} {p['title']} {p['abstract']}"


def expand_query(query: str) -> str:
    extra = []
    ql    = query.lower()
    for key, syns in SYNONYMS.items():
        if key in ql:
            extra.extend(syns)
    return (query + " " + " ".join(extra)).strip() if extra else query


def _jaccard(a: str, b: str) -> float:
    ca = set(_citation_graph.successors(a)) if _citation_graph.has_node(a) else set()
    cb = set(_citation_graph.successors(b)) if _citation_graph.has_node(b) else set()
    if not ca or not cb:
        return 0.0
    return len(ca & cb) / len(ca | cb)


def _cocitation(a: str, b: str) -> float:
    pa = set(_citation_graph.predecessors(a)) if _citation_graph.has_node(a) else set()
    pb = set(_citation_graph.predecessors(b)) if _citation_graph.has_node(b) else set()
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)


# ── Публичная функция поиска ──────────────────────────────────────────────────

def hybrid_search(
    query:         str,
    top_k:         int   = 10,
    pool:          int   = 50,
    vector_weight: float = 0.6,
    bm25_weight:   float = 0.4,
    use_citations: bool  = True,
) -> list:

    expanded     = expand_query(query)
    query_vector = _model.encode(expanded, normalize_embeddings=True).tolist()

    # Векторный поиск
    vec_results = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=pool,
        with_payload=True,
    ).points

    # BM25 поиск
    scores     = _bm25.get_scores(_tokenize(expanded))
    top_idx    = np.argsort(scores)[::-1][:pool]
    bm25_results = [
        {"number": _all_patents[i]["number"], "score": float(scores[i])}
        for i in top_idx if scores[i] > 0
    ]

    # RRF объединение
    k   = 60
    rrf = defaultdict(float)
    for rank, r in enumerate(vec_results, 1):
        rrf[r.payload["number"]] += vector_weight / (k + rank)
    for rank, r in enumerate(bm25_results, 1):
        rrf[r["number"]]         += bm25_weight   / (k + rank)

    rrf_sorted = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    # Собираем payload
    payload_map = {r.payload["number"]: r.payload for r in vec_results}
    for b in bm25_results:
        num = b["number"]
        if num not in payload_map:
            idx = _number_to_idx.get(num)
            if idx is not None:
                payload_map[num] = _all_patents[idx]

    top_number = rrf_sorted[0][0] if rrf_sorted else None

    output = []
    for number, rrf_score in rrf_sorted[:top_k * 2]:
        p = payload_map.get(number)
        if not p:
            continue

        cit_bonus = 0.0
        if use_citations and top_number:
            cit_bonus = 0.7 * _jaccard(number, top_number) + \
                        0.3 * _cocitation(number, top_number)

        final = 0.85 * rrf_score + 0.15 * cit_bonus

        output.append({
            "final_score":   round(final, 6),
            "rrf_score":     round(rrf_score, 6),
            "cit_bonus":     round(cit_bonus, 4),
            "number":        number,
            "date":          p.get("publication_date") or p.get("date", ""),
            "title":         p.get("title", ""),
            "abstract":      p.get("abstract", ""),
            "citations":     p.get("citations", []),
            "image_url":     p.get("image_url", ""),
        })

    output.sort(key=lambda x: x["final_score"], reverse=True)
    return output[:top_k]


def find_similar(patent_number: str, top_k: int = 5) -> list:
    hits = _client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[
            FieldCondition(key="number", match=MatchValue(value=patent_number))
        ]),
        with_vectors=True,
        limit=1,
    )[0]

    if not hits:
        return []

    vec = hits[0].vector
    raw = _client.query_points(
        collection_name=COLLECTION_NAME,
        query=vec,
        limit=top_k + 1,
        with_payload=True,
    ).points

    results = []
    for r in raw:
        num = r.payload["number"]
        if num == patent_number:
            continue
        jacc  = _jaccard(patent_number, num)
        cocit = _cocitation(patent_number, num)
        final = 0.7 * r.score + 0.2 * jacc + 0.1 * cocit
        results.append({
            "final_score": round(final, 4),
            "number":      num,
            "title":       r.payload.get("title", ""),
            "date":        r.payload.get("publication_date", ""),
            "abstract":    r.payload.get("abstract", ""),
            "citations":   r.payload.get("citations", []),
            "cit_bonus":   round(jacc, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]
