# app.py — запуск: streamlit run app.py
# pip install streamlit rank-bm25 sentence-transformers qdrant-client networkx torch
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st

from src.patent_rag.engine import expand_query, find_similar, hybrid_search, init_engine

# ── Настройка страницы ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Поиск патентов РФ",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Стили ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Основные цвета */
  :root {
    --accent:   #1a6bcc;
    --accent2:  #0d4a96;
    --bg-card:  #f8faff;
    --border:   #dde4f0;
    --score-hi: #1a7a3c;
    --score-lo: #888;
  }

  /* Карточка патента */
  .patent-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
    transition: box-shadow .2s;
  }
  .patent-card:hover { box-shadow: 0 4px 16px rgba(26,107,204,.12); }

  /* Заголовок карточки */
  .patent-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 4px;
  }

  /* Метаданные */
  .patent-meta {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 8px;
  }

  /* Реферат */
  .patent-abstract {
    font-size: 0.9rem;
    color: #333;
    line-height: 1.55;
  }

  /* Бейдж score */
  .score-badge {
    display: inline-block;
    background: var(--accent);
    color: white;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 20px;
    margin-right: 8px;
  }

  /* Цитирования */
  .citation-tag {
    display: inline-block;
    background: #e8edf7;
    color: #1a4a8a;
    font-size: 0.75rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin: 2px 3px 2px 0;
  }

  /* Расширенный запрос */
  .expanded-query {
    background: #fffbe6;
    border: 1px solid #ffe58f;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.85rem;
    margin-top: 6px;
    color: #7a5c00;
  }

  /* Метрики в карточке */
  .score-grid {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 0.78rem;
    color: #666;
  }
  .score-item { display: flex; flex-direction: column; align-items: center; }
  .score-val  { font-weight: 700; color: #333; font-size: 0.88rem; }

  h1 { color: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)


# ── Инициализация движка (кэшируется) ─────────────────────────────────────────

@st.cache_resource(show_spinner="Загружаем модель и индексы...")
def load_engine():
    return init_engine()


with st.spinner("Инициализация поискового движка..."):
    client, model, bm25, all_patents, number_to_idx, G = load_engine()


# ── Сайдбар — настройки ───────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Настройки поиска")

    mode = st.radio(
        "Режим поиска",
        ["🔍 По запросу", "📎 Похожие по номеру"],
        help="Выберите тип поиска"
    )

    st.divider()
    st.subheader("Параметры")

    top_k = st.slider("Количество результатов", 3, 20, 10)

    if mode == "🔍 По запросу":
        vec_w = st.slider("Вес векторного поиска", 0.1, 0.9, 0.6, 0.1)
        bm25_w = round(1.0 - vec_w, 1)
        st.caption(f"Вес BM25: **{bm25_w}** (автоматически)")

        use_cit = st.toggle("Учитывать граф цитирований", value=True)

    st.divider()
    st.subheader("📊 База данных")
    st.metric("Патентов в базе", f"{len(all_patents):,}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Узлов графа", f"{G.number_of_nodes():,}")
    with col2:
        st.metric("Рёбер графа", f"{G.number_of_edges():,}")


# ── Главный экран ─────────────────────────────────────────────────────────────

st.title("🔬 Поиск патентов Роспатент")
st.caption("Гибридный поиск: векторная семантика + BM25 + граф цитирований (поле 56)")

# ── Форма поиска ──────────────────────────────────────────────────────────────

if mode == "🔍 По запросу":
    query = st.text_input(
        "Поисковый запрос",
        placeholder="Например: катализатор гидроизодепарафинизации дизельных фракций",
        label_visibility="collapsed",
    )

    # Показываем расширенный запрос
    if query:
        expanded = expand_query(query)
        if expanded != query:
            st.markdown(
                f'<div class="expanded-query">🔁 <b>Расширено синонимами:</b> {expanded}</div>',
                unsafe_allow_html=True,
            )

    search_btn = st.button("🔍 Найти патенты", type="primary", use_container_width=True)

    if search_btn and query:
        with st.spinner("Ищем..."):
            results = hybrid_search(
                query,
                top_k=top_k,
                vector_weight=vec_w,
                bm25_weight=bm25_w,
                use_citations=use_cit,
            )

        st.success(f"Найдено: **{len(results)}** патентов")
        st.divider()

        for i, r in enumerate(results, 1):
            citations = r.get("citations", [])
            cit_html  = "".join(
                f'<span class="citation-tag">{c}</span>'
                for c in citations[:5]
            )
            if len(citations) > 5:
                cit_html += f'<span class="citation-tag">+{len(citations)-5}</span>'

            score_pct = min(int(r["final_score"] * 10000), 100)

            st.markdown(f"""
            <div class="patent-card">
              <div class="patent-title">#{i} &nbsp; {r['title']}</div>
              <div class="patent-meta">
                📄 <b>{r['number']}</b> &nbsp;|&nbsp;
                📅 {r['date']} &nbsp;|&nbsp;
                <span class="score-badge">{score_pct}%</span>
              </div>
              <div class="patent-abstract">{r['abstract'][:400]}...</div>
              {'<div style="margin-top:8px"><b style="font-size:.8rem;color:#555">Цитирует:</b> ' + cit_html + '</div>' if citations else ''}
              <div class="score-grid">
                <div class="score-item"><span>Итог</span><span class="score-val">{r['final_score']:.5f}</span></div>
                <div class="score-item"><span>RRF</span><span class="score-val">{r['rrf_score']:.5f}</span></div>
                <div class="score-item"><span>Цитат.</span><span class="score-val">{r['cit_bonus']:.4f}</span></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Кнопка "Найти похожие" для каждого результата
            if st.button(f"🔗 Найти похожие на {r['number']}", key=f"sim_{i}"):
                with st.spinner(f"Ищем похожие на {r['number']}..."):
                    similar = find_similar(r["number"], top_k=5)
                with st.expander(f"Похожие на {r['number']} ({len(similar)} шт.)", expanded=True):
                    for s in similar:
                        st.markdown(f"""
                        <div class="patent-card" style="border-left-color:#e67e22">
                          <div class="patent-title">{s['title']}</div>
                          <div class="patent-meta">📄 <b>{s['number']}</b> &nbsp;|&nbsp; 📅 {s['date']}
                            &nbsp;|&nbsp; <span class="score-badge" style="background:#e67e22">{round(s['final_score']*100,1)}%</span>
                          </div>
                          <div class="patent-abstract">{s['abstract'][:250]}...</div>
                        </div>
                        """, unsafe_allow_html=True)

# ── Режим похожих по номеру ───────────────────────────────────────────────────

else:
    patent_num = st.text_input(
        "Номер патента",
        placeholder="Например: 2826904",
        label_visibility="collapsed",
    )

    search_btn = st.button("🔗 Найти похожие", type="primary", use_container_width=True)

    if search_btn and patent_num:
        with st.spinner(f"Ищем похожие на {patent_num}..."):
            results = find_similar(patent_num.strip(), top_k=top_k)

        if not results:
            st.warning(f"Патент **{patent_num}** не найден в базе.")
        else:
            st.success(f"Найдено **{len(results)}** похожих патентов")
            st.divider()

            for i, r in enumerate(results, 1):
                citations = r.get("citations", [])
                cit_html  = "".join(
                    f'<span class="citation-tag">{c}</span>'
                    for c in citations[:5]
                )

                st.markdown(f"""
                <div class="patent-card">
                  <div class="patent-title">#{i} &nbsp; {r['title']}</div>
                  <div class="patent-meta">
                    📄 <b>{r['number']}</b> &nbsp;|&nbsp;
                    📅 {r['date']} &nbsp;|&nbsp;
                    <span class="score-badge">{round(r['final_score']*100,1)}%</span>
                  </div>
                  <div class="patent-abstract">{r['abstract'][:400]}...</div>
                  {'<div style="margin-top:8px"><b style="font-size:.8rem;color:#555">Цитирует:</b> ' + cit_html + '</div>' if citations else ''}
                  <div class="score-grid">
                    <div class="score-item"><span>Итог</span><span class="score-val">{r['final_score']:.4f}</span></div>
                    <div class="score-item"><span>Жаккар</span><span class="score-val">{r['cit_bonus']:.4f}</span></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ── Футер ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption("RuSBERT · BM25 · Qdrant · граф цитирований (поле 56) · Роспатент")
