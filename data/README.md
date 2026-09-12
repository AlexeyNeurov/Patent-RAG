# Данные и локальный индекс

Каталог `data/` предназначен для локальных данных Patent RAG. Содержимое индекса не коммитится в Git.

## Ожидаемая структура

```text
data/
└── qdrant_local_db/
    ├── meta.json
    └── collection/
        └── patents_ru/
```

Коллекция `patents_ru` должна использовать косинусную метрику и векторы размерности 768 для модели `DeepPavlov/rubert-base-cased-sentence`.

Минимальный payload одной точки:

```json
{
  "number": "2826904",
  "title": "Название изобретения",
  "abstract": "Текст реферата патента",
  "publication_date": "2024-09-16",
  "citations": ["1234567", "2345678"],
  "image_url": "https://example.org/image.jpg"
}
```

Обязательны `number`, `title` и `abstract`. Поля `publication_date`, `citations` и `image_url` могут быть пустыми, но влияют на полноту интерфейса и графовое ранжирование.

## Собственный путь к индексу

Индекс можно хранить вне репозитория:

```powershell
$env:PATENT_RAG_QDRANT_PATH = "D:\patent-data\qdrant_local_db"
streamlit run app.py
```

Не публикуйте исходные выгрузки и snapshot до проверки лицензии источника, персональных данных и ограничений GitHub на размер файлов.

