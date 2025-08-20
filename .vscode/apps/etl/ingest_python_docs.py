import os, re, uuid, sys
from pathlib import Path
from urllib.parse import urljoin, urldefrag
import trafilatura
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Добавляем путь к корню проекта
sys.path.append(Path(__file__).parent.parent.parent.parent.__str__())


from libs.retriever.qdrant_db import ensure_collection, upsert

COLLECTION = "pydocs"
BASE = "https://docs.python.org/3/"

def list_pages(limit=None):
    """
    Возвращает список страниц из оглавления (BASE/contents.html).

    Функция выполняет HTTP-запрос к странице оглавления, парсит HTML с помощью
    BeautifulSoup, извлекает значения атрибутов `href` у ссылок вида
    `<a class="reference internal">…</a>`, преобразует их в абсолютные URL
    относительно BASE, удаляет дубликаты и лексикографически сортирует результат.
    При заданном `limit` возвращает только первые `limit` ссылок.

    Args:
        limit (int | None): Максимальное число ссылок в выдаче. Если None — вернуть все.

    Returns:
        list[str]: Отсортированный список абсолютных URL страниц из оглавления.

    Raises:
        requests.RequestException: При сетевых ошибках (например, таймаут).

    Notes:
        - Обрабатываются только прямые ссылки со страницы `contents.html`; рекурсивного обхода нет.
        - Требуются глобальная константа BASE и импорты: `requests`, `bs4.BeautifulSoup`, `urllib.parse.urljoin`.
        - Выполняется сетевой ввод-вывод; учитывайте это в тестах (рекомендуется моксеть запрос).

    Complexity:
        O(n log n), где n — число найденных ссылок (из-за сортировки).

    Example:
        >>> pages = list_pages(limit=100)
        >>> pages[:3]
        ['https://docs.python.org/3/library/abc.html', ...]
    """
    toc = requests.get(urljoin(BASE, "contents.html"), timeout=30).text
    soup = BeautifulSoup(toc, "html.parser")
    hrefs = [urldefrag(a.get("href"))[0] for a in soup.select("a.reference.internal") if a.get("href")]
    pages = [urljoin(BASE, h) for h in sorted(set(hrefs))]
    return pages[:limit] if limit else pages

def fetch_clean(url):
    html = requests.get(url, timeout=30).text
    extracted = trafilatura.extract(html, 
                                    include_comments=False, 
                                    include_tables=False, 
                                    output_format="markdown",
                                    with_metadata=True,
                                    url=url)
    return extracted or ""

def chunk(text: str, size=500, overlap=50):
    """
    Делит текст на фрагменты с сохранением смысловых границ (абзацы/строки/слова).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""], # стандартные разделители
        length_function=len,
    )
    return splitter.split_text(text)

def preprocess_text(text: str) -> str:
    """
    Минимальная нормализация markdown-текста:
    - нормализация переводов строк
    - схлопывание множественных пробелов
    - ограничение подряд идущих пустых строк до 1-2
    - обрезка пробелов по краям
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    # Оставляем структуру абзацев, но убираем избыточные пробелы
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = t.strip()
    return t

def main():
    ensure_collection(COLLECTION)

    pages = list_pages(limit=int(os.getenv("INGEST_LIMIT", "50")))
    total, inserted = 0, 0

    for url in pages:
        try:
            txt = fetch_clean(url)
            if not txt: continue
            parts = chunk(txt, size=int(os.getenv("CHUNK_SIZE","500")), overlap=int(os.getenv("CHUNK_OVERLAP","50")))
            # Препроцессинг и фильтрация чанков
            min_chars = int(os.getenv("MIN_CHUNK_CHARS", "200"))
            seen = set()
            cleaned_parts = []
            for raw in parts:
                cleaned = preprocess_text(raw)
                if len(cleaned) < min_chars:
                    continue
                if cleaned in seen:
                    continue
                seen.add(cleaned)
                cleaned_parts.append(cleaned)
            title = url.rsplit("/",1)[-1]
            doc_id = str(uuid.uuid4())

            rows = []
            payloads = []
            for idx, t in enumerate(cleaned_parts):
                rows.append({
                    "id": doc_id, "source": "python-docs", "title": title, "url": url,
                    "chunk_id": idx, "text": t, "section": "", # можно распарсить h2/h3 при желании
                })
                point_id = str(uuid.uuid4())
                payloads.append({
                    "id": point_id,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "title": title,
                    "url": url,
                    "section": "",
                    "text": t
                })

            if payloads:
                upsert(COLLECTION, payloads)

            inserted += len(cleaned_parts)
        except Exception as e:
            print("ERR", url, e)
        finally:
            total += 1

    print(f"Done: pages={total}, chunks={inserted}")

if __name__ == "__main__":
    main()
