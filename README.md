# Smart Book Search

Локальный сервис «Умный поиск по книгам»: загружает книги в форматах `.txt`, `.fb2`, `.epub`, ищет релевантные фрагменты и отвечает на вопросы строго по найденным цитатам (без генеративных галлюцинаций).

## 1. Краткое описание решения и стек

### Что умеет сервис
- Загружать и индексировать книги (`POST /books/upload`)
- Поддерживать форматы загрузки: `.txt`, `.fb2`, `.epub`
- Искать релевантные сниппеты (`POST /search/snippets`)
- Отвечать на вопросы по книгам с указанием источников (`POST /ask`)
- Давать статус сервиса (`GET /health`)
- Предоставлять Web-интерфейс для демонстрации (`GET /`)

### Архитектура (MVP)
- `app/main.py` — HTTP API и Web entrypoint
- `app/core/store.py` — хранение чанков книг и индекса
- `app/core/retrieve.py` — retrieval (BM25/TF-IDF-подобный скоринг + покрытие + близость терминов)
- `app/core/rag.py` — extractive QA: выбирает ответные предложения только из найденных фрагментов
- `app/core/preprocess.py` — нормализация и токенизация

### Технологии
- Python 3.11+
- Flask
- Requests (для CLI-клиента)
- Pytest (тесты)

## 2. Как запустить сервис

### Запускать не обязательно:
Все уже запущено и работает  
https://books.undo.it/ - сайт  
[@smart_book_search_bot](https://t.me/smart_book_search_bot#) - Telegram бот

### Установка
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Запуск
```bash
python -m app.main
```

После запуска:
- Web UI: `http://127.0.0.1:8000/`
- API health: `http://127.0.0.1:8000/health`

## 3. Как загрузить в сервис тексты книг

Поддерживаются форматы: `.txt` (UTF-8/UTF-16/CP1251), `.fb2`, `.epub`.
Максимальный размер загружаемого файла по умолчанию: `100 MB` (`MAX_UPLOAD_BYTES`).

### Вариант A: через Web UI
1. Откройте `http://127.0.0.1:8000/`
2. В блоке «Загрузка книги» выберите файл `.txt`, `.fb2` или `.epub`
3. Нажмите «Загрузить»

### Вариант B: через API
```bash
curl -X POST -F "file=@examples/test_book.txt" http://127.0.0.1:8000/books/upload
```

Ожидаемый ответ:
```json
{
  "book": "test_book.txt",
  "chunks_added": 1,
  "message": "Book uploaded and indexed"
}
```

### Вариант C: через CLI
```bash
python cli.py upload examples/test_book.txt
```

## 4. Примеры работы сервиса (для проверки)

### Тестовая книга
- `examples/test_book.txt`

Содержит факты:
- у Пети есть песик Тайфун
- каждое утро песик встречал хозяина у калитки
- Наташа и Пьер обсуждали судьбу семьи Ростовых

### Пример 1: Поиск сниппетов
Запрос:
```bash
curl -X POST http://127.0.0.1:8000/search/snippets ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"где говорится про собаку\",\"top_k\":5}"
```

Ожидаемо сервис возвращает как минимум 1 сниппет из `test_book.txt`, где есть упоминание песика Тайфуна.

### Пример 2: Вопрос-ответ (factoid)
Запрос:
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"что делал песик утром?\",\"top_k\":5}"
```

Ожидаемый `answer`:
```text
Каждое утро песик встречал хозяина у калитки.
```

### Пример 3: Вопрос по персонажам
Запрос:
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"кто обсуждал судьбу семьи Ростовых?\",\"top_k\":5}"
```

Ожидаемый `answer`:
```text
Наташа и Пьер обсуждали судьбу семьи Ростовых у камина.
```

### Пример 4: Обработка слишком общего вопроса
Запрос:
```bash
curl -X POST http://127.0.0.1:8000/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"почему?\",\"top_k\":5}"
```

Ожидаемая реакция:
- `message: "clarify_needed"`
- ответ с просьбой уточнить вопрос
- список `suggestions` для переформулировки

## 5. Проверка тестами

```bash
python -m pytest -q
```

## 6. Ограничения MVP

- Входные форматы: `.txt`, `.fb2`, `.epub`
- Ответы extractive-only: формируются из найденных цитат
- Для production-версии можно добавить генеративный LLM-слой поверх текущего retrieval/QA

## 7. Telegram-бот (загрузка книг и выбор книг)

Запуск:
```bash
set TELEGRAM_BOT_TOKEN=<your_token>
set WEB_BASE_URL=https://books.undo.it
python telegram_bot.py
```

Команды:
- `/books` — список загруженных книг
- `/use <book1.txt;book2.txt>` — выбрать книги, по которым бот отвечает
- `/use all` — сброс фильтра (поиск по всем книгам)
- `/find <запрос>` — поиск фрагментов
- `/ask <вопрос>` — ответ по выбранным книгам
- `/health` — проверка backend

Загрузка книги через Telegram:
- отправьте боту файл `.txt`, `.fb2` или `.epub` как документ
- бот загрузит книгу в сервис и подтвердит результат
