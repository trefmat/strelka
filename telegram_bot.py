from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import InvalidToken, TelegramError
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
WEB_BASE_URL = os.getenv("WEB_BASE_URL", BASE_URL).rstrip("/")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_TIMEOUT = 30
TG_MESSAGE_LIMIT = 3800
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))

logger = logging.getLogger(__name__)
_CHAT_BOOK_FILTERS: dict[int, set[str]] = {}


def _clip(text: str, limit: int = TG_MESSAGE_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


async def _reply(update: Update, text: str) -> None:
    if update.message:
        await update.message.reply_text(_clip(text))


def _reader_url(book: str, start: int, end: int) -> str:
    params = urlencode(
        {
            "book": Path(book).name,
            "start": max(0, int(start)),
            "end": max(0, int(end)),
        }
    )
    return f"{WEB_BASE_URL}/web/reader.html?{params}"


def _focus_start(item: dict[str, Any]) -> int:
    return int(item.get("focus_start", item.get("offset_start", 0)) or 0)


def _focus_end(item: dict[str, Any]) -> int:
    base = _focus_start(item)
    end = int(item.get("focus_end", item.get("offset_end", base + 1)) or (base + 1))
    return end if end > base else (base + 1)


async def _reply_with_open_button(update: Update, text: str, *, book: str, start: int, end: int) -> None:
    if not update.message:
        return
    url = _reader_url(book, start, end)
    keyboard = InlineKeyboardMarkup(
        [[InlineKeyboardButton(text="Открыть в книге", url=url)]]
    )
    await update.message.reply_text(_clip(text), reply_markup=keyboard)


def _request_json(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"{BASE_URL}{path}"
    method = method.upper()
    if method == "GET":
        response = requests.get(url, timeout=API_TIMEOUT)
    elif method == "POST":
        response = requests.post(url, json=(payload or {}), timeout=API_TIMEOUT)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("API returned non-object JSON")
    return data


def _upload_book_bytes(filename: str, content: bytes) -> dict[str, Any]:
    response = requests.post(
        f"{BASE_URL}/books/upload",
        files={"file": (filename, content, "text/plain")},
        timeout=API_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("API returned non-object JSON")
    return data


def _http_error_details(response: requests.Response | None) -> str:
    if response is None:
        return ""
    if response.status_code == 405:
        return "Метод запроса не поддерживается для этого endpoint."

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "application/json" in content_type:
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("message") or ""
            return str(detail)
        except Exception:
            return ""
    return ""


async def _api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(_request_json, method, path, payload)
    except requests.HTTPError as exc:
        details = _http_error_details(exc.response)
        message = f"Ошибка API ({exc.response.status_code if exc.response else 'HTTP'})."
        if details:
            message += f" {details}"
        logger.warning("API HTTP error: %s", message)
        return {"_error": message}
    except requests.RequestException as exc:
        logger.warning("API request failed: %s", exc)
        return {"_error": "Не удалось связаться с API сервиса. Проверьте, что backend запущен."}
    except ValueError as exc:
        logger.warning("API payload error: %s", exc)
        return {"_error": "API вернул некорректный ответ."}


async def _api_upload_book(filename: str, content: bytes) -> dict[str, Any] | None:
    try:
        return await asyncio.to_thread(_upload_book_bytes, filename, content)
    except requests.HTTPError as exc:
        details = _http_error_details(exc.response)
        message = f"Ошибка загрузки книги ({exc.response.status_code if exc.response else 'HTTP'})."
        if details:
            message += f" {details}"
        logger.warning("Book upload error: %s", message)
        return {"_error": message}
    except requests.RequestException as exc:
        logger.warning("Book upload request failed: %s", exc)
        return {"_error": "Не удалось загрузить книгу: backend недоступен."}
    except ValueError as exc:
        logger.warning("Book upload payload error: %s", exc)
        return {"_error": "API вернул некорректный ответ при загрузке книги."}


def _get_chat_id(update: Update) -> int | None:
    if update.effective_chat is None:
        return None
    return update.effective_chat.id


def _selection_for_chat(chat_id: int | None) -> set[str] | None:
    if chat_id is None:
        return None
    return _CHAT_BOOK_FILTERS.get(chat_id)


def _with_book_filter(chat_id: int | None, payload: dict[str, Any]) -> dict[str, Any]:
    selected = _selection_for_chat(chat_id)
    if selected is not None:
        payload["books"] = sorted(selected)
    return payload


def _extract_books_list(data: dict[str, Any]) -> list[str]:
    raw = data.get("books")
    if not isinstance(raw, list):
        return []

    items: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("book")
        if isinstance(name, str) and name.strip():
            items.append(name.strip())
    return items


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _reply(
        update,
        "Команды:\n"
        "/find <запрос> — найти фрагменты\n"
        "/ask <вопрос> — ответ по цитатам\n"
        "/books — список загруженных книг\n"
        "/use <книга1;книга2> — выбрать книги для поиска\n"
        "/use all — искать по всем книгам\n"
        "/health — статус backend\n\n"
        "Чтобы загрузить книгу: отправьте боту файл .txt/.fb2/.epub как документ.",
    )


async def health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = await _api_request("GET", "/health")
    if not data:
        await _reply(update, "Пустой ответ от API.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return
    await _reply(
        update,
        f"OK: status={data.get('status')} books={data.get('books')} chunks={data.get('chunks')}",
    )


async def books(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    data = await _api_request("GET", "/books")
    if not data:
        await _reply(update, "Пустой ответ от API.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return

    raw_books = data.get("books")
    if not isinstance(raw_books, list) or not raw_books:
        await _reply(update, "Пока нет загруженных книг. Отправьте TXT-файл документом.")
        return

    chat_id = _get_chat_id(update)
    selected = _selection_for_chat(chat_id)

    lines = [f"Загруженные книги: {len(raw_books)}"]
    for idx, item in enumerate(raw_books[:25], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("book") or "")
        chunks = item.get("chunks")
        if not name:
            continue
        lines.append(f"{idx}. {name} (chunks: {chunks})")

    if len(raw_books) > 25:
        lines.append(f"... и еще {len(raw_books) - 25}")

    if selected is None:
        lines.append("Текущий фильтр: все книги")
    else:
        if selected:
            lines.append("Текущий фильтр: " + ", ".join(sorted(selected)))
        else:
            lines.append("Текущий фильтр пустой")

    lines.append("Выбор: /use <книга1;книга2> или /use all")
    await _reply(update, "\n".join(lines))


async def use(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _get_chat_id(update)
    if chat_id is None:
        await _reply(update, "Не удалось определить чат.")
        return

    raw = " ".join(context.args).strip()
    if not raw:
        await _reply(update, "Использование: /use <книга1;книга2> или /use all")
        return

    lowered = raw.lower()
    if lowered in {"all", "*", "все"}:
        _CHAT_BOOK_FILTERS.pop(chat_id, None)
        await _reply(update, "Фильтр сброшен: поиск по всем книгам.")
        return

    pieces = [p.strip() for p in re.split(r"[;,]", raw) if p.strip()]
    if not pieces:
        await _reply(update, "Не удалось распознать названия книг. Пример: /use book1.txt;book2.txt")
        return

    wanted = {Path(piece).name.strip() for piece in pieces if Path(piece).name.strip()}
    if not wanted:
        await _reply(update, "Не удалось распознать названия книг.")
        return

    data = await _api_request("GET", "/books")
    if not data:
        await _reply(update, "Пустой ответ от API.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return

    available = set(_extract_books_list(data))
    unknown = sorted(wanted - available)
    if unknown:
        await _reply(update, "Книги не найдены: " + ", ".join(unknown) + "\nПроверьте список через /books")
        return

    _CHAT_BOOK_FILTERS[chat_id] = set(wanted)
    await _reply(update, "Выбраны книги: " + ", ".join(sorted(wanted)))


async def find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await _reply(update, "Использование: /find <запрос>")
        return

    chat_id = _get_chat_id(update)
    query = " ".join(context.args)
    payload = _with_book_filter(chat_id, {"query": query, "top_k": 3})
    data = await _api_request("POST", "/search/snippets", payload)

    if not data:
        await _reply(update, "Пустой ответ от API.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return

    snippets = data.get("snippets", [])
    if not snippets:
        await _reply(update, "Ничего не найдено.")
        return

    await _reply(update, f"Найдено фрагментов: {len(snippets)}")
    for snippet in snippets:
        text = f"{snippet['book']} score={snippet['score']}\n{snippet['quote']}"
        await _reply_with_open_button(
            update,
            text,
            book=str(snippet.get("book", "")),
            start=_focus_start(snippet),
            end=_focus_end(snippet),
        )


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await _reply(update, "Использование: /ask <вопрос>")
        return

    chat_id = _get_chat_id(update)
    question = " ".join(context.args)
    payload = _with_book_filter(chat_id, {"question": question, "top_k": 3})
    data = await _api_request("POST", "/ask", payload)

    if not data:
        await _reply(update, "Пустой ответ от API.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return

    answer = str(data.get("answer") or "Ответ не получен.")
    parts = [f"Ответ: {answer}"]

    if data.get("message") == "clarify_needed":
        suggestions = data.get("suggestions") or []
        if suggestions:
            parts.append("Уточните вопрос, например:")
            for idx, suggestion in enumerate(suggestions[:3], start=1):
                parts.append(f"{idx}. {suggestion}")

    await _reply(update, "\n".join(parts))

    sources = data.get("sources") or []
    for source in sources:
        text = f"{source.get('book')} score={source.get('score')}\n{source.get('quote')}"
        await _reply_with_open_button(
            update,
            text,
            book=str(source.get("book", "")),
            start=_focus_start(source),
            end=_focus_end(source),
        )


async def upload_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.document:
        return

    document = update.message.document
    filename = (document.file_name or "book.txt").strip()
    if not filename.lower().endswith((".txt", ".fb2", ".epub")):
        await _reply(update, "Поддерживаются файлы .txt, .fb2 и .epub.")
        return
    if int(document.file_size or 0) > MAX_UPLOAD_BYTES:
        max_mb = round(MAX_UPLOAD_BYTES / (1024 * 1024), 1)
        await _reply(update, f"Файл слишком большой. Максимум: {max_mb} MB.")
        return

    try:
        tg_file = await context.bot.get_file(document.file_id)
        content = await tg_file.download_as_bytearray()
    except TelegramError as exc:
        logger.warning("Telegram file download failed: %s", exc)
        await _reply(update, "Не удалось скачать файл из Telegram.")
        return

    data = await _api_upload_book(Path(filename).name, bytes(content))
    if not data:
        await _reply(update, "Пустой ответ от API при загрузке книги.")
        return
    error = data.get("_error")
    if error:
        await _reply(update, str(error))
        return

    book_name = str(data.get("book") or Path(filename).name)
    chunks_added = data.get("chunks_added")

    chat_id = _get_chat_id(update)
    selected = _selection_for_chat(chat_id)
    if selected is not None:
        selected.add(book_name)

    await _reply(
        update,
        f"Книга загружена: {book_name} (chunks_added={chunks_added}).\n"
        "Список книг: /books\n"
        "Выбрать книги: /use <книга1;книга2>\n"
        "Поиск по всем книгам: /use all",
    )


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in environment")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("health", health))
    app.add_handler(CommandHandler("books", books))
    app.add_handler(CommandHandler("use", use))
    app.add_handler(CommandHandler("find", find))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(MessageHandler(filters.Document.ALL, upload_document))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        app.run_polling()
    except InvalidToken:
        raise SystemExit("Неверный TELEGRAM_BOT_TOKEN. Проверьте токен от @BotFather.")
    except TelegramError as exc:
        raise SystemExit(f"Ошибка Telegram API: {exc}")
    except OSError as exc:
        raise SystemExit(f"Сетевая ошибка при запуске бота: {exc}")
    finally:
        asyncio.set_event_loop(None)
        loop.close()


if __name__ == "__main__":
    main()
