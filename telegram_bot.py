from __future__ import annotations

import os

import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Команды: /find <запрос>, /ask <вопрос>")


async def find(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /find <запрос>")
        return
    query = " ".join(context.args)
    r = requests.post(f"{BASE_URL}/search/snippets", json={"query": query, "top_k": 3}, timeout=60)
    data = r.json()
    snippets = data.get("snippets", [])
    if not snippets:
        await update.message.reply_text("Ничего не найдено")
        return
    msg = []
    for s in snippets:
        msg.append(f"{s['book']} score={s['score']}\n{s['quote']}")
    await update.message.reply_text("\n\n".join(msg)[:3800])


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Использование: /ask <вопрос>")
        return
    question = " ".join(context.args)
    r = requests.post(f"{BASE_URL}/ask", json={"question": question, "top_k": 3}, timeout=60)
    data = r.json()
    msg = f"Ответ: {data.get('answer')}"
    await update.message.reply_text(msg[:3800])


def main() -> None:
    if not TOKEN:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN in environment")
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("find", find))
    app.add_handler(CommandHandler("ask", ask))
    app.run_polling()


if __name__ == "__main__":
    main()
