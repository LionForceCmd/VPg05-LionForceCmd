"""Telegram-бот с долговременной векторной памятью (Pinecone) на pyTelegramBotAPI."""
import logging
import os
import time
from collections import defaultdict, deque

import telebot
from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import PineconeManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bot")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise SystemExit("TELEGRAM_BOT_TOKEN не найден в .env")

CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MEMORY_TOP_K = 5          # сколько воспоминаний подмешиваем в промпт
MEMORY_MIN_SCORE = 0.3    # ниже -- считаем нерелевантным
SHORT_TERM_MESSAGES = 10  # размер краткосрочной памяти (последние реплики)

bot = telebot.TeleBot(TOKEN)
memory = PineconeManager()
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
)

# краткосрочная память: последние реплики диалога по chat_id
history = defaultdict(lambda: deque(maxlen=SHORT_TERM_MESSAGES))


def build_user_metadata(user) -> dict:
    """Метаданные пользователя. Telegram не гарантирует first_name/last_name/username,
    поэтому берём только то, что есть (Pinecone не принимает None в metadata)."""
    meta = {"user_id": user.id}
    for field in ("username", "first_name", "last_name"):
        value = getattr(user, field, None)
        if value:
            meta[field] = value
    full_name = " ".join(p for p in (meta.get("first_name"), meta.get("last_name")) if p)
    meta["display_name"] = full_name or meta.get("username") or f"user_{user.id}"
    meta["timestamp"] = int(time.time())
    return meta


def recall(user_id: int, text: str) -> list[str]:
    """Достаёт из долговременной памяти то, что пользователь говорил раньше."""
    try:
        hits = memory.query_by_text(text, top_k=MEMORY_TOP_K, filter={"user_id": user_id})
    except Exception:
        log.exception("Ошибка чтения памяти")
        return []
    return [h["metadata"]["text"] for h in hits
            if h["score"] >= MEMORY_MIN_SCORE and h["metadata"].get("text")]


def remember(user, text: str) -> None:
    """Сохраняет в память ТОЛЬКО оригинальный текст пользователя (без ответов бота
    и служебных шаблонов). Дубликаты по смыслу отфильтровывает PineconeManager."""
    try:
        res = memory.upsert_document(
            text,
            metadata=build_user_metadata(user),
            filter={"user_id": user.id},
        )
        log.info("action: %s | similarity: %s | user: %s",
                 res["action"], res["similarity_score"], user.id)
    except Exception:
        log.exception("Ошибка записи в память")


def ask_llm(chat_id: int, user_name: str, text: str, memories: list[str]) -> str:
    system = (
        "Ты дружелюбный помощник в Telegram. Отвечай кратко, на языке пользователя. "
        f"Пользователя зовут {user_name}."
    )
    if memories:
        system += "\nЧто пользователь говорил раньше (долговременная память):\n- " + "\n- ".join(memories)
    messages = [{"role": "system", "content": system}, *history[chat_id],
                {"role": "user", "content": text}]
    resp = llm.chat.completions.create(model=CHAT_MODEL, messages=messages)
    return resp.choices[0].message.content.strip()


@bot.message_handler(commands=["start"])
def on_start(message):
    name = build_user_metadata(message.from_user)["display_name"]
    bot.reply_to(message, f"Привет, {name}! Я запоминаю, о чём мы говорим. /forget -- стереть память о тебе.")


@bot.message_handler(commands=["forget"])
def on_forget(message):
    memory.delete_by_filter({"user_id": message.from_user.id})
    history.pop(message.chat.id, None)
    bot.reply_to(message, "Готово, я всё забыл.")


@bot.message_handler(func=lambda m: bool(m.text) and not m.text.startswith("/"))
def on_text(message):
    user, text = message.from_user, message.text
    name = build_user_metadata(user)["display_name"]

    memories = recall(user.id, text)       # сначала вспоминаем...
    remember(user, text)                   # ...потом сохраняем (только сообщение пользователя)

    try:
        answer = ask_llm(message.chat.id, name, text, memories)
    except Exception:
        log.exception("Ошибка LLM")
        answer = "Извини, сейчас не получилось ответить."
    history[message.chat.id].append({"role": "user", "content": text})
    history[message.chat.id].append({"role": "assistant", "content": answer})
    bot.reply_to(message, answer)


if __name__ == "__main__":
    log.info("Бот запущен, индекс: %s", memory.index_name)
    bot.infinity_polling()
