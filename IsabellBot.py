import asyncio
import base64
import difflib
import functools
import io
import json
import logging
import os
import random
import re
import signal
import sqlite3
import sys
import unicodedata
import time
import urllib.parse
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import aiohttp
import discord
from discord import app_commands
from openai import AsyncOpenAI
from PIL import Image

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
file_handler = TimedRotatingFileHandler("app.log", when="midnight", interval=1, backupCount=7)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger()
if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

# =============================================================================
# Config (hot-reloadable)
# =============================================================================
def _default_config_path() -> str:
    for cand in ("Config.json", "Isabell.json"):
        if os.path.exists(cand):
            return cand
    return "Config.json"


CONFIG_PATH = os.environ.get("BOT_CONFIG") or _default_config_path()
_config_mtime: float = 0.0


def load_config() -> dict[str, Any]:
    global _config_mtime
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    _config_mtime = os.path.getmtime(CONFIG_PATH)
    return data


try:
    config = load_config()
    logging.info(
        "Loaded config: %s",
        {k: v for k, v in config.items() if k not in {"DiscordToken", "Personality", "OpenAPIKey"}},
    )
    logging.info("Persona chars: %d", len(config.get("Personality") or ""))
    logging.info("=== BOOT: Discord Bot ===")
except Exception as e:
    logging.exception("Failed to load config: %s", e)
    raise

# Conversation history paths
CHANNEL_HISTORY_PATH = config.get("ChannelHistoryPath", "channel_history.json")
DM_HISTORY_DIR = config.get("DMHistoryDir", "dm_history")

# Lore: loaded as plain text, injected into every system prompt
# Two lore texts: the full reference (used by /lore, where depth matters) and a
# compact one injected into every chat turn. The full file is ~9k tokens — sending
# it on every message dominated the API bill, so chat gets the condensed version.
LORE_CONTEXT = ""       # full reference
LORE_CHAT_CONTEXT = ""  # compact, sent with every chat message
_lore_mtimes: dict[str, float] = {}


def _lore_paths() -> tuple[str, str]:
    full = config.get("LorePath", "world_lore.txt")
    return full, (config.get("LoreChatPath") or full)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        logging.exception("Failed to read lore file %s", path)
        return ""


# Retrieval index over the FULL lore: (title, text, matching terms).
# Chat carries only the compact lore; when a message names something from the
# world, the matching section is appended just for that call.
_LORE_CHUNKS: list[tuple[str, str, set[str]]] = []

_LORE_STOP = {
    "the","this","that","they","there","when","what","where","while","their","these","those","some",
    "most","many","each","every","after","before","above","below","during","because","her","his","its",
    "she","and","but","for","not","with","from","into","over","under","been","were","have","has","had",
    "only","more","less","than","then","them","who","how","why","all","any","one","two","new","old",
    "human","humans","women","woman","men","man","people","years","year","time","first","last","other",
    "small","large","black","white","red","blue","green","background","current","player","game",
}


def _split_lore(text: str) -> list[tuple[str, str]]:
    """Split the lore into retrievable chunks: ### subsections where present, else ## sections."""
    chunks: list[tuple[str, str]] = []
    for section in re.split(r"\n(?=## )", text):
        section = section.strip()
        if not section:
            continue
        title = section.split("\n", 1)[0].lstrip("# ").strip()
        subs = re.split(r"\n(?=### )", section)
        if len(subs) > 1:
            if len(subs[0].strip()) > 200:
                chunks.append((title, subs[0].strip()))
            for sub in subs[1:]:
                sub = sub.strip()
                sub_title = sub.split("\n", 1)[0].lstrip("# ").strip()
                chunks.append((f"{title} / {sub_title}", sub))
        else:
            chunks.append((title, section))
    return chunks


def _lore_terms(text: str) -> set[str]:
    """Proper nouns in a chunk — the names a user might reference.

    A single capitalised word only counts when it appears mid-sentence; a capital
    after a period, bullet or line break is just sentence case, not a name.
    """
    terms: set[str] = set()
    # Headings ("### Aeron ...") and bolded labels ("- **Iron Creed:** ...") are
    # where this lore declares its names, so take every capitalised word there.
    for line in re.findall(r"^#{1,4}[ ]*(.+)$", text, re.M) + re.findall(r"\*\*([^*\n]{2,48}?):?\*\*", text):
        terms |= {w.lower() for w in re.findall(r"\b[A-Z][a-z']{3,}\b", line)}
        terms |= {m.lower() for m in re.findall(r"\b[A-Z][a-z']+(?:[ ][A-Z][a-z']+)+\b", line)}
    # Multi-word capitalised phrases anywhere, plus single words capitalised
    # mid-sentence (a capital after a period or line break is just sentence case).
    terms |= {m.lower() for m in re.findall(r"\b[A-Z][a-z']+(?:[ ][A-Z][a-z']+)+\b", text)}
    terms |= {m.lower() for m in re.findall(r"(?<=[a-z,;)] )([A-Z][a-z']{3,})\b", text)}
    return {t for t in terms if t not in _LORE_STOP}


def _build_lore_index():
    """Index the full lore, dropping terms too common to be a useful signal."""
    global _LORE_CHUNKS
    chunks = _split_lore(LORE_CONTEXT) if LORE_CONTEXT else []
    if not chunks:
        _LORE_CHUNKS = []
        return
    # A real proper noun is never written lowercase in the source text. This drops
    # sentence-start capitals ("Female", "Bound", "Horse") that would otherwise
    # match ordinary roleplay and pull in lore nobody asked for.
    lowercase_words = set(re.findall(r"\b[a-z][a-z']{2,}\b", LORE_CONTEXT))
    freq: dict[str, int] = {}
    per_chunk = []
    for title, text in chunks:
        terms = {t for t in _lore_terms(text) if " " in t or t not in lowercase_words}
        per_chunk.append((title, text, terms))
        for t in terms:
            freq[t] = freq.get(t, 0) + 1
    limit = max(2, int(len(chunks) * 0.4))  # a term in >40% of chunks discriminates nothing
    # Weight by rarity: a name unique to one section is a far stronger signal
    # than one sprinkled across several.
    _LORE_CHUNKS = [
        (t, x, {k: 1.0 / freq[k] for k in terms if freq[k] <= limit}) for t, x, terms in per_chunk
    ]
    logging.info(
        "Lore index: %d chunks, %d distinct terms",
        len(_LORE_CHUNKS), len({k for _, _, s in _LORE_CHUNKS for k in s}),  # noqa: E501
    )


def _query_words(q: str) -> set[str]:
    """Words from a query, plus singular forms so 'Vorgath's' and 'drakes' still match."""
    words = set(re.findall(r"[a-z']+", q))
    extra = set()
    for w in words:
        base = re.sub(r"'s$", "", w)
        if base != w:
            extra.add(base)
        for stem in (base, w):
            if stem.endswith("es") and len(stem) > 5:
                extra.add(stem[:-2])
            if stem.endswith("s") and len(stem) > 4:
                extra.add(stem[:-1])
    return words | extra


def retrieve_lore(query: str) -> str:
    """Return full-lore sections matching the query, within the configured budget."""
    if not config.get("LoreRetrievalEnabled", True) or not _LORE_CHUNKS:
        return ""
    q = (query or "").lower()
    if len(q) < 3:
        return ""
    words = _query_words(q)
    scored = []
    for title, text, terms in _LORE_CHUNKS:
        hits = sum(w for t, w in terms.items() if " " not in t and t in words)
        hits += sum(w for t, w in terms.items() if " " in t and t in q)
        if hits:
            scored.append((hits, title, text))
    if not scored:
        return ""
    scored.sort(key=lambda x: (-x[0], len(x[2])))
    budget = float(config.get("LoreRetrievalMaxTokens", 1400))
    picked, titles = [], []
    for hits, title, text in scored[: int(config.get("LoreRetrievalMaxChunks", 2))]:
        cost = len(text) / 3.6
        if cost > budget:
            continue
        picked.append(text)
        titles.append(f"{title}({hits:.2f})")
        budget -= cost
    if picked:
        logging.info("Lore retrieval: %s", ", ".join(titles))
    return "\n\n".join(picked)


def load_lore():
    """(Re)load both lore files. Safe to call repeatedly."""
    global LORE_CONTEXT, LORE_CHAT_CONTEXT, _lore_mtimes
    full_path, chat_path = _lore_paths()
    LORE_CONTEXT = _read_text(full_path) if os.path.exists(full_path) else ""
    LORE_CHAT_CONTEXT = _read_text(chat_path) if os.path.exists(chat_path) else LORE_CONTEXT
    _lore_mtimes = {p: os.path.getmtime(p) for p in {full_path, chat_path} if os.path.exists(p)}
    logging.info(
        "Loaded lore: full=%d chars (%s), chat=%d chars (%s)",
        len(LORE_CONTEXT), full_path, len(LORE_CHAT_CONTEXT), chat_path,
    )
    _build_lore_index()


load_lore()


def _reload_derived_config():
    """Update module-level derived values after a config reload."""
    global IGNORED_USERS, IGNORED_WORDS, OWNER_ID
    global MOD_CHANNEL_ID, MOD_ROLE_ID, TRAP_CHANNEL_ID, EXEMPT_ROLE_IDS
    IGNORED_USERS = set(config.get("IgnoredUsers", []))
    IGNORED_WORDS = {w.lower() for w in config.get("IgnoredWords", [])}
    if "load_lore" in globals():
        load_lore()
    # Owner ID for DM commands and summary delivery (0 = disabled)
    OWNER_ID = int(config.get("SummaryOwnerID", 0))
    # Mod alert channel, mod role, honeypot channel + exempt roles (0/empty = feature disabled)
    MOD_CHANNEL_ID = int(config.get("ModChannelID", 0))
    MOD_ROLE_ID = int(config.get("ModRoleID", 0))
    TRAP_CHANNEL_ID = int(config.get("HoneypotChannelID", 0))
    EXEMPT_ROLE_IDS = {int(r) for r in config.get("HoneypotExemptRoleIDs", [])}


IGNORED_USERS: set[int] = set()
IGNORED_WORDS: set[str] = set()
OWNER_ID: int = 0
MOD_CHANNEL_ID: int = 0
MOD_ROLE_ID: int = 0
TRAP_CHANNEL_ID: int = 0
EXEMPT_ROLE_IDS: set[int] = set()
_reload_derived_config()

# =============================================================================
# Async OpenAI Client (OpenRouter)
# =============================================================================
llm_client = AsyncOpenAI(
    base_url=config["OpenAPIEndpoint"],
    api_key=config.get("OpenAPIKey", ""),
)


def utility_model() -> str | None:
    """Cheaper model for mechanical tasks (summaries, prompt rewriting, FAQ, translation).
    None = use the main model."""
    return config.get("UtilityModel") or None


class LLMResponseError(Exception):
    """Raised when the API returns 200 but no usable completion (e.g. provider error or content flag)."""


async def chat_async(messages: list[dict[str, str]], _retries: int = 3, return_message: bool = False, **kwargs):
    """Run a chat completion with retry + exponential backoff."""
    kwargs.pop("reasoning", None)
    model = kwargs.pop("model", None) or config["OpenaiModel"]
    kwargs["extra_body"] = kwargs.get("extra_body", {})
    kwargs["extra_body"]["reasoning"] = {"enabled": False}

    last_exc: Exception | None = None
    for attempt in range(1, _retries + 1):
        try:
            resp = await llm_client.chat.completions.create(
                messages=messages,
                model=model,
                **kwargs,
            )

            # A 200 with no choices means OpenRouter returned an error payload
            # (upstream provider error, content moderation flag, etc.) rather than
            # a completion. Surface what actually came back instead of an opaque
            # 'NoneType not subscriptable'.
            if not getattr(resp, "choices", None):
                err_payload = getattr(resp, "model_extra", None) or {}
                err_detail = err_payload.get("error") if isinstance(err_payload, dict) else None
                logging.error("LLM returned no choices. error=%r full=%r", err_detail, resp)
                # Provider/moderation errors are deterministic — retrying wastes time.
                raise LLMResponseError(str(err_detail or "no choices returned"))

            msg = resp.choices[0].message
            return msg if return_message else msg.content
        except LLMResponseError:
            # Don't retry — same request will be rejected identically.
            raise
        except Exception as e:
            last_exc = e
            if attempt < _retries:
                wait = 1.5 * (2 ** (attempt - 1))
                logging.warning("chat_async attempt %d/%d failed: %s. Retrying in %.1fs", attempt, _retries, e, wait)
                await asyncio.sleep(wait)
    raise last_exc


# =============================================================================
# Discord Client
# =============================================================================
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
# Joins/leaves in the mod log need the privileged Server Members intent —
# enable it in the developer portal FIRST, then set EnableMembersIntent true.
if config.get("EnableMembersIntent"):
    intents.members = True
# Larger cache so deleted/edited message content is usually available to log.
bot = discord.Client(intents=intents, max_messages=10000)
tree = app_commands.CommandTree(bot)


_background_tasks_started = False


@bot.event
async def on_ready():
    global _background_tasks_started
    logging.info("BOT READY as %s (id=%s)", bot.user, getattr(bot.user, "id", "n/a"))
    logging.info("RUNNING FILE: %s | PID: %s", __file__, os.getpid())
    if not _background_tasks_started:
        _background_tasks_started = True
        bot.add_view(ImageActionsView())  # persistent buttons survive restarts
        bot.loop.create_task(_periodic_save_loop())
        bot.loop.create_task(_daily_summary_scheduler())
        bot.loop.create_task(_config_watch_loop())
        bot.loop.create_task(_faq_answer_loop())
        bot.loop.create_task(_sync_app_commands())
        bot.loop.create_task(_stats_loop())


# =============================================================================
# Small Utilities
# =============================================================================
def clamp_2000(text: str) -> str:
    return (text or "")[:2000]


async def safe_send(channel: discord.abc.Messageable, text: str | None = None, **kwargs):
    try:
        if text is not None:
            return await channel.send(clamp_2000(text), **kwargs)
        return await channel.send(**kwargs)
    except Exception:
        logging.exception("safe_send failed")


def channel_key(message: discord.Message) -> int:
    """Threads share memory with parent; DMs use author ID."""
    if isinstance(message.channel, discord.DMChannel):
        return message.author.id
    parent = getattr(message.channel, "parent", None)
    if parent is not None:
        return parent.id
    return message.channel.id


def is_allowed(message: discord.Message) -> bool:
    if isinstance(message.channel, discord.DMChannel):
        return True
    allowed = set(config.get("AllowedChannels", []))
    parent = getattr(message.channel, "parent", None)
    return (message.channel.id in allowed) or (parent and parent.id in allowed)


def is_ignored(message: discord.Message) -> bool:
    if message.author.id in IGNORED_USERS:
        return True
    text_lower = (message.content or "").lower()
    return any(word in text_lower for word in IGNORED_WORDS)


async def ensure_can_send(message: discord.Message) -> bool:
    if isinstance(message.channel, discord.DMChannel):
        return True
    try:
        me = message.guild.me or await message.guild.fetch_member(bot.user.id)
        perms = message.channel.permissions_for(me)
        if not perms.send_messages:
            logging.warning("No send_messages perm in #%s (%s)", message.channel, message.channel.id)
            return False
        return True
    except Exception:
        logging.exception("Permission check failed; assuming False")
        return False


def _too_similar(a: str, b: str, threshold: float = 0.9) -> bool:
    """True when two replies are near-duplicates (loop detection)."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if not a or not b:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


# =============================================================================
# Rate Limiting
# =============================================================================
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.last_refill_time = time.time()
        self.refill_rate = refill_rate

    def consume(self, tokens: int = 1) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill_time) * self.refill_rate)
        self.last_refill_time = now
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False


# Per-user bucket: 5 images/minute = capacity 5, refill ~0.083/sec
_user_buckets: dict[int, TokenBucket] = {}


def get_user_bucket(user_id: int) -> TokenBucket:
    if user_id not in _user_buckets:
        _user_buckets[user_id] = TokenBucket(capacity=5, refill_rate=5.0 / 60.0)
    return _user_buckets[user_id]


# =============================================================================
# Conversation Memory
# =============================================================================
@dataclass
class Utterance:
    author_id: int
    author_name: str
    content: str
    message_id: int
    ts: float


@dataclass
class ConversationWindow:
    channel_id: int
    turns: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=40))
    utterances: deque[Utterance] = field(default_factory=lambda: deque(maxlen=100))
    summary: str = ""
    is_dm: bool = False


class ConversationManager:
    def __init__(
        self,
        maxlen_turns: int = 40,
        channel_history_path: str | None = None,
        dm_history_dir: str | None = None,
    ):
        self._by_channel: dict[int, ConversationWindow] = {}
        self.maxlen_turns = maxlen_turns
        self.channel_history_path = channel_history_path
        self.dm_history_dir = dm_history_dir
        self._utterance_maxlen = 100
        self._dirty = False
        self._compress_at = maxlen_turns - 2

        if self.dm_history_dir:
            os.makedirs(self.dm_history_dir, exist_ok=True)

        self._load_from_disk()

    # ---------- Persistence ----------

    def _load_from_disk(self):
        self._load_channels()
        self._load_dms()

    def _load_channels(self):
        path = self.channel_history_path
        if not path or not os.path.exists(path):
            logging.info("No channel history file at %s; starting fresh.", path)
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            loaded = 0
            for ch_key, cv_data in data.items():
                try:
                    ch_id = int(cv_data.get("channel_id", ch_key))
                    turns = deque(
                        [tuple(t) for t in cv_data.get("turns", [])],
                        maxlen=self.maxlen_turns,
                    )
                    uttrs: deque[Utterance] = deque(maxlen=self._utterance_maxlen)
                    for u in cv_data.get("utterances", []):
                        uttrs.append(Utterance(
                            author_id=u.get("author_id"),
                            author_name=u.get("author_name", ""),
                            content=u.get("content", ""),
                            message_id=u.get("message_id", 0),
                            ts=u.get("ts", 0.0),
                        ))
                    self._by_channel[ch_id] = ConversationWindow(
                        channel_id=ch_id,
                        turns=turns,
                        utterances=uttrs,
                        summary=cv_data.get("summary", ""),
                        is_dm=bool(cv_data.get("is_dm", False)),
                    )
                    loaded += 1
                except Exception:
                    logging.exception("Failed to load window for channel %r", ch_key)
            logging.info("Loaded %d channel windows from %s", loaded, path)
        except Exception:
            logging.exception("Failed to load channel history from %s", path)

    def _load_dms(self):
        dir_path = self.dm_history_dir
        if not dir_path or not os.path.isdir(dir_path):
            logging.info("No DM history dir at %s; starting fresh.", dir_path)
            return
        loaded = 0
        for fname in os.listdir(dir_path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(dir_path, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    cv_data = json.load(f)
                ch_id = int(cv_data.get("channel_id", os.path.splitext(fname)[0]))
                turns = deque(
                    [tuple(t) for t in cv_data.get("turns", [])],
                    maxlen=self.maxlen_turns,
                )
                uttrs: deque[Utterance] = deque(maxlen=self._utterance_maxlen)
                for u in cv_data.get("utterances", []):
                    uttrs.append(Utterance(
                        author_id=u.get("author_id"),
                        author_name=u.get("author_name", ""),
                        content=u.get("content", ""),
                        message_id=u.get("message_id", 0),
                        ts=u.get("ts", 0.0),
                    ))
                self._by_channel[ch_id] = ConversationWindow(
                    channel_id=ch_id,
                    turns=turns,
                    utterances=uttrs,
                    summary=cv_data.get("summary", ""),
                    is_dm=True,
                )
                loaded += 1
            except Exception:
                logging.exception("Failed to load DM history from %s", fpath)
        logging.info("Loaded %d DM windows from %s", loaded, dir_path)

    def mark_dirty(self):
        self._dirty = True

    def save_if_dirty(self):
        if not self._dirty:
            return
        self._save_channels()
        self._save_dms()
        self._dirty = False

    def force_save(self):
        self._save_channels()
        self._save_dms()
        self._dirty = False

    def _save_channels(self):
        path = self.channel_history_path
        if not path:
            return
        try:
            serializable: dict[str, Any] = {}
            for ch_id, cv in self._by_channel.items():
                if cv.is_dm:
                    continue
                serializable[str(ch_id)] = {
                    "channel_id": ch_id,
                    "turns": [list(t) for t in cv.turns],
                    "utterances": [u.__dict__ for u in cv.utterances],
                    "summary": cv.summary,
                    "is_dm": False,
                }
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            logging.exception("Failed to save channel history to %s", path)

    def _save_dms(self):
        dir_path = self.dm_history_dir
        if not dir_path:
            return
        try:
            for ch_id, cv in self._by_channel.items():
                if not cv.is_dm:
                    continue
                data = {
                    "channel_id": ch_id,
                    "turns": [list(t) for t in cv.turns],
                    "utterances": [u.__dict__ for u in cv.utterances],
                    "summary": cv.summary,
                    "is_dm": True,
                }
                fpath = os.path.join(dir_path, f"{ch_id}.json")
                tmp = fpath + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, fpath)
        except Exception:
            logging.exception("Failed to save DM histories to %s", dir_path)

    # ---------- API ----------

    def get(self, channel_id: int, is_dm: bool | None = None) -> ConversationWindow:
        cv = self._by_channel.get(channel_id)
        if cv is None:
            cv = ConversationWindow(channel_id=channel_id, is_dm=bool(is_dm))
            self._by_channel[channel_id] = cv
            self.mark_dirty()
        elif is_dm is not None and cv.is_dm != is_dm:
            cv.is_dm = is_dm
            self.mark_dirty()
        return cv

    def clear_channel(self, channel_id: int) -> bool:
        """Wipe a channel's conversation history. Returns True if anything was cleared."""
        cv = self._by_channel.get(channel_id)
        if cv is None:
            return False
        cv.turns.clear()
        cv.utterances.clear()
        cv.summary = ""
        self.mark_dirty()
        self.save_if_dirty()  # persist immediately so a restart can't restore the bad state
        return True

    def add_user(
        self,
        channel_id: int,
        is_dm: bool,
        author_id: int,
        author_name: str,
        content: str,
        message_id: int,
    ):
        cv = self.get(channel_id, is_dm=is_dm)
        cv.utterances.append(Utterance(author_id, author_name, content, message_id, time.time()))
        if is_dm:
            cv.turns.append(("user", content))
        else:
            cv.turns.append(("user", f"[{author_name}]: {content}"))
        self.mark_dirty()

    def add_assistant(self, channel_id: int, content: str):
        cv = self.get(channel_id)
        cv.turns.append(("assistant", content))
        self.mark_dirty()

    async def maybe_compress(self, channel_id: int):
        """If the turn window is nearly full, summarize the oldest half."""
        cv = self.get(channel_id)
        if len(cv.turns) < self._compress_at:
            return

        all_turns = list(cv.turns)
        half = len(all_turns) // 2
        old_turns = all_turns[:half]
        keep_turns = all_turns[half:]

        text_block = "\n".join(f"{role}: {content}" for role, content in old_turns)
        existing = f"Previous summary:\n{cv.summary}\n\n" if cv.summary else ""

        system = (
            "Compress the following conversation excerpt into a concise summary paragraph. "
            "Record only facts, events, decisions, user names, and open questions the assistant "
            "needs to continue naturally. Do NOT describe or quote the assistant's writing style, "
            "tone, or recurring phrasing — summarize what happened, never how it was worded. "
            "Write in third person. Keep it under 200 words."
        )
        user_msg = f"{existing}New turns to incorporate:\n{text_block}"

        try:
            new_summary = await chat_async(
                [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
                temperature=0.2,
                max_tokens=300,
                model=utility_model(),
            )
            cv.summary = (new_summary or "").strip()
            cv.turns = deque(
                [tuple(t) for t in keep_turns],
                maxlen=self.maxlen_turns,
            )
            self.mark_dirty()
            logging.info("Compressed %d old turns for channel %s", half, channel_id)
        except Exception:
            logging.exception("Turn compression failed for channel %s", channel_id)

    def build_messages(self, channel_id: int, system_prefix: str) -> list[dict[str, str]]:
        c = self.get(channel_id)
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prefix}]
        if c.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far:\n{c.summary}"})
        msgs.extend({"role": r, "content": t} for r, t in c.turns)
        return msgs


async def _periodic_save_loop():
    while True:
        await asyncio.sleep(30)
        try:
            cm.save_if_dirty()
            ipm.save_if_dirty()
        except Exception:
            logging.exception("Periodic save failed")


cm = ConversationManager(
    channel_history_path=CHANNEL_HISTORY_PATH,
    dm_history_dir=DM_HISTORY_DIR,
)

# =============================================================================
# Image Prompt Memory
# =============================================================================
@dataclass
class ImagePromptRecord:
    channel_id: int
    message_id: int
    user_prompt: str
    final_sd_prompt: str
    negative_prompt: str = ""
    seed: int = -1
    width: int = 0
    height: int = 0
    positive_prefix: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    bot_message_id: int | None = None
    ts: float = 0.0


class ImagePromptMemory:
    """Per-channel image history, persisted so follow-ups and buttons survive restarts."""

    def __init__(self, path: str | None = None, max_per_channel: int = 20):
        self.by_channel: dict[int, list[ImagePromptRecord]] = {}
        self.path = path
        self.max_per_channel = max_per_channel
        self._dirty = False
        self._load()

    def add(self, rec: ImagePromptRecord):
        arr = self.by_channel.setdefault(rec.channel_id, [])
        arr.append(rec)
        del arr[:-self.max_per_channel]
        self._dirty = True

    def last_for_channel(self, channel_id: int) -> ImagePromptRecord | None:
        arr = self.by_channel.get(channel_id, [])
        return arr[-1] if arr else None

    def find_by_message(self, message_id: int | None) -> ImagePromptRecord | None:
        if not message_id:
            return None
        for arr in self.by_channel.values():
            for rec in reversed(arr):
                if rec.bot_message_id == message_id:
                    return rec
        return None

    def save_if_dirty(self):
        if not self._dirty or not self.path:
            return
        try:
            data = {str(ch): [rec.__dict__ for rec in arr] for ch, arr in self.by_channel.items() if arr}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
            self._dirty = False
        except Exception:
            logging.exception("Failed to save image memory to %s", self.path)

    def force_save(self):
        self._dirty = True
        self.save_if_dirty()

    def _load(self):
        if not self.path or not os.path.exists(self.path):
            return
        known = set(ImagePromptRecord.__dataclass_fields__)
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for ch, arr in data.items():
                self.by_channel[int(ch)] = [
                    ImagePromptRecord(**{k: v for k, v in d.items() if k in known}) for d in arr
                ]
            logging.info("Loaded image memory for %d channels", len(self.by_channel))
        except Exception:
            logging.exception("Failed to load image memory from %s", self.path)


ipm = ImagePromptMemory(path=config.get("ImageMemoryPath", "image_memory.json"))

# Latest generated image per channel (base64 PNG) for img2img refinements.
# In-memory only — after a restart, refinements fall back to seed reuse.
_last_image_b64: dict[int, str] = {}

# =============================================================================
# Image Generation (Stable Diffusion)
# =============================================================================
def image_ok(img: Image.Image | None) -> bool:
    if img is None:
        return False
    try:
        w, h = img.size
        return w > 0 and h > 0
    except Exception:
        return False


class SDOfflineError(Exception):
    """The Stable Diffusion backend cannot be reached at all."""


def _sd_base_url() -> str:
    return config["SDURL"].split("/sdapi/")[0]


async def _sd_post(path: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=int(config.get("SDTimeout", 180)))
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(_sd_base_url() + path, json=payload) as r:
                r.raise_for_status()
                return await r.json()
    except aiohttp.ClientConnectorError as e:
        logging.warning("SD backend unreachable: %s", e)
        raise SDOfflineError(str(e)) from e


async def _sd_get(path: str) -> dict | None:
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
            async with s.get(_sd_base_url() + path) as r:
                r.raise_for_status()
                return await r.json()
    except Exception:
        return None


async def sd_generate(
    *,
    prompt: str,
    negative: str,
    seed: int = -1,
    subseed_strength: float = 0.0,
    hires: bool = False,
    batch: int = 1,
    width: int | None = None,
    height: int | None = None,
    positive_prefix: str | None = None,
    init_image_b64: str | None = None,
) -> tuple[list[Image.Image], int, str | None]:
    """txt2img, or img2img when an init image is given.

    Returns (images, seed_used, first_image_b64)."""
    prefix = positive_prefix if positive_prefix is not None else config["SDPositivePrompt"]
    batch = max(1, min(int(config.get("SDMaxBatch", 4)), batch))
    payload = {
        "prompt": prefix + prompt,
        "negative_prompt": negative,
        "steps": config["SDSteps"],
        "width": width or config["SDWidth"],
        "height": height or config["SDHeight"],
        "cfg_scale": config["cfg_scale"],
        "sampler_index": config["SDSampler"],
        "seed": seed,
        "batch_size": batch,
    }
    if config.get("scheduler"):
        payload["scheduler"] = config["scheduler"]
    if subseed_strength > 0:
        payload["subseed"] = -1
        payload["subseed_strength"] = subseed_strength

    endpoint = "/sdapi/v1/txt2img"
    if init_image_b64:
        endpoint = "/sdapi/v1/img2img"
        payload["init_images"] = [init_image_b64]
        payload["denoising_strength"] = float(config.get("SDImg2ImgDenoise", 0.5))
    elif hires:
        payload["enable_hr"] = True
        payload["hr_scale"] = float(config.get("SDUpscaleFactor", 2.0))
        payload["hr_upscaler"] = config.get("SDHiresUpscaler", "Latent")
        payload["denoising_strength"] = 0.4
        # Forge defaults this to None and then tests membership on it, so an API
        # hires request without it fails with "argument of type 'NoneType' is not
        # iterable". "Use same choices" means "keep the base model's modules".
        payload["hr_additional_modules"] = ["Use same choices"]

    try:
        j = await _sd_post(endpoint, payload)
    except SDOfflineError:
        raise
    except Exception as e:
        logging.exception("SD generation failed: %s", e)
        return [], seed, None

    raw_images = (j.get("images") or [])[:batch]
    images: list[Image.Image] = []
    for b in raw_images:
        try:
            images.append(Image.open(io.BytesIO(base64.b64decode(b))))
        except Exception:
            logging.exception("Failed to decode SD image")
    used_seed = seed
    try:
        used_seed = int(json.loads(j.get("info") or "{}").get("seed", seed))
    except Exception:
        pass
    return images, used_seed, (raw_images[0] if raw_images else None)


# One GPU — serialize generations ourselves so a queued request waits with
# feedback instead of burning its HTTP timeout inside the SD backend.
_sd_semaphore: asyncio.Semaphore | None = None
_sd_waiting = 0


def _get_sd_semaphore() -> asyncio.Semaphore:
    global _sd_semaphore
    if _sd_semaphore is None:
        _sd_semaphore = asyncio.Semaphore(int(config.get("SDMaxConcurrent", 1)))
    return _sd_semaphore


async def _progress_updates(status_msg, gen_task: asyncio.Task):
    """Edit the status message with live progress while a generation runs."""
    if status_msg is None:
        return
    try:
        while not gen_task.done():
            await asyncio.sleep(4)
            if gen_task.done():
                return
            j = await _sd_get("/sdapi/v1/progress?skip_current_image=true")
            if not j:
                continue
            pct = int(float(j.get("progress") or 0) * 100)
            if pct <= 0:
                continue
            eta = int(float(j.get("eta_relative") or 0))
            text = f"🎨 {pct}%" + (f" · ~{eta}s left" if eta > 0 else "")
            try:
                await status_msg.edit(content=text)
            except Exception:
                return
    except asyncio.CancelledError:
        return


class ImageActionsView(discord.ui.View):
    """Persistent buttons attached to every generated image."""

    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Redo", emoji="🔁", style=discord.ButtonStyle.secondary, custom_id="imggen:redo")
    async def redo(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "redo")

    @discord.ui.button(label="Variation", emoji="✨", style=discord.ButtonStyle.secondary, custom_id="imggen:vary")
    async def vary(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "vary")

    @discord.ui.button(label="Upscale", emoji="⬆️", style=discord.ButtonStyle.secondary, custom_id="imggen:upscale")
    async def upscale(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "upscale")

    @discord.ui.button(label="Prompt", emoji="📋", style=discord.ButtonStyle.secondary, custom_id="imggen:prompt")
    async def prompt(self, interaction: discord.Interaction, button: discord.ui.Button):
        await _image_button(interaction, "prompt")


async def _image_button(interaction: discord.Interaction, action: str):
    try:
        rec = ipm.find_by_message(interaction.message.id if interaction.message else None)
        if rec is None:
            await interaction.response.send_message("I no longer remember this image, sorry.", ephemeral=True)
            return

        if action == "prompt":
            text = (
                f"**Prompt:**\n```{(rec.final_sd_prompt or '')[:1700]}```\n"
                f"**Negative:** {(rec.negative_prompt or '')[:300]}\n"
                f"**Seed:** `{rec.seed}`"
            )
            await interaction.response.send_message(text, ephemeral=True)
            return

        if not images_enabled():
            await interaction.response.send_message(
                config.get("ImageDisabledNotice", "Image generation is currently disabled."),
                ephemeral=True)
            return
        blocked = image_prompt_blocked(rec.final_sd_prompt) or image_prompt_blocked(rec.user_prompt)
        if blocked:
            await interaction.response.send_message(config.get(
                "ImageRefusalMessage",
                "No. That is not something I will ever draw, and the moderators have been notified."), ephemeral=True)
            await refuse_image_request(interaction.channel, interaction.user.id,
                                       interaction.user.display_name, blocked, rec.final_sd_prompt)
            return
        if not get_user_bucket(interaction.user.id).consume():
            await interaction.response.send_message("You're requesting images too fast — slow down a bit.", ephemeral=True)
            return

        labels = {"redo": "Rolling a fresh take…", "vary": "Painting a variation…", "upscale": "Upscaling…"}
        await interaction.response.send_message(f"🎨 {labels.get(action, 'Working…')}")
        status_msg = await interaction.original_response()

        kwargs: dict[str, Any] = dict(
            ch_id=rec.channel_id,
            user_prompt=rec.user_prompt,
            sd_prompt=rec.final_sd_prompt,
            neg=rec.negative_prompt,
            positive_prefix=rec.positive_prefix or None,
            width=rec.width or None,
            height=rec.height or None,
            requested_by=interaction.user.display_name,
            requester_id=interaction.user.id,
            status_msg=status_msg,
        )
        if action == "vary":
            kwargs.update(seed=rec.seed, subseed_strength=float(config.get("SDVariationStrength", 0.35)))
        elif action == "upscale":
            kwargs.update(seed=rec.seed, hires=True)
        asyncio.create_task(run_image_job(interaction.channel, **kwargs))
    except Exception:
        logging.exception("Image button %s failed", action)


async def run_image_job(
    channel,
    *,
    ch_id: int,
    user_prompt: str,
    sd_prompt: str,
    neg: str,
    seed: int = -1,
    subseed_strength: float = 0.0,
    hires: bool = False,
    batch: int = 1,
    width: int | None = None,
    height: int | None = None,
    positive_prefix: str | None = None,
    init_image_b64: str | None = None,
    requested_by: str = "",
    requester_id: int = 0,
    trigger_message_id: int = 0,
    status_msg=None,
):
    """Queue a generation, show progress, deliver the result with action buttons."""
    global _sd_waiting
    try:
        # Last line of defence: every image path funnels through here, including
        # buttons, refinements and LLM-rewritten prompts.
        if not images_enabled():
            await image_unavailable(channel)
            return
        blocked = image_prompt_blocked(sd_prompt) or image_prompt_blocked(user_prompt)
        if blocked:
            await refuse_image_request(channel, requester_id, requested_by or "unknown",
                                       blocked, sd_prompt)
            return
        sem = _get_sd_semaphore()
        queued = sem.locked()
        if queued:
            _sd_waiting += 1
            await safe_send(channel, f"🎨 The easel is busy — you're #{_sd_waiting} in line.")
        try:
            async with sem:
                if queued:
                    _sd_waiting = max(0, _sd_waiting - 1)
                async with channel.typing():
                    gen_task = asyncio.create_task(sd_generate(
                        prompt=sd_prompt, negative=neg, seed=seed,
                        subseed_strength=subseed_strength, hires=hires, batch=batch,
                        width=width, height=height, positive_prefix=positive_prefix,
                        init_image_b64=init_image_b64,
                    ))
                    progress_task = asyncio.create_task(_progress_updates(status_msg, gen_task))
                    try:
                        images, seed_used, first_b64 = await gen_task
                    finally:
                        progress_task.cancel()
        except SDOfflineError:
            await safe_send(channel, config.get("SDOfflineNotice", "The image engine is offline right now — try again later."))
            return

        images = [im for im in images if image_ok(im)]
        if not images:
            await safe_send(channel, "I couldn't render that image — try tweaking the description.")
            return

        files = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            files.append(discord.File(buf, filename=f"output_{i + 1}.png"))

        content = f"🎨 for **{requested_by}** — use the buttons to iterate." if requested_by else None
        sent = await safe_send(channel, content, files=files, view=ImageActionsView())

        if status_msg is not None:
            try:
                await status_msg.delete()
            except Exception:
                pass

        if first_b64:
            _last_image_b64[ch_id] = first_b64
        ipm.add(ImagePromptRecord(
            channel_id=ch_id,
            message_id=trigger_message_id,
            user_prompt=user_prompt,
            final_sd_prompt=sd_prompt,
            negative_prompt=neg,
            seed=seed_used,
            width=width or 0,
            height=height or 0,
            positive_prefix=positive_prefix or "",
            meta={"by": requested_by},
            bot_message_id=getattr(sent, "id", None),
            ts=time.time(),
        ))
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Image job failed")
        await safe_send(channel, "Oops — image generation hit a snag.")


# =============================================================================
# SD Prompt Builder
# =============================================================================
async def compile_sd_prompt(user_text: str) -> str:
    max_chars = int(config.get("ImagePromptMaxChars", 1600))
    name = (config.get("Name") or "the assistant").strip()

    system = (
        "You are an expert prompt engineer for Pony Realism SDXL models.\n\n"
        "Rewrite the USER PROMPT into ONE comma-separated line of tags optimized for Pony Realistic SDXL.\n\n"
        "RULES:\n"
        "- Output exactly one line of pure tags. No quotes, no explanations.\n"
        f"- Stay under {max_chars} characters.\n"
        "- Amplify and clarify every visual and aesthetic element from the user's description.\n"
        "- Use ( ) with weights for emphasis, e.g. (detailed eyes:1.3)\n"
        "- Add relevant body/lighting/camera tags when implied by the scene.\n"
        "- NEVER add characters, locations, or elements not implied by the user prompt.\n"
        f"- NEVER mention {name} or any persona metadata unless the USER PROMPT explicitly references it.\n"
    )

    user = f"USER PROMPT:\n{user_text}"
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        tok_budget = max(256, min(2000, max_chars // 3))
        raw = await chat_async(msgs, temperature=0.0, max_tokens=tok_budget, model=utility_model())
        raw = (raw or "").strip().strip("`")
        raw = re.sub(r"\((\d(?:\.\d+)?)\)\s*([^,()\n]+)", lambda m: f"({m.group(2).strip()}:{m.group(1)})", raw)
        return raw[:max_chars]
    except Exception:
        logging.exception("LLM prompt compose failed; returning user text")
        return (user_text or "")[:max_chars]


async def refine_image_prompt(last: ImagePromptRecord, followup_text: str) -> dict[str, str]:
    max_chars = int(config.get("ImagePromptMaxChars", 1600))
    name = (config.get("Name") or "the assistant").strip()

    system = (
        "Refine a Stable Diffusion prompt based on a follow-up instruction.\n"
        "- Preserve subject, style, and descriptors from the previous prompt.\n"
        "- Merge ONLY new instructions from the follow-up.\n"
        f"- Do NOT introduce {name} or any persona unless explicitly mentioned.\n"
        f"- Keep under {max_chars} chars.\n"
        '- Output JSON: {"prompt":"...","negative":"..."}.'
    )
    user = (
        f"Previous: {last.final_sd_prompt}\n"
        f"Negative: {last.negative_prompt}\n"
        f"Follow-up: {followup_text}"
    )
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        raw = await chat_async(msgs, temperature=0.0, max_tokens=max(256, min(2000, max_chars // 3)), model=utility_model())
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", (raw or "").strip())
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("bad json")
        data["prompt"] = (data.get("prompt", last.final_sd_prompt) or "")[:max_chars]
        return data
    except Exception:
        logging.warning("Refine parse failed; fallback append")
        return {
            "prompt": f"{last.final_sd_prompt}, {followup_text}"[:max_chars],
            "negative": last.negative_prompt,
        }


# =============================================================================
# Image safety
# =============================================================================
# Two independent controls:
#   1. ImageGenerationEnabled — a master switch that disables all drawing.
#   2. A hard refusal filter for prompts seeking sexualised minors, enforced on
#      BOTH the raw user text and the final prompt sent to Stable Diffusion.
#      The LLM rewrite sits between those two, so checking only one is a bypass.
# The built-in term list cannot be removed or overridden by config; config may
# only add to it.
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                       "@": "a", "$": "s", "!": "i"})

_BLOCK_TERMS_BUILTIN = frozenset({
    "child", "children", "childlike", "childish", "kid", "kids", "minor", "minors",
    "underage", "under age", "under 18", "preteen", "pre teen", "tween", "toddler",
    "infant", "newborn", "baby", "babies", "babyface", "juvenile", "prepubescent",
    "pubescent", "before puberty", "not yet developed", "undeveloped body",
    "loli", "lolis", "lolicon", "lolita", "shota", "shotacon", "toddlercon", "jailbait",
    "schoolgirl", "school girl", "schoolboy", "school boy", "grade school",
    "elementary school", "kindergarten", "middle school", "high school",
    "teen", "teens", "teenage", "teenager", "adolescent",
    "young girl", "young boy", "young one", "little girl", "little boy",
    "small girl", "small boy", "flat chested child", "youngster",
})
# Terms distinctive enough to catch even when spaced or punctuated apart
# ("l.o.l.i", "l o l i") by matching the de-punctuated text.
_BLOCK_TERMS_SQUASHED = frozenset({
    "loli", "lolicon", "shota", "shotacon", "toddlercon", "jailbait",
    "preteen", "underage", "prepubescent",
})
_AGE_NUM_RE = re.compile(
    r"\b(\d{1,2})\s*(?:y\.?o\.?\b|yo\b|yrs?\b|years?\b)(?:\s*old)?|\baged?\s*[:=]?\s*(\d{1,2})\b"
)


def _obfuscated(term: str, text: str) -> bool:
    """True if `term` appears deliberately broken up, e.g. "l.o.l.i" or "l o l i".

    Collapsing all whitespace instead would make "lol is" match "loli", which it
    did against real traffic — "lol" is far too common to treat that way.
    """
    punct = r"[._\-*+~|]+".join(re.escape(c) for c in term)
    spaced = r"\s+".join(re.escape(c) for c in term)
    return bool(re.search(rf"\b{punct}\b", text) or re.search(rf"\b{spaced}\b", text))


def _normalize_for_filter(text: str) -> tuple[str, str, str]:
    """Return (leet-folded, de-punctuated, digit-preserving) forms of the text.

    Leet folding maps digits onto letters so "l0li" is caught, which also
    destroys real numbers — so ages are matched against the untouched form.
    """
    base = unicodedata.normalize("NFKD", (text or "").lower())
    base = "".join(c for c in base if not unicodedata.combining(c))
    folded = re.sub(r"[^a-z0-9]+", " ", base.translate(_LEET)).strip()
    plain = re.sub(r"[^a-z0-9]+", " ", base).strip()
    return folded, folded.replace(" ", ""), plain


def image_prompt_blocked(text: str) -> str | None:
    """The matched term if this prompt must be refused, else None."""
    if not text:
        return None
    norm, squashed, plain = _normalize_for_filter(text)
    folded_raw = unicodedata.normalize("NFKD", (text or "").lower()).translate(_LEET)
    terms = set(_BLOCK_TERMS_BUILTIN) | {
        str(t).lower().strip() for t in (config.get("ImageBlockExtraTerms") or []) if str(t).strip()
    }
    for term in terms:
        if " " in term:
            if term in norm:
                return term
        elif re.search(rf"\b{re.escape(term)}\b", norm):
            return term
    for term in _BLOCK_TERMS_SQUASHED:
        if _obfuscated(term, folded_raw):
            return term
    limit = int(config.get("ImageBlockAgeUnder", 18))
    for m in _AGE_NUM_RE.finditer(plain):
        num = m.group(1) or m.group(2)
        if num is not None and int(num) < limit:
            return f"age {num}"
    return None


async def refuse_image_request(channel, uid: int, name: str, matched: str, text: str):
    """Refuse a blocked prompt: tell the user, record it, alert the mods."""
    logging.warning("Image prompt REFUSED | user=%s (%s) | matched=%r | text=%r",
                    name, uid, matched, (text or "")[:200])
    try:
        add_user_record(uid, "blocked_image_prompt", f"matched '{matched}': {(text or '')[:200]}")
    except Exception:
        logging.exception("Could not record blocked prompt")
    if config.get("ImageBlockAlertMods", True):
        try:
            await _flag_to_mods(
                "Blocked image prompt",
                f"User: **{name}** ({uid})\nMatched: `{matched}`\nPrompt: {(text or '')[:300]}",
            )
        except Exception:
            logging.exception("Could not alert mods about blocked prompt")
    await safe_send(channel, config.get(
        "ImageRefusalMessage",
        "No. That is not something I will ever draw, and the moderators have been notified.",
    ))


# ---- Chat safety -----------------------------------------------------------
# Images can refuse on any age word, because no legitimate prompt needs one.
# Chat cannot: this game's roleplay is about breeding and offspring, so "child",
# "children" and "baby" occur constantly and innocently. So chat is tiered:
#   1. terms with no innocent use               -> always refuse
#   2. words describing a minor as a person     -> refuse when the message is sexual
#   3. a stated age under 18                    -> refuse when the message is sexual
#   4. offspring words (child/kid/baby)         -> refuse only when a hard sexual
#      term sits within a few words of them, which separates "you will bear my
#      child" from "fuck the child".
_CHAT_ALWAYS_BLOCK = frozenset({
    "loli", "lolis", "lolicon", "lolita", "shota", "shotacon", "toddlercon",
    "jailbait", "jail bait", "pedo", "pedophile", "paedophile", "pedophilia",
    "underage", "under age", "preteen", "pre teen", "prepubescent", "child porn",
    "childporn", "csam", "child sex", "sex with a child", "sex with children",
})
_MINOR_DESCRIPTORS = frozenset({
    "young girl", "young boy", "little girl", "little boy", "small girl", "small boy",
    "schoolgirl", "school girl", "schoolboy", "school boy", "teen", "teens",
    "teenage", "teenager", "adolescent", "toddler", "infant", "newborn",
    "grade school", "elementary school", "kindergarten", "middle school",
    "youngster", "minor girl", "minor boy",
})
_AMBIGUOUS_OFFSPRING = ("child", "children", "kid", "kids", "baby", "babies")
_SEXUAL_RE = re.compile(
    r"\b(fuck\w*|cock|dick|pussy|cunt|cum\w*|semen|breed\w*|naked|nude|sex|sexual|horny|"
    r"slut\w*|whore|virgin|penetrat\w*|rape|raping|impregnat\w*|tits|breasts|nipples|"
    r"moan\w*|orgasm\w*|aroused|erect\w*|thrust\w*|mount\w*|suck\w*|lick\w*|anal|oral|"
    r"blowjob|creampie|ravish\w*|deflower\w*|molest\w*|seduc\w*|undress\w*|strip\w*|"
    r"grope\w*|fondl\w*|caress\w*|bondage|submissive|dominate|lust\w*|arousal)\b"
    # Euphemisms only count with an object, so "take a look" stays innocent while
    # "take her hard" does not.
    r"|\b(take|takes|taking|took|claim\w*|bed|ride|rides|riding|use|using|touch\w*|"
    r"kiss\w*|have|had)\s+(you|her|him|me|them|his|their)\b"
    r"|\bmake love\b|\bhave my way\b|\bspread (her|your|his) legs\b", re.IGNORECASE)
# "you're 12", "i am 15", "she is 13" — an age with no "years old" attached.
_BARE_AGE_RE = re.compile(
    r"\b(?:you re|youre|you are|i m|im|i am|she is|shes|he is|hes)\s+(\d{1,2})\b")
# Deliberately narrower: these must sit *next to* an offspring word to trigger.
_HARD_SEXUAL = frozenset({
    "fuck", "fucks", "fucking", "fucked", "rape", "raped", "raping", "penetrate",
    "penetrated", "penetrating", "cock", "dick", "pussy", "cunt", "anal", "oral",
    "blowjob", "cum", "cumming", "suck", "sucking", "lick", "licking", "thrust",
    "thrusting", "deflower", "molest", "molesting", "slut", "whore", "horny",
})


def chat_message_blocked(text: str, context: str = "") -> str | None:
    """The matched reason if this chat text must be refused, else None.

    `context` is the recent conversation. A minor established a few turns earlier
    ("roleplay as a 15 year old") is still a minor when the sexual turn arrives,
    so age indicators are searched across the exchange while the sexual trigger
    must be in the current message.
    """
    if not config.get("ChatFilterEnabled", True) or not text:
        return None
    norm, squashed, plain = _normalize_for_filter(text)
    if context:
        c_norm, _, c_plain = _normalize_for_filter(context)
        scope_norm, scope_plain = f"{c_norm} {norm}", f"{c_plain} {plain}"
    else:
        scope_norm, scope_plain = norm, plain
    extra = {str(t).lower().strip() for t in (config.get("ChatBlockExtraTerms") or []) if str(t).strip()}
    for term in set(_CHAT_ALWAYS_BLOCK) | extra:
        if (term in norm) if " " in term else re.search(rf"\b{re.escape(term)}\b", norm):
            return term
    folded_raw = unicodedata.normalize("NFKD", (text or "").lower()).translate(_LEET)
    for term in ("loli", "lolicon", "shota", "shotacon", "toddlercon", "jailbait", "pedo"):
        if _obfuscated(term, folded_raw):
            return term

    # "she is 13", "i am 15" — a person's stated age needs no sexual context to be
    # disqualifying here. ("the game is 4 years old" does not match: this pattern
    # requires a personal pronoun.)
    limit = int(config.get("ImageBlockAgeUnder", 18))
    for m in _BARE_AGE_RE.finditer(scope_plain):
        if int(m.group(1)) < limit:
            return f"stated age {m.group(1)}"

    if not _SEXUAL_RE.search(norm):
        return None

    for term in _MINOR_DESCRIPTORS:
        if (term in scope_norm) if " " in term else re.search(rf"\b{re.escape(term)}\b", scope_norm):
            return f"{term} + sexual context"
    for m in _AGE_NUM_RE.finditer(scope_plain):
        num = next((g for g in m.groups() if g), None)
        if num is not None and int(num) < limit:
            return f"age {num} + sexual context"

    words = norm.split()
    hard = [i for i, w in enumerate(words) if w in _HARD_SEXUAL]
    if hard:
        window = int(config.get("ChatOffspringProximity", 3))
        for i, w in enumerate(words):
            if w in _AMBIGUOUS_OFFSPRING and any(abs(i - j) <= window for j in hard):
                return f"{w} near sexual term"
    return None


async def refuse_chat(channel, uid: int, name: str, matched: str, text: str, source: str):
    """Refuse a blocked chat message or model reply."""
    logging.warning("Chat REFUSED (%s) | user=%s (%s) | matched=%r | text=%r",
                    source, name, uid, matched, (text or "")[:200])
    try:
        add_user_record(uid, f"blocked_chat_{source}", f"matched '{matched}': {(text or '')[:200]}")
    except Exception:
        logging.exception("Could not record blocked chat")
    if config.get("ChatBlockAlertMods", True):
        try:
            await _flag_to_mods(
                f"Blocked chat ({source})",
                f"User: **{name}** ({uid})\nMatched: `{matched}`\nText: {(text or '')[:300]}",
            )
        except Exception:
            logging.exception("Could not alert mods about blocked chat")
    await safe_send(channel, config.get(
        "ChatRefusalMessage",
        "No. I will not go there, and the moderators have been notified."))


def images_enabled() -> bool:
    return bool(config.get("ImageGenerationEnabled", True))


async def image_unavailable(channel) -> None:
    await safe_send(channel, config.get(
        "ImageDisabledNotice", "Image generation is currently disabled."))


# =============================================================================
# Image tool (model-decided drawing)
# =============================================================================
# Keyword matching catches explicit requests; this catches the rest — phrasings
# in any language, and "show me what she'd look like". The model also writes the
# SD prompt itself using the conversation, replacing the separate compose call.
IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": (
            "Draw and post a picture. Call this ONLY when the user is asking to be shown "
            "or drawn something. Never call it for ordinary conversation, roleplay narration, "
            "or when the user is merely describing or commenting on something visual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "A detailed comma-separated Stable Diffusion prompt for the image the user "
                        "wants, using the conversation for any context they left implicit."
                    ),
                },
                "aspect": {"type": "string", "enum": ["square", "portrait", "landscape"]},
            },
            "required": ["prompt"],
        },
    },
}


def image_tools_for(message: discord.Message):
    """The tool list for this message, or None when model-decided drawing is off."""
    if not config.get("ImageToolEnabled", False) or not images_enabled():
        return None
    if not _image_channel_allowed(message.channel):
        return None
    return [IMAGE_TOOL]


async def run_tool_image(message: discord.Message, args: dict) -> bool:
    """Act on a generate_image tool call. Returns True if a job was started."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return False
    if not images_enabled():
        await image_unavailable(message.channel)
        return True
    blocked = image_prompt_blocked(prompt)
    if blocked:
        await refuse_image_request(message.channel, message.author.id,
                                   message.author.display_name, blocked, prompt)
        return True
    if not get_user_bucket(message.author.id).consume():
        await safe_send(message.channel, "You're requesting images too fast — slow down a bit.")
        return True
    width = height = None
    aspect = (args.get("aspect") or "").lower()
    if aspect in ("portrait", "landscape"):
        size = config.get("SDPortraitSize" if aspect == "portrait" else "SDLandscapeSize") or []
        if len(size) == 2:
            width, height = int(size[0]), int(size[1])
    status = await safe_send(message.channel, "Hang on while I sketch that for you…")
    logging.info("Image tool fired | ch=%s | prompt=%r", channel_key(message), prompt[:90])
    asyncio.create_task(run_image_job(
        message.channel,
        ch_id=channel_key(message),
        user_prompt=message.content or "",
        sd_prompt=prompt[: int(config.get("ImagePromptMaxChars", 1600))],
        neg=config.get("SDNegativePrompt", "(lowres, blurry, deformed)"),
        width=width,
        height=height,
        requested_by=message.author.display_name,
        requester_id=message.author.id,
        trigger_message_id=message.id,
        status_msg=status,
    ))
    return True


# =============================================================================
# Heuristics
# =============================================================================
_IMAGE_TRIGGER_RE = re.compile(
    r"\b(draw|paint|sketch|illustrate|render|generate an image|make a picture)\b",
    re.IGNORECASE,
)
EXACT_TRIGGERS = ("draw exact", "image exact", "img exact", "exact:")

FOLLOWUP_STARTS = (
    "same", "again", "keep", "also", "but now", "make it",
    "change", "adjust", "brighter", "darker", "add",
)


def is_exact_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EXACT_TRIGGERS)


# Users often send bare SD prompt fragments as follow-ups — "(Translucent skin:1.6)",
# "Labia spreading:1.3" — with no trigger word at all. Measured against real traffic,
# these were the bulk of the requests keyword matching missed.
_SD_WEIGHT_RE = re.compile(r"\([^()\n]{2,60}:\s*\d(?:\.\d+)?\)|\b[a-z][a-z ]{2,40}:\s*\d\.\d\b", re.IGNORECASE)


def looks_like_sd_syntax(text: str) -> bool:
    t = (text or "").strip()
    if not t or "http://" in t or "https://" in t:
        return False
    return bool(_SD_WEIGHT_RE.search(t))


def looks_like_image_request(text: str) -> bool:
    t = (text or "").strip()
    if bool(_IMAGE_TRIGGER_RE.search(t)) or t.lower().startswith(("img:", "image:", "art:")):
        return True
    return looks_like_sd_syntax(t)


def _looks_like_tag_prompt(text: str) -> bool:
    """Comma-heavy tag lists are already SD-ready — skip the LLM rewrite."""
    if not config.get("SkipRewriteForTagPrompts", True):
        return False
    return (text or "").count(",") >= 5


def looks_like_followup(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(t.startswith(s) for s in FOLLOWUP_STARTS)


def should_route_to_image_followup(message: discord.Message) -> bool:
    # Replying to ANY of the bot's images routes to refinement of that image.
    if message.reference and ipm.find_by_message(message.reference.message_id):
        return True
    last = ipm.last_for_channel(channel_key(message))
    if not last:
        return False
    if not looks_like_followup(message.content or ""):
        return False
    window = int(config.get("ImageFollowupWindowSec", 600))
    if last.ts <= 0 or (time.time() - last.ts > window):
        return False
    return True


# =============================================================================
# System Prompt
# =============================================================================
def build_system_prefix(query: str = "") -> str:
    name = config.get("Name", "Assistant")
    persona = (config.get("Personality") or "").strip()

    parts = [f"You are {name}."]
    if persona:
        parts.append(f"\nStay in character as {name}:\n{persona}")
    if LORE_CHAT_CONTEXT:
        parts.append(f"\n\nWorld knowledge (use this to answer questions about the world):\n{LORE_CHAT_CONTEXT}")
    detail = retrieve_lore(query)
    if detail:
        parts.append(f"\n\nRelevant lore detail for this message:\n{detail}")
    return "\n".join(parts)


# =============================================================================
# User Records (SQLite)
# =============================================================================
# Per-user incident history: auto-recorded flags/honeypot trips plus manual
# mod notes. Queried via !record / shown in !whois. No automated actions yet.
_db: sqlite3.Connection | None = None


def get_db() -> sqlite3.Connection:
    global _db
    if _db is None:
        _db = sqlite3.connect(config.get("DatabasePath", "isabot.db"))
        _db.execute("PRAGMA journal_mode=WAL")
        _db.execute(
            """CREATE TABLE IF NOT EXISTS user_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                detail TEXT DEFAULT '',
                moderator_id INTEGER DEFAULT 0,
                ts REAL NOT NULL
            )"""
        )
        _db.execute("CREATE INDEX IF NOT EXISTS idx_user_records_user ON user_records(user_id)")
        _db.execute("CREATE TABLE IF NOT EXISTS highlights (message_id INTEGER PRIMARY KEY, ts REAL NOT NULL)")
        _db.execute(
            """CREATE TABLE IF NOT EXISTS user_xp (
                user_id INTEGER PRIMARY KEY,
                name TEXT DEFAULT '',
                xp INTEGER NOT NULL DEFAULT 0,
                level INTEGER NOT NULL DEFAULT 0,
                messages INTEGER NOT NULL DEFAULT 0,
                last_award REAL NOT NULL DEFAULT 0
            )"""
        )
        _db.commit()
    return _db


def add_user_record(user_id: int, kind: str, detail: str = "", moderator_id: int = 0):
    try:
        db = get_db()
        db.execute(
            "INSERT INTO user_records (user_id, kind, detail, moderator_id, ts) VALUES (?, ?, ?, ?, ?)",
            (int(user_id), kind, (detail or "")[:500], int(moderator_id), time.time()),
        )
        db.commit()
    except Exception:
        logging.exception("Failed to add user record")


def get_user_records(user_id: int, limit: int = 25) -> list[tuple]:
    try:
        cur = get_db().execute(
            "SELECT kind, detail, moderator_id, ts FROM user_records WHERE user_id = ? ORDER BY ts DESC LIMIT ?",
            (int(user_id), limit),
        )
        return cur.fetchall()
    except Exception:
        logging.exception("Failed to read user records")
        return []


def summarize_user_record_counts(user_id: int) -> str:
    try:
        cur = get_db().execute(
            "SELECT kind, COUNT(*) FROM user_records WHERE user_id = ? GROUP BY kind ORDER BY COUNT(*) DESC",
            (int(user_id),),
        )
        parts = [f"{n}× {kind.replace('_', ' ')}" for kind, n in cur.fetchall()]
        return ", ".join(parts)
    except Exception:
        logging.exception("Failed to summarize user records")
        return ""


def _parse_user_ref(arg: str) -> int | None:
    """Accept a raw user ID or a <@mention>."""
    m = re.match(r"^<@!?(\d+)>$", (arg or "").strip()) or re.match(r"^(\d+)$", (arg or "").strip())
    return int(m.group(1)) if m else None


# =============================================================================
# XP / Leveling
# =============================================================================
# MEE6-style: 15-25 XP per message with a per-user cooldown, so chatting
# earns and spamming doesn't. Level N -> N+1 costs 5N² + 50N + 100 XP.
def xp_needed_for(level: int) -> int:
    return 5 * level * level + 50 * level + 100


def level_progress(total_xp: int) -> tuple[int, int, int]:
    """Returns (level, xp_into_level, xp_needed_for_next)."""
    level = 0
    remaining = int(total_xp)
    while remaining >= xp_needed_for(level):
        remaining -= xp_needed_for(level)
        level += 1
    return level, remaining, xp_needed_for(level)


_DEFAULT_LEVELUP_MESSAGES = ["🎉 **{name}** reached level **{level}**!"]


async def award_xp(message: discord.Message):
    try:
        if not config.get("XPEnabled", True):
            return
        if message.channel.id in set(config.get("XPExcludedChannels", [])):
            return
        now = time.time()
        db = get_db()
        row = db.execute(
            "SELECT xp, level, last_award FROM user_xp WHERE user_id = ?", (message.author.id,)
        ).fetchone()
        if row and now - row[2] < int(config.get("XPCooldownSec", 60)):
            return
        gain = random.randint(int(config.get("XPPerMessageMin", 15)), int(config.get("XPPerMessageMax", 25)))
        old_xp, old_level = (row[0], row[1]) if row else (0, 0)
        new_xp = old_xp + gain
        new_level, _, _ = level_progress(new_xp)
        db.execute(
            """INSERT INTO user_xp (user_id, name, xp, level, messages, last_award)
               VALUES (?, ?, ?, ?, 1, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 name = excluded.name, xp = excluded.xp, level = excluded.level,
                 messages = user_xp.messages + 1, last_award = excluded.last_award""",
            (message.author.id, message.author.display_name, new_xp, new_level, now),
        )
        db.commit()
        if new_level > old_level:
            await _handle_level_up(message, old_level, new_level)
    except Exception:
        logging.exception("award_xp failed")


async def _handle_level_up(message: discord.Message, old_level: int, new_level: int):
    rewards = {int(k): int(v) for k, v in (config.get("XPRoleRewards") or {}).items()}
    crossed = [lvl for lvl in sorted(rewards) if old_level < lvl <= new_level]
    reached_tier = crossed[-1] if crossed else None

    # With XPAnnounceRanksOnly, ordinary level-ups (no rank crossed) are silent.
    if reached_tier is None and config.get("XPAnnounceRanksOnly"):
        return

    # Announcement: tier-specific line when a rank was just reached, else generic.
    template = None
    if reached_tier is not None:
        template = (config.get("XPTierMessages") or {}).get(str(reached_tier))
    if not template:
        template = str(random.choice(config.get("LevelUpMessages") or _DEFAULT_LEVELUP_MESSAGES))
    try:
        text = template.format(name=message.author.display_name, level=new_level)
    except Exception:
        text = f"🎉 {message.author.display_name} reached level {new_level}!"
    channel = message.channel
    ann_id = int(config.get("XPAnnounceChannelID", 0))
    if ann_id:
        channel = bot.get_channel(ann_id) or channel
    await safe_send(channel, text)

    # Rank roles: promotion is a SWAP — grant the highest earned rank and drop
    # lower ranks, so each member wears exactly one. Roles for tiers ABOVE the
    # earned one are never touched (protects manually-granted top roles).
    member = message.author
    if message.guild and isinstance(member, discord.Member) and rewards:
        earned = [lvl for lvl in sorted(rewards) if lvl <= new_level]
        if earned:
            try:
                member_role_ids = {r.id for r in member.roles}
                # A member already holding a HIGHER rank role (e.g. a manually
                # granted top tier) never gets lower ranks pinned on them.
                higher_held = [lvl for lvl in rewards if lvl > earned[-1] and rewards[lvl] in member_role_ids]
                top_tier = max(higher_held) if higher_held else earned[-1]
                if not higher_held:
                    to_add = message.guild.get_role(rewards[earned[-1]])
                    if to_add and to_add not in member.roles:
                        await member.add_roles(to_add, reason=f"Level {earned[-1]} rank")
                lower_ids = {rewards[lvl] for lvl in rewards if lvl < top_tier}
                to_remove = [r for r in member.roles if r.id in lower_ids]
                if to_remove:
                    await member.remove_roles(*to_remove, reason="Rank promotion")
            except Exception:
                logging.exception("Failed to update rank roles")

    # Top-tier ceremony: commemorative portrait.
    star_level = int(config.get("XPStarLevel", 0))
    if star_level and reached_tier == star_level and config.get("XPStarPortraitPrompt"):
        try:
            asyncio.create_task(run_image_job(
                channel,
                ch_id=channel_key(message),
                user_prompt=f"Commemorative portrait for {member.display_name}",
                sd_prompt=str(config["XPStarPortraitPrompt"]),
                neg=config.get("SDNegativePrompt", "(lowres, blurry, deformed)"),
                requested_by=member.display_name,
                requester_id=member.id,
            ))
        except Exception:
            logging.exception("Star portrait failed")


async def handle_xp_command(message: discord.Message) -> bool:
    """Public commands, usable by anyone anywhere: !rank / !level, !top / !leaderboard."""
    text = (message.content or "").strip()
    parts = text.split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    if cmd not in ("!rank", "!level", "!top", "!leaderboard"):
        return False
    if not config.get("XPEnabled", True):
        return False
    arg = parts[1].strip() if len(parts) > 1 else ""
    db = get_db()

    if cmd in ("!rank", "!level"):
        target = _parse_user_ref(arg) if arg else message.author.id
        if not target:
            target = message.author.id
        row = db.execute("SELECT name, xp, messages FROM user_xp WHERE user_id = ?", (target,)).fetchone()
        if not row:
            who = "You haven't" if target == message.author.id else "They haven't"
            await safe_send(message.channel, f"{who} earned any XP yet — join the conversation!")
            return True
        name, xp, msgs = row
        level, progress, needed = level_progress(xp)
        rank = db.execute("SELECT COUNT(*) + 1 FROM user_xp WHERE xp > ?", (xp,)).fetchone()[0]
        await safe_send(
            message.channel,
            f"**{name}** — Level **{level}** · Rank **#{rank}**\n"
            f"XP: {xp} ({progress}/{needed} into the next level) · Messages counted: {msgs}",
        )
        return True

    rows = db.execute("SELECT name, level, xp FROM user_xp ORDER BY xp DESC LIMIT 10").fetchall()
    if not rows:
        await safe_send(message.channel, "The leaderboard is empty — someone say something!")
        return True
    medals = ["🥇", "🥈", "🥉"]
    lines = ["**🏆 Leaderboard**"]
    for i, (name, level, xp) in enumerate(rows):
        tag = medals[i] if i < 3 else f"`#{i + 1}`"
        lines.append(f"{tag} **{name}** — Level {level} · {xp:,} XP")
    await safe_send(message.channel, "\n".join(lines))
    return True


# =============================================================================
# Spam & Flood Detection
# =============================================================================
_INVITE_RE = re.compile(
    r"(discord\.gg|discord\.com/invite|discordapp\.com/invite)/\S+",
    re.IGNORECASE,
)
_SCAM_URL_RE = re.compile(
    r"(free\s*nitro|steam\s*community\.ru|discord.*gift|claim.*reward|verify.*airdrop)",
    re.IGNORECASE,
)

# Track recent messages per user for flood detection: deque of (channel_id, content_hash, timestamp)
_recent_messages: dict[int, deque[tuple[int, str, float]]] = {}
_FLOOD_WINDOW = 30.0
_FLOOD_CHANNEL_THRESHOLD = 3

# Recent flags for !flags command: deque of (timestamp, title, details)
_flag_history: deque[tuple[float, str, str]] = deque(maxlen=100)


async def _is_privileged(user_id: int) -> bool:
    """Check if a user is the owner or has the moderator role in any guild."""
    if user_id == OWNER_ID:
        return True
    for guild in bot.guilds:
        try:
            member = guild.get_member(user_id) or await guild.fetch_member(user_id)
            if member and any(r.id == MOD_ROLE_ID for r in member.roles):
                return True
        except Exception:
            continue
    return False


async def _flag_to_mods(title: str, details: str):
    _flag_history.append((time.time(), title, details))
    if not MOD_CHANNEL_ID:
        return
    try:
        mod_ch = bot.get_channel(MOD_CHANNEL_ID) or await bot.fetch_channel(MOD_CHANNEL_ID)
        if mod_ch:
            await safe_send(mod_ch, f"@here ⚠️ **{title}**\n{details}")
    except Exception:
        logging.exception("Failed to send mod alert")


async def check_spam(message: discord.Message) -> bool:
    """Flag invite/scam links, especially from new accounts. Returns True if flagged."""
    if message.guild is None:
        return False

    text = message.content or ""
    has_invite = bool(_INVITE_RE.search(text))
    has_scam = bool(_SCAM_URL_RE.search(text))

    if not has_invite and not has_scam:
        return False

    member = message.author if isinstance(message.author, discord.Member) else None
    if member is None:
        return False

    account_age = datetime.now(timezone.utc) - member.created_at
    is_new = account_age < timedelta(hours=24)

    if is_new or has_scam:
        reason = []
        if is_new:
            reason.append(f"new account ({account_age.total_seconds() / 3600:.1f}h old)")
        if has_invite:
            reason.append("Discord invite link")
        if has_scam:
            reason.append("scam URL pattern")

        await _flag_to_mods(
            "Possible spam detected",
            f"User: **{member.display_name}** ({member.id})\n"
            f"Channel: #{message.channel.name}\n"
            f"Reason: {', '.join(reason)}\n"
            f"Message: {text[:200]}"
        )
        add_user_record(member.id, "spam_flag", f"{', '.join(reason)} | {text[:150]}")
        return True
    return False


async def check_flood(message: discord.Message) -> bool:
    """Detect same user posting identical messages across multiple channels."""
    if message.guild is None:
        return False

    uid = message.author.id
    content = (message.content or "").strip().lower()
    if len(content) < 10:
        return False

    now = time.time()

    if uid not in _recent_messages:
        _recent_messages[uid] = deque(maxlen=50)

    history = _recent_messages[uid]
    history.append((message.channel.id, content, now))

    channels_with_same = set()
    for ch_id, msg_content, ts in history:
        if now - ts <= _FLOOD_WINDOW and msg_content == content:
            channels_with_same.add(ch_id)

    if len(channels_with_same) >= _FLOOD_CHANNEL_THRESHOLD:
        member = message.author
        await _flag_to_mods(
            "Flood detected",
            f"User: **{member.display_name}** ({member.id})\n"
            f"Same message posted in {len(channels_with_same)} channels within {_FLOOD_WINDOW}s\n"
            f"Content: {content[:200]}"
        )
        add_user_record(member.id, "flood_flag", f"same msg in {len(channels_with_same)} channels | {content[:150]}")
        history.clear()
        return True
    return False


# =============================================================================
# Honeypot
# =============================================================================
# Users the bot itself just kicked/banned, so the mod log can skip the
# ban/unban/delete events those actions generate — one honeypot event
# should produce exactly one log entry.
_bot_actioned: dict[int, float] = {}


def _recently_actioned(user_id: int | None = None, window: float = 300.0) -> bool:
    now = time.time()
    for uid, ts in list(_bot_actioned.items()):
        if now - ts > window:
            del _bot_actioned[uid]
    if user_id is None:
        return bool(_bot_actioned)
    return user_id in _bot_actioned


# Pre-written removal announcements — no LLM call needed for a one-liner.
# Override with a "HoneypotQuips" list in the config ({name} = offender).
_DEFAULT_QUIPS = [
    "{name} found the one channel nobody should post in. Impressive, briefly.",
    "The trap was clearly a trap. {name} posted anyway. Farewell.",
    "{name} wandered into the honeypot and got escorted out.",
    "Another one for the honeypot. Goodbye, {name}.",
]


def pick_mod_quip(offender_name: str) -> str:
    quips = config.get("HoneypotQuips") or _DEFAULT_QUIPS
    try:
        return str(random.choice(quips)).format(name=offender_name)
    except Exception:
        return f"{offender_name} tripped the honeypot — clean removal completed."


async def honeypot_guard(message: discord.Message) -> bool:
    try:
        if message.guild is None or message.author.bot:
            return False
        if not TRAP_CHANNEL_ID or message.channel.id != TRAP_CHANNEL_ID:
            return False

        guild = message.guild
        member = message.author if isinstance(message.author, discord.Member) else None
        if member is None:
            try:
                member = await guild.fetch_member(message.author.id)
            except Exception:
                logging.exception("Failed to fetch member for honeypot")
                return False

        if member == guild.owner or member.guild_permissions.administrator:
            return False
        if any(r.id in EXEMPT_ROLE_IDS for r in member.roles):
            return False

        try:
            await message.delete()
        except Exception:
            logging.exception("Could not delete honeypot message")

        reason = f"Posted in honeypot channel ({TRAP_CHANNEL_ID})"
        action = str(config.get("HoneypotAction", "kick")).lower()
        delete_seconds = max(0, min(604800, int(config.get("HoneypotDeleteSeconds", 600))))
        removal_word = "banned" if action == "ban" else "kicked"
        removed_ok = False
        total_deleted = -1  # -1 = Discord wiped messages server-side

        # Ban with delete_message_seconds so Discord deletes the user's recent
        # messages server-side — this also catches messages the history endpoint
        # hasn't surfaced yet and ones posted mid-cleanup, which a manual purge
        # misses. With HoneypotAction "kick" (default) the ban is lifted right
        # away ("softban"), so the user can rejoin like after a normal kick.
        _bot_actioned[member.id] = time.time()
        try:
            await guild.ban(member, reason=reason, delete_message_seconds=delete_seconds)
            if action != "ban":
                await guild.unban(member, reason="Honeypot softban — kick semantics")
            removed_ok = True
            logging.info("Honeypot: %s %s (server-side wipe of last %ds).", removal_word, member, delete_seconds)
        except discord.Forbidden:
            logging.warning("Honeypot: no ban permission; falling back to kick + manual purge")
        except Exception:
            logging.exception("Honeypot: ban failed; falling back to kick + manual purge")

        if not removed_ok:
            # Fallback without ban permission: kick FIRST so no new messages
            # arrive, then sweep history twice — the second pass catches
            # messages the history endpoint returned late.
            try:
                await guild.kick(member, reason=reason)
                removed_ok = True
                removal_word = "kicked"
            except Exception:
                logging.exception("Honeypot: failed to kick")

            cutoff = datetime.now(timezone.utc) - timedelta(seconds=delete_seconds)
            total_deleted = 0

            async def purge_channel(ch: discord.TextChannel) -> int:
                try:
                    me = guild.me or await guild.fetch_member(bot.user.id)
                    perms = ch.permissions_for(me)
                    if not (perms.read_message_history and perms.manage_messages):
                        return 0
                    deleted = await ch.purge(
                        limit=None, after=cutoff,
                        check=lambda m: m.author.id == member.id,
                        bulk=True,
                    )
                    return len(deleted)
                except Exception:
                    return 0

            for sweep in range(2):
                if sweep:
                    await asyncio.sleep(3)
                for ch in guild.text_channels:
                    total_deleted += await purge_channel(ch)
            logging.info("Honeypot: %s %s; purged ~%d msgs.", removal_word if removed_ok else "FAILED to remove", member, total_deleted)

        add_user_record(
            member.id,
            "honeypot_ban" if action == "ban" else "honeypot_kick",
            f"posted in honeypot channel; removed_ok={removed_ok}",
        )

        try:
            notify_id = (
                int(config.get("HoneypotNotifyChannelID", 0))
                or int(config.get("ModLogChannelID", 0))
                or MOD_CHANNEL_ID
            )
            notify_ch = None
            if notify_id:
                notify_ch = bot.get_channel(notify_id) or await bot.fetch_channel(notify_id)
            if notify_ch:
                quip = pick_mod_quip(member.display_name)
                status = removal_word if removed_ok else "NOT removed (action failed)"
                if total_deleted < 0:
                    cleanup_line = f"🧹 Discord wiped their messages from the last {delete_seconds // 60} min."
                else:
                    cleanup_line = f"🧹 Deleted ~{total_deleted} message(s) from the last {delete_seconds // 60} min."
                embed = discord.Embed(
                    title=f"👢 Honeypot: {member.display_name} {status}",
                    description=f"{cleanup_line}\n{quip}",
                    color=0x992D22 if removed_ok else 0xE67E22,
                )
                embed.add_field(name="User", value=f"{member} ({member.id})")
                await notify_ch.send(embed=embed)
        except Exception:
            logging.exception("Failed to notify mods")

        return True
    except Exception:
        logging.exception("Honeypot guard failed")
        return False


# =============================================================================
# Mod Log
# =============================================================================
def _modlog_channel_id() -> int:
    return int(config.get("ModLogChannelID", 0))


async def _modlog_send(embed: discord.Embed):
    ch_id = _modlog_channel_id()
    if not ch_id:
        return
    try:
        ch = bot.get_channel(ch_id) or await bot.fetch_channel(ch_id)
        await ch.send(embed=embed)
    except Exception:
        logging.exception("Mod log send failed")


def _trunc(text: str, n: int = 900) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


@bot.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent):
    try:
        if payload.guild_id is None or payload.channel_id == _modlog_channel_id():
            return
        msg = payload.cached_message
        if msg is not None and msg.author.bot:
            return
        if payload.channel_id == TRAP_CHANNEL_ID:
            return  # honeypot trigger message — covered by the honeypot entry
        if msg is not None and _recently_actioned(msg.author.id):
            return  # wiped by our own ban — covered by the honeypot entry
        embed = discord.Embed(title="Message deleted", color=0xE74C3C)
        if msg is not None:
            embed.add_field(name="Author", value=f"{msg.author} ({msg.author.id})", inline=False)
            if msg.content:
                embed.add_field(name="Content", value=_trunc(msg.content), inline=False)
            if msg.attachments:
                embed.add_field(name="Attachments", value=_trunc(", ".join(a.filename for a in msg.attachments), 200), inline=False)
        else:
            embed.description = "Content unknown (message was not cached)."
        embed.add_field(name="Channel", value=f"<#{payload.channel_id}>")
        embed.add_field(name="When", value=f"<t:{int(time.time())}:R>")
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_raw_message_delete failed")


@bot.event
async def on_raw_bulk_message_delete(payload: discord.RawBulkMessageDeleteEvent):
    try:
        if payload.guild_id is None or payload.channel_id == _modlog_channel_id():
            return
        if _recently_actioned():
            return  # purge from our own honeypot action
        embed = discord.Embed(
            title="Bulk delete",
            description=f"{len(payload.message_ids)} messages removed in <#{payload.channel_id}> (e.g. a purge).",
            color=0xE74C3C,
        )
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_raw_bulk_message_delete failed")


@bot.event
async def on_raw_message_edit(payload: discord.RawMessageUpdateEvent):
    try:
        if payload.guild_id is None or payload.channel_id == _modlog_channel_id():
            return
        data = payload.data or {}
        if "content" not in data:
            return  # embed/pin/component update, not a text edit
        new_content = data.get("content") or ""
        cached = payload.cached_message
        if cached is not None:
            if cached.author.bot:
                return
            if (cached.content or "") == new_content:
                return  # link unfurl etc.
            old_content = cached.content or ""
            author_desc = f"{cached.author} ({cached.author.id})"
        else:
            author = data.get("author") or {}
            if author.get("bot"):
                return
            old_content = "*unknown (not cached)*"
            author_desc = f"<@{author.get('id', '?')}> ({author.get('id', '?')})"
        embed = discord.Embed(title="Message edited", color=0xE67E22)
        embed.add_field(name="Author", value=author_desc, inline=False)
        embed.add_field(name="Before", value=_trunc(old_content), inline=False)
        embed.add_field(name="After", value=_trunc(new_content), inline=False)
        jump = f"https://discord.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}"
        embed.add_field(name="Where", value=f"<#{payload.channel_id}> · [jump]({jump})")
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_raw_message_edit failed")


@bot.event
async def on_member_ban(guild: discord.Guild, user):
    try:
        if _recently_actioned(user.id):
            return  # our own honeypot action — already logged
        embed = discord.Embed(title="Member banned", color=0x992D22)
        embed.add_field(name="User", value=f"{user} ({user.id})")
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_member_ban failed")


@bot.event
async def on_member_unban(guild: discord.Guild, user):
    try:
        if _recently_actioned(user.id):
            return  # softban lift — not a real unban
        embed = discord.Embed(title="Member unbanned", color=0x2ECC71)
        embed.add_field(name="User", value=f"{user} ({user.id})")
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_member_unban failed")


# =============================================================================
# Handlers
# =============================================================================
async def handle_text_message(message: discord.Message, text_override: str | None = None):
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)
        is_dm = isinstance(message.channel, discord.DMChannel)

        incoming = text_override if text_override is not None else (message.content or "")
        recent_ctx = " ".join(t for _, t in list(cm.get(ch_id).turns)[-6:])[-1500:]
        blocked = chat_message_blocked(incoming, recent_ctx)
        if blocked:
            # Never reaches the model and never enters conversation memory, so it
            # cannot steer later replies.
            await refuse_chat(message.channel, message.author.id,
                              message.author.display_name, blocked, incoming, "input")
            return

        cm.add_user(ch_id, is_dm, message.author.id, message.author.display_name, message.content, message.id)
        await cm.maybe_compress(ch_id)

        # Build the retrieval query from the recent exchange, not just this line —
        # otherwise "tell me more about him" retrieves nothing.
        this_text = text_override if text_override is not None else (message.content or "")
        recent = [t for r, t in list(cm.get(ch_id).turns)[-4:] if r == "user"]
        retrieval_query = " ".join(recent[-2:] + [this_text])[-1200:]
        system_prefix = build_system_prefix(retrieval_query)
        msgs = cm.build_messages(ch_id, system_prefix=system_prefix)

        if text_override is not None and msgs and msgs[-1]["role"] == "user":
            if not is_dm:
                msgs[-1] = {"role": "user", "content": f"[{message.author.display_name}]: {text_override}"}
            else:
                msgs[-1] = {"role": "user", "content": text_override}

        logging.info("TEXT -> LLM | ch=%s | msg_id=%s", ch_id, message.id)
        freq_pen = float(config.get("FrequencyPenalty", 0.3))
        pres_pen = float(config.get("PresencePenalty", 0.3))
        tools = image_tools_for(message)
        async with message.channel.typing():
            result = await chat_async(
                msgs, temperature=0.6, max_tokens=600,
                frequency_penalty=freq_pen, presence_penalty=pres_pen,
                **({"tools": tools, "return_message": True} if tools else {}),
            )

        reply = result
        if tools:
            calls = getattr(result, "tool_calls", None) or []
            for call in calls:
                if getattr(call.function, "name", "") != "generate_image":
                    continue
                try:
                    args = json.loads(call.function.arguments or "{}")
                except Exception:
                    logging.exception("Image tool: bad arguments %r", call.function.arguments)
                    break
                said = (getattr(result, "content", "") or "").strip()
                if said:
                    cm.add_assistant(ch_id, said)
                    await safe_send(message.channel, said)
                if await run_tool_image(message, args):
                    return
            reply = getattr(result, "content", None)

        if not (reply or "").strip():
            # Model returned nothing (e.g. provider refusal). Don't store or
            # send an empty turn — it would 400 on Discord and pollute memory.
            logging.warning("Empty LLM reply in ch %s; sending fallback", ch_id)
            await safe_send(message.channel, config.get("EmptyReplyFallback", "…I have nothing to say to that."))
            return

        # Loop breaker: a reply that near-duplicates a recent one gets one
        # retry with an explicit nudge. A still-duplicated reply is sent but
        # NOT stored, so the repetition cannot reinforce itself in memory.
        recent = [t for r, t in list(cm.get(ch_id).turns)[-8:] if r == "assistant"]
        if any(_too_similar(reply, prev) for prev in recent):
            logging.warning("Repetition detected in ch %s; retrying with nudge", ch_id)
            retry_msgs = msgs + [
                {"role": "assistant", "content": reply},
                {
                    "role": "system",
                    "content": (
                        "Your last reply repeats an earlier one almost verbatim. Write a completely "
                        "different reply: new sentence structure, new imagery, no reused phrases."
                    ),
                },
            ]
            fresh = await chat_async(
                retry_msgs, temperature=0.9, max_tokens=600,
                frequency_penalty=max(freq_pen, 0.5), presence_penalty=max(pres_pen, 0.5),
            )
            if (fresh or "").strip() and not any(_too_similar(fresh, prev) for prev in recent):
                reply = fresh
            else:
                logging.warning("Repetition persists in ch %s; reply withheld from memory", ch_id)
                await safe_send(message.channel, (fresh or "").strip() or reply)
                return

        # Include the user's current turn: they may have set the scene in the very
        # message that prompted this reply.
        out_blocked = chat_message_blocked(reply, f"{recent_ctx} {incoming}")
        if out_blocked:
            # Drop the exchange entirely: storing it would let the reply seed
            # later turns through the conversation window and summaries.
            cv = cm.get(ch_id)
            if cv.turns:
                cv.turns.pop()
            cm.mark_dirty()
            await refuse_chat(message.channel, message.author.id,
                              message.author.display_name, out_blocked, reply, "model output")
            return

        cm.add_assistant(ch_id, reply)
        await safe_send(message.channel, reply)
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Error in handle_text_message")
        await safe_send(message.channel, "Oops — something went wrong with that one.")


async def handle_image_message(message: discord.Message, text_override: str | None = None):
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)

        if not get_user_bucket(message.author.id).consume():
            await safe_send(message.channel, "You're requesting images too fast — slow down a bit.")
            return

        text_in = text_override if text_override is not None else (message.content or "")
        if not images_enabled():
            await image_unavailable(message.channel)
            return
        blocked = image_prompt_blocked(text_in)
        if blocked:
            await refuse_image_request(message.channel, message.author.id,
                                       message.author.display_name, blocked, text_in)
            return
        exact_mode = is_exact_trigger(text_in)

        raw = text_in
        for phrase in EXACT_TRIGGERS:
            raw = re.sub(re.escape(phrase), "", raw, count=1, flags=re.IGNORECASE)
        # Strip only a LEADING trigger so words inside the prompt survive
        # ("art nouveau", "a dragon drawing a sword").
        raw = re.sub(r"^\s*(img|image|art)\s*:\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"^\s*(please\s+)?(draw|paint|sketch|illustrate|render)\b\s*(me\s+)?", "", raw, flags=re.IGNORECASE)
        if exact_mode:
            raw = re.sub(r"^\s*exact\b:?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s+", " ", raw).strip(" -:;,. \n\t")

        # --- lightweight parameters parsed from the request ---
        width = height = None
        if re.search(r"\b(portrait|tall)\b", raw, re.IGNORECASE):
            size = config.get("SDPortraitSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])
        elif re.search(r"\b(landscape|wide)\b", raw, re.IGNORECASE):
            size = config.get("SDLandscapeSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])

        batch = 1
        m = re.search(r"\b([2-9])x\b|\bx([2-9])\b", raw)
        if m:
            batch = int(m.group(1) or m.group(2))
            raw = (raw[:m.start()] + raw[m.end():]).strip()

        positive_prefix = None
        preset_neg = None
        presets = config.get("SDStylePresets") or {}
        m = re.search(r"\bstyle:\s*(\w+)\b", raw, re.IGNORECASE)
        if m:
            preset = next((v for k, v in presets.items() if k.lower() == m.group(1).lower()), None)
            if preset:
                positive_prefix = preset.get("positive", "")
                preset_neg = preset.get("negative")
                raw = (raw[:m.start()] + raw[m.end():]).strip()

        neg = preset_neg or config.get("SDNegativePrompt", "(lowres, blurry, deformed)")

        ref_rec = ipm.find_by_message(message.reference.message_id if message.reference else None)
        base = ref_rec or ipm.last_for_channel(ch_id)
        t_norm = re.sub(r"[\s!.…]+$", "", (text_in or "").strip().lower())
        is_reroll = t_norm in {"again", "same", "same again", "again please", "reroll", "another", "another one", "one more"}

        seed = -1
        init_b64 = None
        if base and is_reroll:
            # Bare "again": same prompt, fresh random seed — a new take.
            status_msg = await safe_send(message.channel, "Rolling a fresh take on that…")
            sd_prompt, neg = base.final_sd_prompt, base.negative_prompt
            positive_prefix = base.positive_prefix or None
            width, height = base.width or None, base.height or None
        elif base and (looks_like_followup(text_in) or ref_rec):
            # A change request: refine the prompt, keep the seed, and — when we
            # still hold the source image — run img2img so the composition stays put.
            status_msg = await safe_send(message.channel, config.get("ImageRefinementNotice", "Refining the previous image…"))
            refined = await refine_image_prompt(base, text_in)
            sd_prompt = refined.get("prompt", base.final_sd_prompt)
            neg = refined.get("negative", base.negative_prompt)
            seed = base.seed
            positive_prefix = base.positive_prefix or None
            width, height = base.width or None, base.height or None
            if base is ipm.last_for_channel(ch_id):
                init_b64 = _last_image_b64.get(ch_id)
        else:
            status_msg = await safe_send(message.channel, "Hang on while I sketch that for you…")
            sd_prompt = raw if (exact_mode or _looks_like_tag_prompt(raw)) else await compile_sd_prompt(raw)

        await run_image_job(
            message.channel,
            ch_id=ch_id,
            user_prompt=text_in,
            sd_prompt=sd_prompt,
            neg=neg,
            seed=seed,
            batch=batch,
            width=width,
            height=height,
            positive_prefix=positive_prefix,
            init_image_b64=init_b64,
            requested_by=message.author.display_name,
            requester_id=message.author.id,
            trigger_message_id=message.id,
            status_msg=status_msg,
        )
    except asyncio.CancelledError:
        return
    except Exception:
        logging.exception("Error in handle_image_message")
        await safe_send(message.channel, "Oops — image generation hit a snag.")


# =============================================================================
# Bot Commands (Owner + Moderators) — DM or mod channel
# =============================================================================
# Command access levels
_OWNER_COMMANDS = {"!reload"}
_MOD_COMMANDS = {"!summary", "!activity", "!whois", "!flags", "!search", "!clearhistory", "!note", "!record", "!help"}


def _is_command_channel(message: discord.Message) -> bool:
    """Commands are accepted in DMs or the mod channel."""
    if isinstance(message.channel, discord.DMChannel):
        return True
    return getattr(message.channel, "id", None) == MOD_CHANNEL_ID


async def handle_bot_command(message: discord.Message) -> bool:
    """Process commands from owner or moderators. Returns True if handled."""
    if not _is_command_channel(message):
        return False

    text = (message.content or "").strip()
    if not text.startswith("!"):
        return False

    parts = text.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    is_owner = message.author.id == OWNER_ID

    # Check if command exists at all
    if cmd not in _OWNER_COMMANDS and cmd not in _MOD_COMMANDS:
        return False

    # Owner-only commands
    if cmd in _OWNER_COMMANDS and not is_owner:
        await message.channel.send("That command is owner-only.")
        return True

    # Mod commands require owner OR moderator role
    if cmd in _MOD_COMMANDS and not is_owner:
        if not await _is_privileged(message.author.id):
            return False  # silently ignore — they're not authorized

    ch = message.channel

    # --- !summary ---
    if cmd == "!summary":
        hours = 24
        if arg:
            match = re.match(r"(\d+)\s*h?", arg)
            if match:
                hours = int(match.group(1))
        await ch.send(f"⏳ Gathering messages from the last {hours}h — this may take a minute…")
        digest = await generate_server_summary(hours=hours)
        if digest:
            chunks = [digest[i:i + 1900] for i in range(0, len(digest), 1900)]
            for chunk in chunks:
                await ch.send(chunk)
        else:
            await ch.send("No significant activity found.")
        return True

    # --- !reload (owner only) ---
    elif cmd == "!reload":
        try:
            global config
            config = load_config()
            _reload_derived_config()
            await ch.send("✅ Config reloaded successfully.")
            logging.info("Config manually reloaded by %s", message.author)
        except Exception as e:
            await ch.send(f"❌ Reload failed: {e}")
        return True

    # --- !activity ---
    elif cmd == "!activity":
        await ch.send("⏳ Scanning channels…")
        after = datetime.now(timezone.utc) - timedelta(hours=6)
        counts: list[tuple[str, int]] = []
        for guild in bot.guilds:
            for tc in guild.text_channels:
                try:
                    me = guild.me
                    if me and not tc.permissions_for(me).read_message_history:
                        continue
                    n = 0
                    async for _ in tc.history(after=after, limit=200):
                        n += 1
                    if n > 0:
                        counts.append((f"#{tc.name}", n))
                except Exception:
                    continue
        counts.sort(key=lambda x: x[1], reverse=True)
        top = counts[:10]
        if top:
            lines = [f"{name}: {count} msgs" for name, count in top]
            await ch.send("**Activity (last 6h):**\n" + "\n".join(lines))
        else:
            await ch.send("No activity found in the last 6 hours.")
        return True

    # --- !whois <username or user_id> ---
    elif cmd == "!whois":
        if not arg:
            await ch.send("Usage: `!whois <username or user_id>`")
            return True
        await ch.send("⏳ Looking up user…")

        found_member: discord.Member | None = None
        for guild in bot.guilds:
            # Try by ID first
            try:
                uid = int(arg)
                found_member = guild.get_member(uid) or await guild.fetch_member(uid)
                if found_member:
                    break
            except (ValueError, discord.NotFound):
                pass
            # Try by name/display name
            if not found_member:
                query = arg.lower()
                for m in guild.members:
                    if query in m.display_name.lower() or query in m.name.lower():
                        found_member = m
                        break
            if found_member:
                break

        if not found_member:
            await ch.send(f"Could not find user matching `{arg}`.")
            return True

        m = found_member
        account_age = datetime.now(timezone.utc) - m.created_at
        join_age = datetime.now(timezone.utc) - m.joined_at if m.joined_at else None
        roles = ", ".join(r.name for r in m.roles if r.name != "@everyone") or "None"

        # Count recent messages (last 24h, across readable channels)
        msg_count = 0
        after_24h = datetime.now(timezone.utc) - timedelta(hours=24)
        for tc in m.guild.text_channels:
            try:
                me = m.guild.me
                if me and not tc.permissions_for(me).read_message_history:
                    continue
                async for msg in tc.history(after=after_24h, limit=100):
                    if msg.author.id == m.id:
                        msg_count += 1
            except Exception:
                continue

        info = (
            f"**{m.display_name}** ({m.name}, ID: {m.id})\n"
            f"Account created: {m.created_at.strftime('%Y-%m-%d')} ({account_age.days}d ago)\n"
        )
        if join_age:
            info += f"Joined server: {m.joined_at.strftime('%Y-%m-%d')} ({join_age.days}d ago)\n"
        info += (
            f"Roles: {roles}\n"
            f"Messages (last 24h): ~{msg_count}"
        )
        rec_summary = summarize_user_record_counts(m.id)
        if rec_summary:
            info += f"\nRecord: {rec_summary} — `!record {m.id}` for details"
        await ch.send(info)
        return True

    # --- !flags ---
    elif cmd == "!flags":
        now = time.time()
        cutoff = now - 86400  # last 24h
        recent = [(ts, title, details) for ts, title, details in _flag_history if ts >= cutoff]

        if not recent:
            await ch.send("No flags in the last 24 hours.")
            return True

        lines: list[str] = []
        for ts, title, details in reversed(recent):  # newest first
            dt = datetime.fromtimestamp(ts, tz=SUMMARY_TZ)
            time_str = dt.strftime("%H:%M")
            # Compact the details to one line
            short = details.replace("\n", " | ")
            if len(short) > 200:
                short = short[:200] + "…"
            lines.append(f"`{time_str}` **{title}** — {short}")

        header = f"**Flags (last 24h): {len(recent)} total**\n"
        body = "\n".join(lines)
        full = header + body
        chunks = [full[i:i + 1900] for i in range(0, len(full), 1900)]
        for chunk in chunks:
            await ch.send(chunk)
        return True

    # --- !search <term> ---
    elif cmd == "!search":
        if not arg:
            await ch.send("Usage: `!search <term>`")
            return True
        if len(arg) < 3:
            await ch.send("Search term must be at least 3 characters.")
            return True

        await ch.send(f"⏳ Searching for `{arg}` across channels…")
        query = arg.lower()
        results: list[tuple[str, str, str, datetime]] = []  # (channel, author, content, timestamp)
        after = datetime.now(timezone.utc) - timedelta(hours=24)

        for guild in bot.guilds:
            for tc in guild.text_channels:
                try:
                    me = guild.me
                    if me and not tc.permissions_for(me).read_message_history:
                        continue
                    async for msg in tc.history(after=after, limit=200):
                        if msg.author.bot:
                            continue
                        content = msg.content or ""
                        if query in content.lower():
                            snippet = content[:150] + "…" if len(content) > 150 else content
                            results.append((f"#{tc.name}", msg.author.display_name, snippet, msg.created_at))
                except Exception:
                    continue

        if not results:
            await ch.send(f"No results for `{arg}` in the last 24h.")
            return True

        results.sort(key=lambda x: x[3], reverse=True)  # newest first
        results = results[:20]  # cap at 20

        lines = []
        for ch_name, author, snippet, ts in results:
            time_str = ts.astimezone(SUMMARY_TZ).strftime("%H:%M")
            lines.append(f"`{time_str}` {ch_name} — **{author}**: {snippet}")

        header = f"**Search results for `{arg}` (last 24h): {len(results)} found**\n"
        body = "\n".join(lines)
        full = header + body
        chunks = [full[i:i + 1900] for i in range(0, len(full), 1900)]
        for chunk in chunks:
            await ch.send(chunk)
        return True

    # --- !clearhistory [channel_id] ---
    elif cmd == "!clearhistory":
        # In the mod channel with no arg, clears that channel. In DM, requires an explicit ID.
        target_id: int | None = None
        if arg:
            try:
                target_id = int(arg.strip().lstrip("#<").rstrip(">"))
            except ValueError:
                await ch.send("Usage: `!clearhistory <channel_id>` (or run in a channel with no argument).")
                return True
        elif not isinstance(message.channel, discord.DMChannel):
            target_id = channel_key(message)
        else:
            await ch.send("In a DM you must specify a channel ID: `!clearhistory <channel_id>`.")
            return True

        cleared = cm.clear_channel(target_id)
        if cleared:
            await ch.send(f"✅ Cleared the bot's conversation history for channel `{target_id}`.")
            logging.info("History cleared for channel %s by %s", target_id, message.author)
        else:
            await ch.send(f"No stored history found for channel `{target_id}`.")
        return True

    # --- !note <user> <text> ---
    elif cmd == "!note":
        note_parts = arg.split(None, 1)
        target = _parse_user_ref(note_parts[0]) if note_parts else None
        if not target or len(note_parts) < 2:
            await ch.send("Usage: `!note <user_id or @mention> <text>`")
            return True
        add_user_record(target, "note", note_parts[1], moderator_id=message.author.id)
        await ch.send(f"📝 Note added for <@{target}> (`{target}`).")
        logging.info("Note added for %s by %s", target, message.author)
        return True

    # --- !record <user> ---
    elif cmd == "!record":
        target = _parse_user_ref(arg)
        if not target:
            await ch.send("Usage: `!record <user_id or @mention>`")
            return True
        rows = get_user_records(target, limit=25)
        if not rows:
            await ch.send(f"No records for <@{target}> (`{target}`).")
            return True
        summary = summarize_user_record_counts(target)
        lines = [f"**Record for <@{target}>** (`{target}`) — {summary}"]
        for kind, detail, moderator_id, ts in rows:
            when = f"<t:{int(ts)}:d>"
            entry = f"{when} · **{kind.replace('_', ' ')}**"
            if detail:
                entry += f" — {detail[:150]}"
            if moderator_id:
                entry += f" (by <@{moderator_id}>)"
            lines.append(entry)
        full = "\n".join(lines)
        for chunk in [full[i:i + 1900] for i in range(0, len(full), 1900)]:
            await ch.send(chunk, allowed_mentions=discord.AllowedMentions.none())
        return True

    # --- !help ---
    elif cmd == "!help":
        help_text = (
            "**Commands** (DM or mod channel):\n"
            "`!summary [Nh]` — server digest (default 24h)\n"
            "`!activity` — most active channels (last 6h)\n"
            "`!whois <user>` — look up a member\n"
            "`!flags` — recent spam/flood alerts (last 24h)\n"
            "`!search <term>` — search messages (last 24h)\n"
            "`!clearhistory [channel_id]` — wipe the bot's memory for a channel\n"
            "`!note <user> <text>` — add a note to a user's record\n"
            "`!record <user>` — show a user's record (flags, honeypot trips, notes)\n"
            "`!help` — this message"
        )
        if is_owner:
            help_text += "\n\n**Owner only:**\n`!reload` — reload config from disk"
        await ch.send(help_text)
        return True

    return False


# =============================================================================
# Router
# =============================================================================
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user or message.author.bot:
        return

    # Honeypot — runs before everything
    try:
        if await honeypot_guard(message):
            return
    except Exception:
        logging.exception("honeypot_guard error")

    # Spam & flood detection (guild only, runs before ignore/allowlist)
    if message.guild:
        try:
            await check_spam(message)
            await check_flood(message)
        except Exception:
            logging.exception("Spam/flood check error")

    # Ignored users/words
    if is_ignored(message):
        return

    # Bot commands (owner + moderators, in DMs or mod channel)
    if _is_command_channel(message):
        try:
            if await handle_bot_command(message):
                return
        except Exception:
            logging.exception("Bot command error")

    # XP commands — public, anyone, any channel (and DMs)
    try:
        if await handle_xp_command(message):
            return
    except Exception:
        logging.exception("XP command error")

    # Passive XP for guild chatter (commands excluded)
    if message.guild and not (message.content or "").startswith("!"):
        asyncio.create_task(award_xp(message))

    try:
        ch_id = channel_key(message)
        logging.info(
            "ROUTER in=%r dm=%s ch_id=%s author=%s",
            message.content, isinstance(message.channel, discord.DMChannel), ch_id, message.author,
        )

        if not is_allowed(message):
            return

        raw_text = message.content or ""
        text_for_logic = raw_text

        if config.get("OnlyWhenCalled") and not isinstance(message.channel, discord.DMChannel):
            bot_name = config.get("Name", "")
            mentioned = (bot_name.lower() in raw_text.lower()) or (bot.user in message.mentions)
            if not mentioned:
                return
            text_for_logic = re.sub(re.escape(bot_name), "", raw_text, flags=re.IGNORECASE).strip()

        if should_route_to_image_followup(message):
            asyncio.create_task(handle_image_message(message, text_override=text_for_logic))
            return

        if looks_like_image_request(text_for_logic):
            asyncio.create_task(handle_image_message(message, text_override=text_for_logic))
            return

        asyncio.create_task(handle_text_message(message, text_override=text_for_logic))
    except Exception:
        logging.exception("on_message router failure")


# =============================================================================
# FAQ Auto-Answer (questions forum)
# =============================================================================
# Community-first: a question thread only gets a bot answer after it has sat
# without any human reply for FAQAnswerDelayMin minutes. Answers are grounded
# strictly in the FAQ file — no coverage, no answer.
_FAQ_CACHE: dict[str, Any] = {"path": None, "mtime": 0.0, "text": "", "entries": []}


def _load_faq() -> tuple[str, list[tuple[str, dict[str, float]]]]:
    """FAQ text plus per-entry term weights, reparsed only when the file changes."""
    path = config.get("FAQPath", "game_faq.txt")
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return "", []
    if _FAQ_CACHE["path"] == path and _FAQ_CACHE["mtime"] == mtime:
        return _FAQ_CACHE["text"], _FAQ_CACHE["entries"]
    text = _read_text(path)
    parsed, freq = [], {}
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block.lower().startswith("q:"):
            continue
        words = set(re.findall(r"[a-z]{3,}", block.lower()))
        parsed.append((block, words))
        for w in words:
            freq[w] = freq.get(w, 0) + 1
    entries = [(b, {w: 1.0 / freq[w] for w in words}) for b, words in parsed]
    _FAQ_CACHE.update(path=path, mtime=mtime, text=text, entries=entries)
    logging.info("FAQ loaded: %d entries, ~%d tokens", len(entries), len(text) / 3.6)
    return text, entries


def retrieve_faq(question: str) -> str:
    """The slice of the FAQ worth showing for this question.

    While the whole file still fits comfortably it is sent intact — retrieval can
    only lose recall, and there is nothing to gain. Once the FAQ grows past
    FAQRetrievalMinTokens, only the best-matching entries are sent.
    """
    text, entries = _load_faq()
    if not text:
        return ""
    if len(text) / 3.6 <= float(config.get("FAQRetrievalMinTokens", 8000)) or not entries:
        return text
    qwords = _query_words(question.lower())
    scored = [(sum(w for t, w in terms.items() if t in qwords), b) for b, terms in entries]
    scored = sorted((s for s in scored if s[0] > 0), key=lambda x: -x[0])
    if not scored:
        return text
    budget = float(config.get("FAQRetrievalMaxTokens", 2500))
    picked = []
    for _, block in scored:
        cost = len(block) / 3.6
        if cost > budget:
            break
        picked.append(block)
        budget -= cost
    logging.info("FAQ retrieval: %d/%d entries", len(picked), len(entries))
    return "\n\n".join(picked) if picked else text


async def faq_answer(question: str) -> str | None:
    """Answer a question strictly from the FAQ file; None when it isn't covered."""
    faq = retrieve_faq(question)
    if not faq:
        return None

    name = config.get("Name", "the bot")
    system = (
        f"You are {name}, answering a player's question about the game. "
        "Answer ONLY with information from the FAQ below — never invent, never use outside knowledge. "
        "Be concrete and concise (under 150 words). A touch of in-character flavor is fine, "
        "but clarity beats persona. If the FAQ does not clearly answer the question, reply with exactly: NO_ANSWER\n\n"
        "FAQ:\n" + faq
    )
    try:
        reply = await chat_async(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Player question:\n{question}"}],
            temperature=0.3,
            max_tokens=400,
            model=config.get("FAQModel") or utility_model(),
        )
    except Exception:
        logging.exception("FAQ: LLM call failed")
        return None
    reply = (reply or "").strip()
    if not reply or "NO_ANSWER" in reply:
        return None
    return reply


async def _post_faq_answer(thread, question: str) -> bool:
    reply = await faq_answer(question)
    if reply is None:
        logging.info("FAQ: no coverage for thread %r", question[:80])
        return False
    footer = "\n-# I answer from the FAQ when a question has waited a while — fellow islanders may know even more."
    await safe_send(thread, reply + footer)
    logging.info("FAQ: answered thread %r", question[:80])
    return True


# ---- /ask: private, on-demand FAQ lookup ----
_ask_buckets: dict[int, TokenBucket] = {}


def _get_ask_bucket(user_id: int) -> TokenBucket:
    if user_id not in _ask_buckets:
        _ask_buckets[user_id] = TokenBucket(capacity=3, refill_rate=1.0 / 20.0)
    return _ask_buckets[user_id]


@tree.command(name="ask", description="Ask the game FAQ — the answer is shown only to you")
@app_commands.describe(question="Your question about the game")
async def ask_command(interaction: discord.Interaction, question: str):
    try:
        if not config.get("FAQEnabled", True):
            await interaction.response.send_message("The FAQ is currently disabled.", ephemeral=True)
            return
        if not _get_ask_bucket(interaction.user.id).consume():
            await interaction.response.send_message("Easy, darling — one question at a time. Try again in a moment.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        answer = await faq_answer(question.strip()[:500])
        if answer is None:
            forum_id = int(config.get("QuestionsForumID", 0))
            hint = f" Ask in <#{forum_id}> and a fellow islander will help." if forum_id else ""
            await interaction.followup.send(f"The FAQ doesn't cover that one.{hint}", ephemeral=True)
            logging.info("/ask: no coverage for %r", question[:80])
            return
        await interaction.followup.send(answer[:1900], ephemeral=True)
        logging.info("/ask: answered %r", question[:80])
    except Exception:
        logging.exception("/ask failed")


# ---- /wiki: search the game wiki (MediaWiki API, no LLM) ----
def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").replace("&quot;", '"').replace("&amp;", "&").replace("&#039;", "'")


@tree.command(name="wiki", description="Search the game wiki")
@app_commands.describe(term="What to search for", share="Post the result publicly instead of only to you")
async def wiki_command(interaction: discord.Interaction, term: str, share: bool = False):
    try:
        base = (config.get("WikiBaseURL") or "").rstrip("/")
        if not base:
            await interaction.response.send_message("No wiki is configured.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=not share, thinking=True)
        url = f"{base}/w/api.php?action=query&list=search&format=json&srlimit=5&srprop=snippet&srsearch=" + urllib.parse.quote(term)
        results = []
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
                async with s.get(url, headers={"User-Agent": "IsaBot"}) as r:
                    r.raise_for_status()
                    results = (await r.json()).get("query", {}).get("search", [])
        except Exception:
            logging.exception("Wiki search failed")
            await interaction.followup.send("The wiki isn't answering right now — try again in a bit.", ephemeral=not share)
            return
        if not results:
            await interaction.followup.send(f"No wiki page found for **{term[:100]}**.", ephemeral=not share)
            return
        top = results[0]
        link = lambda t: f"{base}/wiki/" + urllib.parse.quote(t.replace(" ", "_"))
        embed = discord.Embed(title=top["title"], url=link(top["title"]), description=_strip_html(top.get("snippet", ""))[:400] + "…", color=0x3498DB)
        if len(results) > 1:
            embed.add_field(name="More results", value="\n".join(f"[{r['title']}]({link(r['title'])})" for r in results[1:5]), inline=False)
        embed.set_footer(text="Wicked Island wiki")
        await interaction.followup.send(embed=embed, ephemeral=not share)
    except Exception:
        logging.exception("/wiki failed")


# ---- /lore: in-character lore lookup grounded in the lore file ----
@tree.command(name="lore", description="Ask about the world's lore — the answer is shown only to you")
@app_commands.describe(topic="What do you want to know about?")
async def lore_command(interaction: discord.Interaction, topic: str):
    try:
        if not LORE_CONTEXT:
            await interaction.response.send_message("I have no lore to share.", ephemeral=True)
            return
        if not _get_ask_bucket(interaction.user.id).consume():
            await interaction.response.send_message("Patience, darling — one tale at a time.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True, thinking=True)
        name = config.get("Name", "the bot")
        persona = (config.get("Personality") or "").strip()
        system = (
            f"You are {name}. Answer the question about the world using ONLY the lore below — never invent facts. "
            "Stay in character and keep it under 200 words; substance first, flavor second. "
            "If the lore does not cover the topic, say so in character in one or two lines.\n\n"
            + (f"Character:\n{persona[:1500]}\n\n" if persona else "")
            + "World lore:\n" + LORE_CONTEXT
        )
        try:
            reply = await chat_async(
                [{"role": "system", "content": system}, {"role": "user", "content": topic.strip()[:500]}],
                temperature=0.5,
                max_tokens=450,
                model=config.get("LoreModel") or utility_model(),
            )
        except Exception:
            logging.exception("/lore LLM call failed")
            reply = None
        reply = (reply or "").strip() or "The pages are silent on that, darling. Ask me something else."
        await interaction.followup.send(reply[:1900], ephemeral=True)
        logging.info("/lore: %r", topic[:80])
    except Exception:
        logging.exception("/lore failed")


# ---- /draw: the image pipeline as a proper command ----
def _image_channel_allowed(channel) -> bool:
    if channel is None:
        return False
    if isinstance(channel, discord.DMChannel):
        return True
    allowed = set(config.get("AllowedChannels", []))
    parent = getattr(channel, "parent", None)
    return channel.id in allowed or (parent is not None and parent.id in allowed)


@tree.command(name="draw", description="Ask for an image")
@app_commands.describe(
    prompt="What to draw",
    style="Style preset (optional)",
    aspect="Image shape",
    count="How many images (1-4)",
    exact="Send your prompt to Stable Diffusion unchanged (skip the rewrite)",
)
@app_commands.choices(aspect=[
    app_commands.Choice(name="square", value="square"),
    app_commands.Choice(name="portrait", value="portrait"),
    app_commands.Choice(name="landscape", value="landscape"),
])
async def draw_command(
    interaction: discord.Interaction,
    prompt: str,
    style: str | None = None,
    aspect: app_commands.Choice[str] | None = None,
    count: app_commands.Range[int, 1, 4] = 1,
    exact: bool = False,
):
    try:
        channel = interaction.channel
        if not images_enabled():
            await interaction.response.send_message(
                config.get("ImageDisabledNotice", "Image generation is currently disabled."),
                ephemeral=True)
            return
        blocked = image_prompt_blocked(prompt)
        if blocked:
            await interaction.response.send_message(config.get(
                "ImageRefusalMessage",
                "No. That is not something I will ever draw, and the moderators have been notified."), ephemeral=True)
            await refuse_image_request(channel, interaction.user.id,
                                       interaction.user.display_name, blocked, prompt)
            return
        if not _image_channel_allowed(channel):
            allowed = ", ".join(f"<#{c}>" for c in config.get("AllowedChannels", [])) or "my channels"
            await interaction.response.send_message(f"I only paint in {allowed} — or in DMs.", ephemeral=True)
            return
        if not get_user_bucket(interaction.user.id).consume():
            await interaction.response.send_message("You're requesting images too fast — slow down a bit.", ephemeral=True)
            return

        width = height = None
        if aspect and aspect.value == "portrait":
            size = config.get("SDPortraitSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])
        elif aspect and aspect.value == "landscape":
            size = config.get("SDLandscapeSize") or []
            if len(size) == 2:
                width, height = int(size[0]), int(size[1])

        positive_prefix = None
        neg = config.get("SDNegativePrompt", "(lowres, blurry, deformed)")
        if style:
            presets = config.get("SDStylePresets") or {}
            preset = next((v for k, v in presets.items() if k.lower() == style.lower()), None)
            if preset:
                positive_prefix = preset.get("positive", "")
                neg = preset.get("negative") or neg

        await interaction.response.send_message("Hang on while I sketch that for you…")
        status_msg = await interaction.original_response()
        raw = prompt.strip()[:1500]
        sd_prompt = raw if (exact or _looks_like_tag_prompt(raw)) else await compile_sd_prompt(raw)
        parent = getattr(channel, "parent", None)
        ch_id = parent.id if parent is not None else channel.id
        asyncio.create_task(run_image_job(
            channel,
            ch_id=ch_id,
            user_prompt=raw,
            sd_prompt=sd_prompt,
            neg=neg,
            batch=int(count),
            width=width,
            height=height,
            positive_prefix=positive_prefix,
            requested_by=interaction.user.display_name,
            requester_id=interaction.user.id,
            status_msg=status_msg,
        ))
    except Exception:
        logging.exception("/draw failed")


@draw_command.autocomplete("style")
async def _draw_style_autocomplete(interaction: discord.Interaction, current: str):
    presets = config.get("SDStylePresets") or {}
    return [app_commands.Choice(name=k, value=k) for k in presets if current.lower() in k.lower()][:25]


# ---- Highlights (starboard) ----
def _norm_emoji(e) -> str:
    return str(e).replace("\ufe0f", "")


def _highlight_emojis() -> set[str]:
    raw = config.get("HighlightEmojis") or [config.get("HighlightEmoji", "⭐")]
    return {_norm_emoji(e) for e in raw}


def _emoji_matches(e, accepted: set[str]) -> bool:
    """Unicode emoji match by character; custom server emoji also match by name."""
    if _norm_emoji(e) in accepted:
        return True
    name = getattr(e, "name", None)
    return bool(name) and _norm_emoji(name) in accepted


_highlight_in_flight: set[int] = set()


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    try:
        hl_id = int(config.get("HighlightsChannelID", 0))
        if not hl_id or payload.guild_id is None or payload.channel_id == hl_id:
            return
        accepted = _highlight_emojis()
        if not _emoji_matches(payload.emoji, accepted):
            return
        channel = bot.get_channel(payload.channel_id) or await bot.fetch_channel(payload.channel_id)
        msg = await channel.fetch_message(payload.message_id)
        images_only = config.get("HighlightImagesOnly", True)
        is_bot_image = msg.author.id == bot.user.id and any(
            (a.content_type or "").startswith("image/") for a in msg.attachments
        )
        if images_only and not is_bot_image:
            return
        if msg.author.bot and not is_bot_image:
            return
        # Unique voters across all accepted reactions (⭐ + ❤️ from one person = 1 vote).
        voters: set[int] = set()
        for reaction in msg.reactions:
            if _emoji_matches(reaction.emoji, accepted):
                async for u in reaction.users():
                    if u.id != bot.user.id:
                        voters.add(u.id)
        count = len(voters)
        if count < int(config.get("HighlightThreshold", 3)):
            return
        if msg.id in _highlight_in_flight:
            return
        db = get_db()
        if db.execute("SELECT 1 FROM highlights WHERE message_id = ?", (msg.id,)).fetchone():
            return
        _highlight_in_flight.add(msg.id)
        try:
            await _post_highlight(payload, msg, hl_id, count, is_bot_image)
        finally:
            _highlight_in_flight.discard(msg.id)
    except Exception:
        logging.exception("Highlight handler failed")


async def _post_highlight(payload, msg, hl_id: int, count: int, is_bot_image: bool):
    """Post a highlight, and only record it once the post actually succeeded —
    a failed post must stay eligible, or the image is silently lost for good."""
    try:
        hl = bot.get_channel(hl_id) or await bot.fetch_channel(hl_id)
        jump = f"https://discord.com/channels/{payload.guild_id}/{payload.channel_id}/{payload.message_id}"
        embed = discord.Embed(color=0xF1C40F)
        files = []
        if is_bot_image:
            rec = ipm.find_by_message(msg.id)
            requester = (rec.meta.get("by") if rec else None) or "unknown"
            prompt = (rec.user_prompt if rec else "") or ""
            embed.title = f"⭐ {count} · requested by {requester}"
            if prompt:
                embed.description = prompt[:300]
            att = next(a for a in msg.attachments if (a.content_type or "").startswith("image/"))
            data = await att.read()
            files.append(discord.File(io.BytesIO(data), filename=att.filename))
            embed.set_image(url=f"attachment://{att.filename}")
        else:
            embed.title = f"⭐ {count} · {msg.author.display_name}"
            embed.description = (msg.content or "")[:1000]
            embed.set_thumbnail(url=msg.author.display_avatar.url)
        embed.add_field(name="Source", value=f"[jump to message]({jump}) in <#{payload.channel_id}>")
        await hl.send(embed=embed, files=files)
    except discord.Forbidden:
        logging.error(
            "Highlight: cannot post in channel %s — the bot needs Send Messages, "
            "Embed Links and Attach Files there. Message %s will be retried on its next reaction.",
            hl_id, msg.id,
        )
        return
    db = get_db()
    db.execute("INSERT OR IGNORE INTO highlights (message_id, ts) VALUES (?, ?)", (msg.id, time.time()))
    db.commit()
    logging.info("Highlight: message %s (%d voters)", msg.id, count)


# ---- Server stats channels (member count + Steam players online) ----
async def _rename_if_changed(channel_id: int, name: str):
    ch = bot.get_channel(channel_id) or await bot.fetch_channel(channel_id)
    if ch.name != name:
        await ch.edit(name=name, reason="Stats update")
        logging.info("Stats: renamed channel %s -> %r", channel_id, name)


async def _update_stats_channels():
    if not config.get("StatsEnabled"):
        return
    mid = int(config.get("StatsMemberChannelID", 0))
    if mid:
        ch = bot.get_channel(mid)
        guild = ch.guild if ch else None
        count = getattr(guild, "member_count", None)
        if count:
            await _rename_if_changed(mid, str(config.get("StatsMemberTemplate", "all-members-{count}")).format(count=count))
    pid = int(config.get("StatsPlayersChannelID", 0))
    appid = int(config.get("SteamAppID", 0))
    if pid and appid:
        url = f"https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(url) as r:
                r.raise_for_status()
                players = (await r.json()).get("response", {}).get("player_count")
        if players is not None:
            await _rename_if_changed(pid, str(config.get("StatsPlayersTemplate", "in-game-now-{count}")).format(count=players))


async def _stats_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            await _update_stats_channels()
        except Exception:
            logging.exception("Stats update failed")
        # Discord allows 2 channel-name edits per 10 minutes — never poll faster.
        await asyncio.sleep(max(600, int(config.get("StatsUpdateMin", 10)) * 60))


# ---- Translate (message context menu) ----
_LOCALE_LANG = {
    "en-US": "English", "en-GB": "English", "de": "German", "sv-SE": "Swedish", "fr": "French",
    "es-ES": "Spanish", "es-419": "Spanish (Latin America)", "pt-BR": "Portuguese (Brazil)",
    "it": "Italian", "nl": "Dutch", "pl": "Polish", "ru": "Russian", "uk": "Ukrainian", "tr": "Turkish",
    "ja": "Japanese", "ko": "Korean", "zh-CN": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
    "cs": "Czech", "da": "Danish", "fi": "Finnish", "no": "Norwegian", "hu": "Hungarian", "ro": "Romanian",
    "el": "Greek", "bg": "Bulgarian", "hr": "Croatian", "lt": "Lithuanian", "th": "Thai", "vi": "Vietnamese",
    "id": "Indonesian", "hi": "Hindi", "ar": "Arabic", "he": "Hebrew",
}


def _locale_language(locale) -> str:
    code = str(locale)
    return _LOCALE_LANG.get(code) or _LOCALE_LANG.get(code.split("-")[0]) or f"the language for locale '{code}'"


_translate_buckets: dict[int, TokenBucket] = {}


@tree.context_menu(name="Translate")
async def translate_message(interaction: discord.Interaction, message: discord.Message):
    """Right-click a message → Apps → Translate: translation into the user's own Discord language, shown only to them."""
    try:
        text = (message.content or "").strip()
        if not text:
            await interaction.response.send_message("Nothing to translate in that message.", ephemeral=True)
            return
        bucket = _translate_buckets.setdefault(interaction.user.id, TokenBucket(capacity=5, refill_rate=5.0 / 60.0))
        if not bucket.consume():
            await interaction.response.send_message("Easy, darling — a few translations a minute is plenty.", ephemeral=True)
            return
        language = _locale_language(interaction.locale)
        await interaction.response.defer(ephemeral=True, thinking=True)
        system = (
            f"Translate the user's Discord message into {language}. Output ONLY the translation — no commentary. "
            "Preserve meaning, tone, slang, emoji, @mentions, links, and Discord formatting. "
            f"If the message is already in {language}, reply with a one-line note in {language} saying so."
        )
        try:
            translated = await chat_async(
                [{"role": "system", "content": system}, {"role": "user", "content": text[:1800]}],
                temperature=0.2,
                max_tokens=700,
                model=config.get("TranslateModel") or utility_model(),
            )
        except Exception:
            logging.exception("Translate: LLM call failed")
            translated = None
        translated = (translated or "").strip()
        if not translated:
            await interaction.followup.send("I couldn't translate that one — try again in a moment.", ephemeral=True)
            return
        await interaction.followup.send(f"**{language}:**\n{translated[:1850]}", ephemeral=True)
        logging.info("Translate: %s -> %s (%d chars)", interaction.user, language, len(text))
    except Exception:
        logging.exception("Translate failed")


async def _sync_app_commands():
    """Register slash commands. A guild-scoped sync is instant; global takes up to an hour."""
    await bot.wait_until_ready()
    try:
        gid = int(config.get("AppCommandGuildID", 0))
        if gid:
            guild = discord.Object(id=gid)
            tree.copy_global_to(guild=guild)
            synced = await tree.sync(guild=guild)
        else:
            synced = await tree.sync()
        logging.info("Synced %d app command(s): %s", len(synced), [c.name for c in synced])
    except Exception:
        logging.exception("App command sync failed")


# thread_id -> FAQ file mtime when it was judged. A thread is evaluated once;
# editing the FAQ file gives every still-open thread one fresh look.
_faq_evaluated: dict[int, float] = {}


async def _faq_scan_once():
    forum_id = int(config.get("QuestionsForumID", 0))
    if not forum_id or not config.get("FAQEnabled", True):
        return
    forum = bot.get_channel(forum_id)
    if forum is None or not hasattr(forum, "threads"):
        return
    try:
        faq_mtime = os.path.getmtime(config.get("FAQPath", "game_faq.txt"))
    except OSError:
        return
    delay = timedelta(minutes=int(config.get("FAQAnswerDelayMin", 30)))
    max_age = timedelta(hours=int(config.get("FAQMaxThreadAgeHours", 24)))
    max_answers = int(config.get("FAQMaxAnswersPerScan", 3))
    now = datetime.now(timezone.utc)
    answered = 0

    for thread in list(forum.threads):
        if answered >= max_answers:
            break
        created = getattr(thread, "created_at", None)
        if created is None:
            continue
        age = now - created
        if age < delay or age > max_age:
            continue
        if _faq_evaluated.get(thread.id) == faq_mtime:
            continue  # already judged against this version of the FAQ
        try:
            msgs = [m async for m in thread.history(limit=50, oldest_first=True)]
        except Exception:
            continue
        if any(m.author.id == bot.user.id for m in msgs):
            continue  # we already answered
        owner_id = thread.owner_id
        if any(not m.author.bot and m.author.id != owner_id for m in msgs):
            continue  # a human already replied
        starter = next((m for m in msgs if m.author.id == owner_id and (m.content or "").strip()), None)
        question = thread.name or ""
        if starter:
            question = f"{question}\n{starter.content[:1500]}"
        _faq_evaluated[thread.id] = faq_mtime
        if await _post_faq_answer(thread, question):
            answered += 1


async def _faq_answer_loop():
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            await _faq_scan_once()
        except Exception:
            logging.exception("FAQ scan failed")
        await asyncio.sleep(600)


# =============================================================================
# Daily Server Summary
# =============================================================================
SUMMARY_HOUR = int(config.get("SummaryHour", 8))
SUMMARY_TZ = ZoneInfo(config.get("SummaryTimezone", "Europe/Stockholm"))
SUMMARY_MAX_PER_CHANNEL = int(config.get("SummaryMaxPerChannel", 500))


async def _fetch_channel_messages(channel: discord.TextChannel, after: datetime) -> list[str]:
    lines: list[str] = []
    try:
        me = channel.guild.me
        if me is None:
            return []
        perms = channel.permissions_for(me)
        if not perms.read_message_history:
            return []

        count = 0
        async for msg in channel.history(after=after, limit=SUMMARY_MAX_PER_CHANNEL, oldest_first=True):
            if msg.author.bot:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue
            if len(content) > 300:
                content = content[:300] + "…"
            lines.append(f"{msg.author.display_name}: {content}")
            count += 1

        logging.info("Summary: fetched %d messages from #%s", count, channel.name)
    except discord.Forbidden:
        pass
    except Exception:
        logging.exception("Summary: failed to fetch from #%s", channel.name)
    return lines


async def _summarize_channel(channel_name: str, messages: list[str]) -> str:
    joined = "\n".join(messages)
    if len(joined) > 12000:
        joined = joined[:12000] + "\n[…truncated]"

    system = (
        "You are a concise server activity summarizer. "
        "Summarize the following Discord channel conversation. "
        "Focus on: key topics discussed, decisions made, questions asked, "
        "notable community interactions, and general sentiment. "
        "Skip greetings, small talk, and bot responses. "
        "Write 2-5 sentences. Be specific — mention usernames when relevant."
    )
    user = f"Channel: #{channel_name}\n\n{joined}"

    try:
        result = await chat_async(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3,
            max_tokens=300,
            model=utility_model(),
        )
        return (result or "").strip()
    except Exception:
        logging.exception("Summary: LLM failed for #%s", channel_name)
        return f"(Summary failed for #{channel_name})"


async def _compile_digest(channel_summaries: list[tuple[str, str, int]]) -> str:
    parts = []
    for name, summary, count in channel_summaries:
        parts.append(f"#{name} ({count} messages):\n{summary}")
    combined = "\n\n".join(parts)

    system = (
        "You are a server activity digest writer. "
        "Compile the per-channel summaries below into a clean daily briefing for a server owner. "
        "Group related topics across channels if they connect. "
        "Highlight anything that might need the owner's attention (complaints, questions directed at devs, "
        "heated discussions, bug reports, feature requests). "
        "End with a quick overall sentiment read. "
        "Keep the total digest under 800 words. Use markdown formatting for readability."
    )

    try:
        result = await chat_async(
            [{"role": "system", "content": system}, {"role": "user", "content": combined}],
            temperature=0.3,
            max_tokens=1200,
            model=utility_model(),
        )
        return (result or "").strip()
    except Exception:
        logging.exception("Summary: final digest LLM failed")
        return combined


async def generate_server_summary(hours: int = 24) -> str | None:
    guilds = bot.guilds
    if not guilds:
        logging.warning("Summary: bot is not in any guilds")
        return None

    after = datetime.now(timezone.utc) - timedelta(hours=hours)
    channel_summaries: list[tuple[str, str, int]] = []

    for guild in guilds:
        for channel in guild.text_channels:
            messages = await _fetch_channel_messages(channel, after)
            if len(messages) < 3:
                continue
            summary = await _summarize_channel(channel.name, messages)
            channel_summaries.append((channel.name, summary, len(messages)))

    if not channel_summaries:
        return f"No significant activity in the last {hours} hours."

    channel_summaries.sort(key=lambda x: x[2], reverse=True)
    digest = await _compile_digest(channel_summaries)

    header = f"**Server Summary — last {hours}h** ({len(channel_summaries)} active channels)\n\n"
    return header + digest


SUMMARY_LAST_RUN_PATH = config.get("SummaryLastRunPath", ".summary_last_run")


def _read_last_summary_ts() -> float:
    """Read the timestamp of the last successful summary from disk."""
    try:
        if os.path.exists(SUMMARY_LAST_RUN_PATH):
            return float(open(SUMMARY_LAST_RUN_PATH).read().strip())
    except Exception:
        pass
    return 0.0


def _write_last_summary_ts():
    """Write the current time as the last successful summary timestamp."""
    try:
        with open(SUMMARY_LAST_RUN_PATH, "w") as f:
            f.write(str(time.time()))
    except Exception:
        logging.exception("Failed to write summary timestamp")


async def _daily_summary_scheduler():
    await bot.wait_until_ready()

    while not bot.is_closed():
        try:
            now = datetime.now(SUMMARY_TZ)
            target = now.replace(hour=SUMMARY_HOUR, minute=0, second=0, microsecond=0)
            if now >= target:
                target += timedelta(days=1)

            wait_seconds = (target - now).total_seconds()
            logging.info("Summary: next run at %s (%d seconds)", target.isoformat(), int(wait_seconds))
            await asyncio.sleep(wait_seconds)

            # Scheduled digest can be disabled via config (manual !summary still works).
            # Checked here (not at startup) so it honors hot-reload.
            if not config.get("DailySummaryEnabled", True):
                logging.info("Summary: scheduled digest disabled via config; skipping.")
                continue

            # Skip if a summary was already sent in the last 20 hours (survives restarts)
            last_run = _read_last_summary_ts()
            if time.time() - last_run < 72000:  # 20 hours
                logging.info("Summary: skipping — already sent %.1fh ago", (time.time() - last_run) / 3600)
                continue

            logging.info("Summary: generating daily digest…")
            digest = await generate_server_summary(hours=24)

            if digest:
                try:
                    owner = await bot.fetch_user(OWNER_ID)
                    chunks = [digest[i:i + 1900] for i in range(0, len(digest), 1900)]
                    for chunk in chunks:
                        await owner.send(chunk)
                    _write_last_summary_ts()
                    logging.info("Summary: sent digest to owner (%d chars)", len(digest))
                except discord.Forbidden:
                    logging.error("Summary: cannot DM owner (DMs disabled?)")
                except Exception:
                    logging.exception("Summary: failed to send digest")
            else:
                logging.info("Summary: no digest generated")
        except asyncio.CancelledError:
            return
        except Exception:
            logging.exception("Summary scheduler error; retrying in 60s")
            await asyncio.sleep(60)


# =============================================================================
# Config Hot-Reload (file watcher)
# =============================================================================
async def _config_watch_loop():
    """Poll config file mtime every 10s; reload on change."""
    global config, _config_mtime
    while True:
        await asyncio.sleep(10)
        try:
            current_mtime = os.path.getmtime(CONFIG_PATH)
            if current_mtime > _config_mtime:
                config = load_config()
                _reload_derived_config()
                logging.info("Config hot-reloaded (file changed)")
            else:
                # Lore files are edited far more often than the config itself.
                for path, was in list(_lore_mtimes.items()):
                    if os.path.exists(path) and os.path.getmtime(path) > was:
                        load_lore()
                        logging.info("Lore hot-reloaded (%s changed)", path)
                        break
        except FileNotFoundError:
            pass
        except Exception:
            logging.exception("Config watch error")


# =============================================================================
# Graceful Shutdown
# =============================================================================
def _setup_signal_handlers():
    loop = asyncio.get_event_loop()

    def _handle_shutdown(sig):
        logging.info("Received %s — flushing state…", sig.name)
        cm.force_save()
        ipm.force_save()
        logging.info("State flushed. Closing bot.")
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(_handle_shutdown, sig))


# =============================================================================
# Run
# =============================================================================
_setup_signal_handlers()

# A rejected token is permanent until a human fixes it. Exiting with a code the
# service unit refuses to restart stops the bot hammering Discord's login
# endpoint every few seconds, which risks a temporary IP ban.
try:
    bot.run(config["DiscordToken"])
except discord.LoginFailure:
    logging.error(
        "Discord rejected the bot token. Generate a new one in the Developer Portal "
        "(Applications -> Bot -> Reset Token), put it in %s as DiscordToken, then start "
        "the service again. Not retrying.",
        CONFIG_PATH,
    )
    sys.exit(78)