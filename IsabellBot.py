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
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from logging.handlers import TimedRotatingFileHandler
from typing import Any

import aiohttp
import discord
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
LORE_PATH = config.get("LorePath", "world_lore.txt")
LORE_CONTEXT = ""
if os.path.exists(LORE_PATH):
    try:
        with open(LORE_PATH, "r", encoding="utf-8") as f:
            LORE_CONTEXT = f.read().strip()
        logging.info("Loaded lore from %s (%d chars)", LORE_PATH, len(LORE_CONTEXT))
    except Exception:
        logging.exception("Failed to load lore file")


def _reload_derived_config():
    """Update module-level derived values after a config reload."""
    global IGNORED_USERS, IGNORED_WORDS, OWNER_ID
    global MOD_CHANNEL_ID, MOD_ROLE_ID, TRAP_CHANNEL_ID, EXEMPT_ROLE_IDS
    IGNORED_USERS = set(config.get("IgnoredUsers", []))
    IGNORED_WORDS = {w.lower() for w in config.get("IgnoredWords", [])}
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


class LLMResponseError(Exception):
    """Raised when the API returns 200 but no usable completion (e.g. provider error or content flag)."""


async def chat_async(messages: list[dict[str, str]], _retries: int = 3, **kwargs) -> str:
    """Run a chat completion with retry + exponential backoff."""
    kwargs.pop("reasoning", None)
    kwargs["extra_body"] = kwargs.get("extra_body", {})
    kwargs["extra_body"]["reasoning"] = {"enabled": False}

    last_exc: Exception | None = None
    for attempt in range(1, _retries + 1):
        try:
            resp = await llm_client.chat.completions.create(
                messages=messages,
                model=config["OpenaiModel"],
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

            return resp.choices[0].message.content
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
    trigger_message_id: int = 0,
    status_msg=None,
):
    """Queue a generation, show progress, deliver the result with action buttons."""
    global _sd_waiting
    try:
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
        raw = await chat_async(msgs, temperature=0.0, max_tokens=tok_budget)
        return (raw or "")[:max_chars]
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
        raw = await chat_async(msgs, temperature=0.0, max_tokens=max(256, min(2000, max_chars // 3)))
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


def looks_like_image_request(text: str) -> bool:
    t = (text or "").strip()
    return bool(_IMAGE_TRIGGER_RE.search(t)) or t.lower().startswith(("img:", "image:", "art:"))


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
def build_system_prefix() -> str:
    name = config.get("Name", "Assistant")
    persona = (config.get("Personality") or "").strip()

    parts = [f"You are {name}."]
    if persona:
        parts.append(f"\nStay in character as {name}:\n{persona}")
    if LORE_CONTEXT:
        parts.append(f"\n\nWorld knowledge (use this to answer questions about the world):\n{LORE_CONTEXT}")
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
            mod_ch = None
            if MOD_CHANNEL_ID:
                mod_ch = bot.get_channel(MOD_CHANNEL_ID) or await bot.fetch_channel(MOD_CHANNEL_ID)
            if mod_ch:
                quip = pick_mod_quip(member.display_name)
                status = removal_word if removed_ok else "NOT removed (action failed)"
                if total_deleted < 0:
                    cleanup_line = f"🧹 Discord wiped their messages from the last {delete_seconds // 60} min."
                else:
                    cleanup_line = f"🧹 Deleted ~{total_deleted} message(s) from the last {delete_seconds // 60} min."
                msg = (
                    f"👢 **{member.display_name}** was {status} (honeypot).\n"
                    f"{cleanup_line}\n"
                    f"{quip}"
                )
                await safe_send(mod_ch, msg)
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
        embed = discord.Embed(title="Member banned", color=0x992D22)
        embed.add_field(name="User", value=f"{user} ({user.id})")
        await _modlog_send(embed)
    except Exception:
        logging.exception("on_member_ban failed")


@bot.event
async def on_member_unban(guild: discord.Guild, user):
    try:
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

        cm.add_user(ch_id, is_dm, message.author.id, message.author.display_name, message.content, message.id)
        await cm.maybe_compress(ch_id)

        system_prefix = build_system_prefix()
        msgs = cm.build_messages(ch_id, system_prefix=system_prefix)

        if text_override is not None and msgs and msgs[-1]["role"] == "user":
            if not is_dm:
                msgs[-1] = {"role": "user", "content": f"[{message.author.display_name}]: {text_override}"}
            else:
                msgs[-1] = {"role": "user", "content": text_override}

        logging.info("TEXT -> LLM | ch=%s | msg_id=%s", ch_id, message.id)
        freq_pen = float(config.get("FrequencyPenalty", 0.3))
        pres_pen = float(config.get("PresencePenalty", 0.3))
        async with message.channel.typing():
            reply = await chat_async(
                msgs, temperature=0.6, max_tokens=600,
                frequency_penalty=freq_pen, presence_penalty=pres_pen,
            )

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
async def _post_faq_answer(thread, question: str) -> bool:
    faq_path = config.get("FAQPath", "game_faq.txt")
    try:
        with open(faq_path, "r", encoding="utf-8") as f:
            faq = f.read().strip()
    except Exception:
        logging.exception("FAQ: cannot read %s", faq_path)
        return False
    if not faq:
        return False

    name = config.get("Name", "the bot")
    system = (
        f"You are {name}, answering a player's question in the game's Discord questions forum. "
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
        )
    except Exception:
        logging.exception("FAQ: LLM call failed")
        return False
    reply = (reply or "").strip()
    if not reply or "NO_ANSWER" in reply:
        logging.info("FAQ: no coverage for thread %r", question[:80])
        return False
    footer = "\n-# I answer from the FAQ when a question has waited a while — fellow islanders may know even more."
    await safe_send(thread, reply + footer)
    logging.info("FAQ: answered thread %r", question[:80])
    return True


async def _faq_scan_once():
    forum_id = int(config.get("QuestionsForumID", 0))
    if not forum_id or not config.get("FAQEnabled", True):
        return
    forum = bot.get_channel(forum_id)
    if forum is None or not hasattr(forum, "threads"):
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
bot.run(config["DiscordToken"])