import asyncio
import base64
import functools
import io
import json
import logging
import os
import random
import re
import signal
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
bot = discord.Client(intents=intents)


_background_tasks_started = False


@bot.event
async def on_ready():
    global _background_tasks_started
    logging.info("BOT READY as %s (id=%s)", bot.user, getattr(bot.user, "id", "n/a"))
    logging.info("RUNNING FILE: %s | PID: %s", __file__, os.getpid())
    if not _background_tasks_started:
        _background_tasks_started = True
        bot.loop.create_task(_periodic_save_loop())
        bot.loop.create_task(_daily_summary_scheduler())
        bot.loop.create_task(_config_watch_loop())


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


def _strip_token_ci(text: str, token: str) -> str:
    pattern = rf"\b{re.escape(token)}\b:?"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


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


# Per-channel bucket (existing behavior)
_channel_buckets: dict[int, TokenBucket] = {}


def get_channel_bucket(ch_id: int) -> TokenBucket:
    if ch_id not in _channel_buckets:
        _channel_buckets[ch_id] = TokenBucket(capacity=3, refill_rate=0.5)
    return _channel_buckets[ch_id]


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
            "Preserve key facts, decisions, user names, and context the assistant needs to continue naturally. "
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
    meta: dict[str, Any] = field(default_factory=dict)
    bot_message_id: int | None = None
    ts: float = 0.0


class ImagePromptMemory:
    def __init__(self):
        self.by_channel: dict[int, list[ImagePromptRecord]] = {}

    def add(self, rec: ImagePromptRecord):
        self.by_channel.setdefault(rec.channel_id, []).append(rec)

    def last_for_channel(self, channel_id: int) -> ImagePromptRecord | None:
        arr = self.by_channel.get(channel_id, [])
        return arr[-1] if arr else None


ipm = ImagePromptMemory()

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


async def stable_diffusion_generate_image(prompt: str) -> Image.Image | None:
    payload = {
        "prompt": config["SDPositivePrompt"] + prompt,
        "steps": config["SDSteps"],
        "width": config["SDWidth"],
        "height": config["SDHeight"],
        "cfg_scale": config["cfg_scale"],
        "negative_prompt": config["SDNegativePrompt"],
        "sampler_index": config["SDSampler"],
    }
    if config.get("scheduler"):
        payload["scheduler"] = config["scheduler"]

    timeout = aiohttp.ClientTimeout(total=120)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(config["SDURL"], json=payload) as r:
                r.raise_for_status()
                j = await r.json()
                return Image.open(io.BytesIO(base64.b64decode(j["images"][0])))
    except Exception as e:
        logging.exception("SD generation failed: %s", e)
        return None


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
IMAGE_TRIGGERS = ("draw", "paint", "generate an image", "make a picture", "render", "sketch", "illustrate")
EXACT_TRIGGERS = ("draw exact", "image exact", "img exact", "exact:")

FOLLOWUP_STARTS = (
    "same", "again", "keep", "also", "but now", "make it",
    "change", "adjust", "brighter", "darker", "add",
)


def is_exact_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EXACT_TRIGGERS)


def looks_like_image_request(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(k in t for k in IMAGE_TRIGGERS) or t.startswith(("img:", "image:", "art:"))


def looks_like_followup(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(t.startswith(s) for s in FOLLOWUP_STARTS)


def should_route_to_image_followup(message: discord.Message) -> bool:
    ch_id = channel_key(message)
    last = ipm.last_for_channel(ch_id)
    if not last:
        return False
    if message.reference and last.bot_message_id:
        try:
            if message.reference.message_id == last.bot_message_id:
                return True
        except Exception:
            pass
    if not looks_like_followup(message.content or ""):
        return False
    if last.ts <= 0 or (time.time() - last.ts > 120):
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
        async with message.channel.typing():
            reply = await chat_async(msgs, temperature=0.6, max_tokens=600)
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

        if not get_channel_bucket(ch_id).consume():
            await safe_send(message.channel, "I'm busy sketching — please wait.")
            return
        if not get_user_bucket(message.author.id).consume():
            await safe_send(message.channel, "You're requesting images too fast — slow down a bit.")
            return

        text_in = text_override if text_override is not None else (message.content or "")
        exact_mode = is_exact_trigger(text_in)

        raw = text_in
        for tok in ("draw", "image", "img", "art"):
            raw = _strip_token_ci(raw, tok)
        for phrase in EXACT_TRIGGERS:
            raw = re.sub(re.escape(phrase), "", raw, flags=re.IGNORECASE)
        if exact_mode:
            raw = _strip_token_ci(raw, "exact")
        raw = re.sub(r"\s+", " ", raw).strip(" -:;,. \n\t")

        last = ipm.last_for_channel(ch_id)
        if last and (looks_like_followup(text_in) or message.reference):
            await safe_send(message.channel, config.get("ImageRefinementNotice", "Refining the previous image…"))
            refined = await refine_image_prompt(last, text_in)
            sd_prompt = refined.get("prompt", last.final_sd_prompt)
            neg = refined.get("negative", last.negative_prompt)
        else:
            await safe_send(message.channel, "Hang on while I sketch that for you…")
            if exact_mode:
                sd_prompt = raw
                neg = config.get("SDNegativePrompt", "(lowres, blurry, deformed)")
            else:
                sd_prompt = await compile_sd_prompt(raw)
                neg = config.get("SDNegativePrompt", "(lowres, blurry, deformed)")

        async with message.channel.typing():
            img = await stable_diffusion_generate_image(sd_prompt)

        if not image_ok(img):
            await safe_send(message.channel, "I couldn't render that image — try tweaking the description.")
            return

        image_bytes = io.BytesIO()
        img.save(image_bytes, format="PNG")
        image_bytes.seek(0)

        files = [discord.File(image_bytes, filename="output.png")]
        content = sd_prompt
        if len(sd_prompt) > 1800:
            prompt_bytes = io.BytesIO(sd_prompt.encode("utf-8"))
            files.append(discord.File(prompt_bytes, filename="prompt.txt"))
            content = "Rendered image. Full prompt attached as prompt.txt"

        sent = await safe_send(message.channel, content, files=files)

        ipm.add(ImagePromptRecord(
            channel_id=ch_id,
            message_id=message.id,
            user_prompt=text_in,
            final_sd_prompt=sd_prompt,
            negative_prompt=neg,
            meta={"by": str(message.author)},
            bot_message_id=getattr(sent, "id", None),
            ts=time.time(),
        ))
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
_MOD_COMMANDS = {"!summary", "!activity", "!whois", "!flags", "!search", "!clearhistory", "!help"}


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
        logging.info("State flushed. Closing bot.")
        loop.create_task(bot.close())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, functools.partial(_handle_shutdown, sig))


# =============================================================================
# Run
# =============================================================================
_setup_signal_handlers()
bot.run(config["DiscordToken"])