import discord
import json
import time
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Any
import io
import base64
import aiohttp
from PIL import Image
from openai import OpenAI
import os
import re
from datetime import datetime, timezone, timedelta  # <-- added

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
file_handler = TimedRotatingFileHandler('app.log', when='midnight', interval=1, backupCount=7)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger()
# avoid duplicate handlers on reload
if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
    logger.addHandler(file_handler)

# =============================================================================
# Config
# =============================================================================
CONFIG_PATH = os.environ.get("BOT_CONFIG", "Config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    logging.info("Loaded config: %s", {k: v for k, v in config.items() if k not in {"DiscordToken","Personality","OpenAPIKey"}})
    logging.info("Persona chars: %d", len((config.get("Personality") or "")))
    logging.info("=== BOOT: Discord Bot (single-file, SD-only) ===")
except Exception as e:
    logging.exception("Failed to load config: %s", e)
    raise

# Optional lore/RAG path (used for TEXT; gated for IMAGES)
LORE_PATH = config.get("LorePath", "world_lore.json")

# Conversation history paths:
# - Channels -> single JSON file
# - DMs -> one JSON per user ID in a directory
CHANNEL_HISTORY_PATH = config.get("ChannelHistoryPath", "channel_history.json")
DM_HISTORY_DIR = config.get("DMHistoryDir", "dm_history")

# =============================================================================
# OpenAI Client
# =============================================================================
if config.get("OpenAPIKey"):
    llm_client = OpenAI(base_url=config["OpenAPIEndpoint"], api_key=config["OpenAPIKey"])
else:
    llm_client = OpenAI(base_url=config["OpenAPIEndpoint"])

async def chat_async(messages: List[Dict[str, str]], **kwargs) -> str:
    """Run a chat completion in a thread; return text.
    Reasoning is hard-disabled for all models (e.g., Grok 4.1 Fast).
    """
    loop = asyncio.get_running_loop()
    # Ensure callers can't override reasoning: always disabled
    if "reasoning" in kwargs:
        kwargs.pop("reasoning", None)
    
    # NEW: Explicitly disable reasoning for Grok 4.1 Fast (or similar models)
    # This overrides any default and ensures pure fast mode
    kwargs["extra_body"] = kwargs.get("extra_body", {})
    kwargs["extra_body"]["reasoning"] = {"enabled": False}
    
    resp = await loop.run_in_executor(
        None,
        lambda: llm_client.chat.completions.create(
            messages=messages,
            model=config["OpenaiModel"],  # Ensure this is "x-ai/grok-4.1-fast" or similar
            **kwargs,
        )
    )
    return resp.choices[0].message.content

# =============================================================================
# Discord Client
# =============================================================================
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
bot = discord.Client(intents=intents)

@bot.event
async def on_ready():
    try:
        logging.info("BOT READY as %s (id=%s)", bot.user, getattr(bot.user, 'id', 'n/a'))
        logging.info("RUNNING FILE: %s | PID: %s", __file__, os.getpid())
    except Exception:
        logging.exception("on_ready logging failed")

# =============================================================================
# Small Utilities
# =============================================================================
def clamp_2000(text: str) -> str:
    """Discord message hard limit."""
    return (text or "")[:2000]

async def safe_send(channel: discord.abc.Messageable, text: str = None, **kwargs):
    """Send but don't crash the task if Discord rejects; logs on failure."""
    try:
        if text is not None:
            return await channel.send(clamp_2000(text), **kwargs)
        return await channel.send(**kwargs)
    except Exception:
        logging.exception("safe_send failed")

def channel_key(message: discord.Message) -> int:
    """Threads share memory with parent; DMs are per-user (author ID as key)."""
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

async def ensure_can_send(message: discord.Message) -> bool:
    """Check basic send/attach perms in guild channels; DMs always ok."""
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

# Case-insensitive token stripper (with optional trailing colon)
def _strip_token_ci(text: str, token: str) -> str:
    """Remove a token (optionally followed by ':') case-insensitively with word boundaries."""
    pattern = rf"\b{re.escape(token)}\b:?"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)

# =============================================================================
# Token Bucket (per-channel)
# =============================================================================
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_time = time.time()
        self.refill_rate = refill_rate

    def consume(self, tokens: int) -> bool:
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill_time) * self.refill_rate)
        self.last_refill_time = now
        if tokens <= self.tokens:
            self.tokens -= tokens
            return True
        return False

_buckets: Dict[int, TokenBucket] = {}
def get_bucket(ch_id: int) -> TokenBucket:
    b = _buckets.get(ch_id)
    if not b:
        b = _buckets[ch_id] = TokenBucket(capacity=3, refill_rate=0.5)
    return b

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
    turns: Deque[Tuple[str, str]] = field(default_factory=lambda: deque(maxlen=40))
    utterances: Deque[Utterance] = field(default_factory=lambda: deque(maxlen=100))
    summary: str = ""
    is_dm: bool = False  # True for DM conversations, False for guild channels

class ConversationManager:
    def __init__(
        self,
        maxlen_turns: int = 40,
        channel_history_path: Optional[str] = None,
        dm_history_dir: Optional[str] = None,
    ):
        self._by_channel: Dict[int, ConversationWindow] = {}
        self.maxlen_turns = maxlen_turns
        self.channel_history_path = channel_history_path
        self.dm_history_dir = dm_history_dir
        self._utterance_maxlen = 100

        if self.dm_history_dir:
            os.makedirs(self.dm_history_dir, exist_ok=True)

        self._load_from_disk()

    # ---------- Persistence helpers ----------

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
                    turns = deque(cv_data.get("turns", []), maxlen=self.maxlen_turns)
                    uttrs = deque(maxlen=self._utterance_maxlen)
                    for u in cv_data.get("utterances", []):
                        uttrs.append(Utterance(
                            author_id=u.get("author_id"),
                            author_name=u.get("author_name", ""),
                            content=u.get("content", ""),
                            message_id=u.get("message_id", 0),
                            ts=u.get("ts", 0.0),
                        ))
                    is_dm = bool(cv_data.get("is_dm", False))
                    self._by_channel[ch_id] = ConversationWindow(
                        channel_id=ch_id,
                        turns=turns,
                        utterances=uttrs,
                        summary=cv_data.get("summary", ""),
                        is_dm=is_dm,
                    )
                    loaded += 1
                except Exception:
                    logging.exception("Failed to load conversation window for channel %r", ch_key)
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
                turns = deque(cv_data.get("turns", []), maxlen=self.maxlen_turns)
                uttrs = deque(maxlen=self._utterance_maxlen)
                for u in cv_data.get("utterances", []):
                    uttrs.append(Utterance(
                        author_id=u.get("author_id"),
                        author_name=u.get("author_name", ""),
                        content=u.get("content", ""),
                        message_id=u.get("message_id", 0),
                        ts=u.get("ts", 0.0),
                    ))
                # DMs are always is_dm=True
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

    def _save_to_disk(self):
        self._save_channels()
        self._save_dms()

    def _save_channels(self):
        path = self.channel_history_path
        if not path:
            return
        try:
            serializable: Dict[str, Any] = {}
            for ch_id, cv in self._by_channel.items():
                if cv.is_dm:
                    continue  # DMs saved separately
                serializable[str(ch_id)] = {
                    "channel_id": ch_id,
                    "turns": list(cv.turns),
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
                    "turns": list(cv.turns),
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

    # ---------- Existing-ish API ----------

    def get(self, channel_id: int, is_dm: Optional[bool] = None) -> ConversationWindow:
        cv = self._by_channel.get(channel_id)
        if cv is None:
            cv = ConversationWindow(channel_id=channel_id, is_dm=bool(is_dm))
            self._by_channel[channel_id] = cv
            self._save_to_disk()
        else:
            # If we learn later that this is DM vs channel, update flag
            if is_dm is not None and cv.is_dm != is_dm:
                cv.is_dm = is_dm
                self._save_to_disk()
        return cv

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
        # IMPORTANT: don't embed speaker name in LLM content
        cv.turns.append(("user", content))
        self._save_to_disk()

    def add_assistant(self, channel_id: int, content: str):
        cv = self.get(channel_id)
        cv.turns.append(("assistant", content))
        self._save_to_disk()

    def build_messages(self, channel_id: int, system_prefix: str, extra_context: str = "") -> List[Dict[str, str]]:
        c = self.get(channel_id)
        msgs = [{"role": "system", "content": system_prefix.strip()}]
        if c.summary:
            msgs.append({"role": "system", "content": f"Conversation summary so far:\n{c.summary}"})
        if extra_context:
            msgs.append({"role": "system", "content": extra_context.strip()})
        msgs.extend({"role": r, "content": t} for r, t in list(c.turns))
        return msgs

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
    meta: Dict[str, Any] = field(default_factory=dict)
    bot_message_id: Optional[int] = None
    ts: float = 0.0  # timestamp of when the image was generated

class ImagePromptMemory:
    def __init__(self):
        self.by_channel: Dict[int, List[ImagePromptRecord]] = {}

    def add(self, rec: ImagePromptRecord):
        arr = self.by_channel.setdefault(rec.channel_id, [])
        arr.append(rec)

    def last_for_channel(self, channel_id: int) -> Optional[ImagePromptRecord]:
        arr = self.by_channel.get(channel_id, [])
        return arr[-1] if arr else None

ipm = ImagePromptMemory()

# =============================================================================
# RAG (simple overlap)
# =============================================================================
class KnowledgeIndex:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.docs: List[str] = []
        self.ready = False

    def build(self):
        if not os.path.exists(self.json_path):
            logging.info("Lore file not found at %s — RAG disabled.", self.json_path)
            self.ready = False
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            logging.exception("Failed to load lore file")
            self.ready = False
            return
        if not isinstance(data, list):
            logging.error("Lore file must be a JSON list of entries")
            self.ready = False
            return
        self.docs = [f"{row.get('title','')}\n{row.get('text','')}" for row in data]
        self.ready = True
        logging.info("RAG index built from %d entries.", len(self.docs))

    def query(self, q: str, k: int = 4) -> List[str]:
        if not self.ready or not q.strip():
            return []
        q_words = set((q or "").lower().split())
        scored = []
        for d in self.docs:
            text = d.lower()
            score = sum(1 for w in q_words if w in text)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:k]]

    @staticmethod
    def format_context(results: List[str]) -> str:
        if not results:
            return ""
        return "Relevant world info:\n" + "\n".join(results)

kb = KnowledgeIndex(LORE_PATH)
kb.build()

# =============================================================================
# Image Generation (Stable Diffusion only)
# =============================================================================
def image_ok(img: Optional[Image.Image]) -> bool:
    if img is None:
        return False
    try:
        w, h = img.size
        return w > 0 and h > 0
    except Exception:
        return False

async def stable_diffusion_generate_image(prompt: str) -> Optional[Image.Image]:
    payload = {
        "prompt": config["SDPositivePrompt"] + prompt,
        "steps": config["SDSteps"],
        "width": config["SDWidth"],
        "height": config["SDHeight"],
        "cfg_scale": config["cfg_scale"],
        "negative_prompt": config["SDNegativePrompt"],
        "sampler_index": config["SDSampler"],
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(config["SDURL"], json=payload, timeout=120) as r:
                r.raise_for_status()
                j = await r.json()
                return Image.open(io.BytesIO(base64.b64decode(j["images"][0])))
    except Exception as e:
        logging.exception("SD generation failed: %s", e)
        return None

# =============================================================================
# Intent-focused SD prompt builder (neutral; NSFW handled via instruction only)
# =============================================================================
async def compile_sd_prompt(user_text: str, rag_context: str) -> str:
    """
    Build a well-formatted Stable Diffusion prompt that reflects the user's *intent*.
    - Rephrase, organize, and clarify what's already there; do not introduce new subjects/styles.
    - Use optional context only to disambiguate terms already present in the user text.
    - Return a single comma-separated line suitable for SD.
    - If (and only if) the user's intent is clearly adult/NSFW, enhance explicit intent tastefully and safely.
    """
    max_chars = int(config.get("ImagePromptMaxChars", 1600))  # configurable and larger
    name = (config.get("Name") or "the assistant").strip()

    system = (
    "You are an expert, ultra-lewd prompt engineer specialized in Pony Realism models (Pony Diffusion V6, AutismMix, etc.).\n\n"
    "Your only task: rewrite the USER PROMPT into ONE single, comma-separated line that is perfectly optimized for Pony Realistic SDXL and as graphically pornographic as possible.\n\n"
    "STRICT RULES:\n"
    "- Output exactly one line of pure tags, no quotes, no explanations, nothing else.\n"
    f"- Stay under {max_chars} characters (including spaces).\n"
    "- Massively amplify every existing sexual element — turn anything vaguely erotic into raw, explicit pornography.\n"
    "- Use ( ) around crucial parts for emphasis → example: (wet pussy dripping grool:1.3)\n"
    "- Use [[ultra-detailed]] or (concept:1.4) when you need extreme strength.\n"
    "- Intensify vague terms → \"touching\" → (fingers deep inside vagina, spreading labia), \"horny\" → (ahegao, heart-shaped pupils, tongue out, drooling)\n"
    "- Always add relevant Pony-favorite body tags when implied: perfect breasts, erect nipples, detailed anus, glistening skin, sweat, saliva strands, pubic hair or shaved pussy, cameltoe, thighhighs, torn clothes, micro bikini, etc.\n"
    "- Camera & lighting only when it fits the scene: from below, side view, depth of field, cinematic lighting, backlighting, rim lighting\n"
    "- NEVER add new characters, new locations, new sex acts, or new clothing not implied by the user prompt.\n"    
    f"- NEVER mention {name}, sorceress, purple eyes, gold dress, Queen’s Refuge, or any persona metadata unless the USER PROMPT explicitly references it.\n\n"
    "Example output (just showing style):\n"
    "1girl, solo, (spread legs:1.3), (pussy juice dripping down thighs:1.4), ahegao, tongue out, erect clitoris, detailed vagina, anus visible, sweat, torn black thighhighs, micro bikini pulled aside, cinematic lighting, depth of field"
    )

    user = (
        "USER PROMPT:\n"
        f"{user_text}\n\n"
        "OPTIONAL CONTEXT (use ONLY to disambiguate terms already in the USER PROMPT; ignore otherwise):\n"
        f"{rag_context or '(none)'}"
    )

    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    try:
        tok_budget = max(256, min(2000, max_chars // 3))  # scale with desired length
        raw = await chat_async(msgs, temperature=0.0, max_tokens=tok_budget)
        return (raw or "")[:max_chars]
    except Exception:
        logging.exception("LLM prompt compose failed; returning user text")
        return (user_text or "")[:max_chars]

# =============================================================================
# Refiner for follow-ups (neutral; same NSFW guidance)
# =============================================================================
async def refine_image_prompt(last: ImagePromptRecord, followup_text: str) -> Dict[str, str]:
    max_chars = int(config.get("ImagePromptMaxChars", 1600))
    name = (config.get("Name") or "the assistant").strip()
    system = (
        "Refine a Stable Diffusion prompt based on a follow-up.\n"
        "- Preserve subject, style tokens, and critical descriptors from the previous prompt.\n"
        "- Merge ONLY new instructions.\n"
        f"- Do NOT introduce {name}, any assistant persona, or backstory unless explicitly mentioned in the follow-up.\n"
        "NSFW HANDLING (only if the FOLLOW-UP clearly implies adult/NSFW intent):\n"
        "  • Adults only (18+), consent-only, exclude illegal/taboo content.\n"
        "  • Clarify adult aesthetic details already implied; do not invent new subjects/kinks.\n"
        f"- Keep it concise (<= {max_chars} chars).\n"
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
        logging.warning("Refine parse failed; fallback append strategy")
        return {"prompt": f"{last.final_sd_prompt}, {followup_text}"[:max_chars], "negative": last.negative_prompt}

# =============================================================================
# Heuristics
# =============================================================================
IMAGE_TRIGGERS = ("draw", "paint", "generate an image", "make a picture", "render", "sketch", "illustrate",)
FOLLOWUP_TRIGGERS = ("same", "again", "keep", "also", "but now", "make it", "change", "adjust", "brighter", "darker", "add",)
EXACT_TRIGGERS = ("draw exact", "image exact", "img exact", "exact:")

def is_exact_trigger(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in EXACT_TRIGGERS)

def looks_like_image_request(text: str) -> bool:
    t = (text or "").lower().strip()
    return any(k in t for k in IMAGE_TRIGGERS) or t.startswith(("img:", "image:", "art:"))

def looks_like_followup(text: str) -> bool:
    """
    Stricter: only treat as follow-up if the message STARTS with these patterns,
    so casual 'add'/'again' in the middle of a sentence won't match.
    """
    t = (text or "").lower().strip()
    starts = (
        "same",
        "again",
        "keep",
        "also",
        "but now",
        "make it",
        "change",
        "adjust",
        "brighter",
        "darker",
        "add",
    )
    return any(t.startswith(s) for s in starts)

def should_route_to_image_followup(message: discord.Message) -> bool:
    ch_id = channel_key(message)
    last = ipm.last_for_channel(ch_id)
    if not last:
        return False

    text = message.content or ""

    # 1) If the user is explicitly replying to the last image message, allow follow-up
    if message.reference and getattr(last, "bot_message_id", None):
        try:
            if message.reference.message_id == last.bot_message_id:
                return True
        except Exception:
            # If something goes wrong, fall back to non-reply logic
            pass

    # 2) Otherwise, only treat as follow-up if:
    #    - The message *looks* like a follow-up, AND
    #    - The last image is recent (within 2 minutes / 120 seconds)
    if not looks_like_followup(text):
        return False

    try:
        last_ts = float(getattr(last, "ts", 0.0))
    except Exception:
        last_ts = 0.0

    if last_ts <= 0.0:
        return False

    # 2 minutes = 120 seconds
    if time.time() - last_ts > 120:
        return False

    return True

# =============================================================================
# Persona / System Prompt (for TEXT chat only)
# =============================================================================
def build_system_prefix() -> str:
    name = config.get("Name", "Assistant")
    persona = (config.get("Personality") or "").strip()
    sys = f"You are {name}."
    if persona:
        sys += f"\n\nStay in character as {name}:\n{persona}"
    return sys

# =============================================================================
# Honeypot / Auto-kick config  (ADDED)
# =============================================================================
TRAP_CHANNEL_ID = 1415613507499724841  # channel where any post triggers a kick

EXEMPT_ROLE_IDS = {
    949556508906111027,
    997378434386899024,
    1244927969340817431,
    1113924929218420831,
    1045951663690756228,
    1086263211906584646,
}

# Dev moderator channel that receives kick notices
MOD_CHANNEL_ID = 1027977666755837973

# =============================================================================
# Moderator quip using Isabell's personality (ADDED)
# =============================================================================
async def generate_mod_quip(offender_name: str) -> str:
    """
    Ask the LLM for a short, playful (but respectful) moderation quip in Isabell's style.
    We use the bot's Name and Personality from config to keep the voice consistent.
    """
    name = (config.get("Name") or "Isabell").strip()
    persona = (config.get("Personality") or "").strip()

    system = (
        f"You are {name}. Speak in first person as {name}. "
        "Tone: witty, friendly, and professional—helpful to moderators, never rude. "
        "Write exactly ONE short line (10–25 words) announcing that a user was removed. "
        "No profanity, no personal attacks. Keep it light and readable in Discord."
    )
    if persona:
        system += "\n\nStay in character with the following personality:\n" + persona

    user = (
        f"User '{offender_name}' was kicked for posting in a restricted honeypot channel. "
        "Write the single-line announcement."
    )
    try:
        text = await chat_async(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=60
        )
        return (text or "").strip()
    except Exception:
        logging.exception("LLM quip generation failed")
        return f"{offender_name} tripped the honeypot—clean removal completed."

# =============================================================================
# Honeypot Guard (kick + purge + mod notice)  (ADDED)
# =============================================================================
async def honeypot_guard(message: discord.Message) -> bool:
    """
    If message is in the trap channel, delete it, purge author's messages from last 10 minutes,
    kick the author (unless exempt), and notify moderators with an LLM quip.
    Returns True if action was taken.
    """
    try:
        if message.guild is None:
            return False
        if message.author.bot:
            return False
        if message.channel.id != TRAP_CHANNEL_ID:
            return False

        guild = message.guild

        # Ensure we have a Member object
        member: discord.Member
        if isinstance(message.author, discord.Member):
            member = message.author
        else:
            try:
                member = await guild.fetch_member(message.author.id)
            except Exception:
                logging.exception("Failed to fetch member for honeypot action")
                return False

        # Exemptions: owner, admins, roles provided
        if member == guild.owner:
            return False
        if member.guild_permissions.administrator:
            return False
        if any(getattr(r, "id", 0) in EXEMPT_ROLE_IDS for r in getattr(member, "roles", [])):
            return False

        # Delete triggering message to keep the trap clean
        try:
            await message.delete()
        except Exception:
            logging.exception("Could not delete triggering honeypot message")

        # Purge last 10 minutes of this user's messages across text channels (where permitted)
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)

        async def purge_in_channel(ch: discord.TextChannel) -> int:
            try:
                me = guild.me or await guild.fetch_member(bot.user.id)
                perms = ch.permissions_for(me)
                if not (perms.read_message_history and perms.manage_messages):
                    return 0
                deleted = await ch.purge(
                    limit=None,
                    after=cutoff,
                    check=lambda m: m.author.id == member.id,
                    bulk=True,
                )
                return len(deleted)
            except Exception:
                return 0

        total_deleted = 0

        # Current channel first
        if isinstance(message.channel, discord.TextChannel):
            total_deleted += await purge_in_channel(message.channel)

        # Then other text channels in the guild
        for ch in guild.text_channels:
            if ch.id == getattr(message.channel, "id", None):
                continue
            total_deleted += await purge_in_channel(ch)

        reason = f"Posted in restricted honeypot channel #{getattr(message.channel, 'name', '?')} ({TRAP_CHANNEL_ID})"

        # Kick the member
        kicked_ok = False
        try:
            await guild.kick(member, reason=reason)
            kicked_ok = True
            logging.info("Honeypot: kicked %s; deleted ~%d msgs (last 10 min).", member, total_deleted)
        except Exception:
            logging.exception("Honeypot: failed to kick member")

        # Notify moderators (even if kick fails, if we got here we tried)
        try:
            mod_ch = bot.get_channel(MOD_CHANNEL_ID) or await bot.fetch_channel(MOD_CHANNEL_ID)
            if mod_ch:
                offender_name = getattr(member, "display_name", str(member))
                quip = await generate_mod_quip(offender_name)
                status = "kicked" if kicked_ok else "attempted to kick (failed)"
                msg = (
                    f"👢 **{offender_name}** was {status} for tripping the honeypot in "
                    f"#{getattr(message.channel, 'name', '?')}.\n"
                    f"🧹 Deleted ~{total_deleted} message(s) from the last 10 minutes.\n"
                    f"{quip}"
                )
                await safe_send(mod_ch, msg)
            else:
                logging.warning("Could not resolve moderator channel %s", MOD_CHANNEL_ID)
        except Exception:
            logging.exception("Failed to notify moderators about honeypot action")

        return True
    except Exception:
        logging.exception("Honeypot guard failed unexpectedly")
        return False

# =============================================================================
# Handlers
# =============================================================================
async def handle_text_message(message: discord.Message, text_override: Optional[str] = None):
    """Main text handler. Uses `text_override` for LLM if provided, but stores original for memory."""
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)
        is_dm = isinstance(message.channel, discord.DMChannel)

        cm.add_user(
            ch_id,
            is_dm,
            message.author.id,
            message.author.display_name,
            message.content,
            message.id,
        )

        rag_ctx = ""
        if kb.ready:
            hits = kb.query(message.content)
            rag_ctx = KnowledgeIndex.format_context(hits)

        system_prefix = build_system_prefix()
        msgs = cm.build_messages(ch_id, system_prefix=system_prefix, extra_context=rag_ctx)

        # If OnlyWhenCalled stripped the bot name, replace only the *content*, not with "name: msg"
        if text_override is not None and msgs and msgs[-1]["role"] == "user":
            msgs = msgs[:-1] + [{"role": "user", "content": text_override}]

        logging.info("TEXT -> LLM | ch=%s | msg_id=%s", ch_id, message.id)
        async with message.channel.typing():
            reply = await chat_async(msgs, temperature=0.6, max_tokens=600)
        cm.add_assistant(ch_id, reply)
        await safe_send(message.channel, reply)
    except asyncio.CancelledError:
        logging.info("TASK cancelled in handle_text_message")
        return
    except Exception:
        logging.exception("TASK error in handle_text_message")
        await safe_send(message.channel, "Oops—something went wrong with that one.")

async def handle_image_message(message: discord.Message, text_override: Optional[str] = None):
    """Image handler. Uses `text_override` when provided (sanitized text)."""
    try:
        if not await ensure_can_send(message):
            return
        ch_id = channel_key(message)
        if not get_bucket(ch_id).consume(1):
            await safe_send(message.channel, "I'm busy sketching—please wait.")
            return

        text_in = (text_override if text_override is not None else (message.content or ""))
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
            notice = config.get("ImageRefinementNotice", "Refining the previous image…")
            await safe_send(message.channel, notice)
            refined = await refine_image_prompt(last, text_in)
            sd_prompt, neg = refined.get("prompt", last.final_sd_prompt), refined.get("negative", last.negative_prompt)
        else:
            await safe_send(message.channel, "Hang on while I sketch that for you…")
            if exact_mode:
                sd_prompt = raw
                neg = config.get("SDNegativePrompt", "(lowres, blurry, deformed)")
            else:
                # RAG gated for IMAGE prompts
                use_ctx = bool(config.get("ImagePromptUseContext", False))
                ctx_terms = {t.lower() for t in config.get("ImageContextTerms", [])}
                name_token = (config.get("Name", "") or "").lower()
                if name_token:
                    ctx_terms.add(name_token)

                rag_ctx = ""
                if use_ctx and kb.ready and any(t in (text_in or "").lower() for t in ctx_terms):
                    hits = kb.query(text_in, k=4)
                    rag_ctx = KnowledgeIndex.format_context(hits)

                sd_prompt = await compile_sd_prompt(raw, rag_ctx)
                neg = config.get("SDNegativePrompt", "(lowres, blurry, deformed)")

        async with message.channel.typing():
            img = await stable_diffusion_generate_image(sd_prompt)

        if not image_ok(img):
            await safe_send(message.channel, "I couldn't render that image—try tweaking the description.")
            return

        image_bytes = io.BytesIO()
        img.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        file = discord.File(image_bytes, filename='output.png')

        # If the SD prompt is very long, attach it as a file so users can view it fully.
        files = [file]
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
            ts=time.time(),  # record when this image was generated
        ))
    except asyncio.CancelledError:
        logging.info("TASK cancelled in handle_image_message")
        return
    except Exception:
        logging.exception("TASK error in handle_image_message")
        await safe_send(message.channel, "Oops—image generation hit a snag.")

# =============================================================================
# Router
# =============================================================================
# Changed: we track each job by (channel_id, kind, job_id) so image jobs don't cancel each other.
channel_jobs: Dict[Tuple[int, str, int], asyncio.Task] = {}

def start_job(channel_id: int, kind: str, coro, job_id: Optional[int] = None):
    key = (channel_id, kind, job_id or int(time.time() * 1000))
    task = asyncio.create_task(coro)
    channel_jobs[key] = task
    # Auto-clean on completion
    def _cleanup(_):
        channel_jobs.pop(key, None)
    task.add_done_callback(_cleanup)

@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    # Honeypot check runs BEFORE any allowlist or routing logic  (ADDED)
    try:
        if await honeypot_guard(message):
            return  # action taken; stop further handling
    except Exception:
        logging.exception("honeypot_guard error")

    try:
        ch_id = channel_key(message)
        logging.info("ROUTER in=%r dm=%s ch_id=%s author=%s", message.content, isinstance(message.channel, discord.DMChannel), ch_id, message.author)

        # Gate by allowlist (DMs always allowed; threads inherit parent)
        if not is_allowed(message):
            logging.info("Gate: message blocked by allowlist in #%s (%s)", getattr(message.channel, 'name', '?'), getattr(message.channel, 'id', '?'))
            return

        raw_text = message.content or ""
        text_for_logic = raw_text  # we'll possibly sanitize below

        # OnlyWhenCalled: require name/mention; strip name case-insensitively before LLM
        if config.get("OnlyWhenCalled") and not isinstance(message.channel, discord.DMChannel):
            mentioned = (config.get("Name", "").lower() in raw_text.lower()) or (bot.user in message.mentions)
            if not mentioned:
                logging.info("Gate: OnlyWhenCalled active and bot not mentioned")
                return
            bot_name = re.escape(config.get("Name", ""))
            text_for_logic = re.sub(bot_name, "", raw_text, flags=re.IGNORECASE).strip()

        # Image follow-up takes precedence
        if should_route_to_image_followup(message):
            logging.info("ROUTER -> image_followup")
            start_job(ch_id, "image", handle_image_message(message, text_override=text_for_logic), message.id)
            return

        # New image request
        if looks_like_image_request(text_for_logic):
            logging.info("ROUTER -> image_new")
            start_job(ch_id, "image", handle_image_message(message, text_override=text_for_logic), message.id)
            return

        # Otherwise text
        logging.info("ROUTER -> text")
        start_job(ch_id, "text", handle_text_message(message, text_override=text_for_logic), message.id)
    except Exception:
        logging.exception("on_message router failure")

# =============================================================================
# Run
# =============================================================================
bot.run(config["DiscordToken"])
