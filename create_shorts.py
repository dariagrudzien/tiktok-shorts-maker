import argparse, json, os, re, shutil, subprocess, sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    import numpy as np


def run(cmd, check=True, capture_output=True, text=True):
    print("➤", " ".join(cmd))
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text)


def which_or_die(bin_name: str):
    if shutil.which(bin_name) is None:
        print(f"ERROR: '{bin_name}' not found on PATH. Please install it.")
        sys.exit(1)


def ffprobe_stream_dims(input_path: str):
    which_or_die("ffprobe")
    cmd_dims = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        input_path,
    ]
    proc_dims = run(cmd_dims)
    dims_line = proc_dims.stdout.strip()
    if not dims_line:
        raise RuntimeError("ffprobe failed to read video stream dimensions.")
    w_str, h_str = dims_line.split("x")
    w, h = int(w_str), int(h_str)
    cmd_dur = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        input_path,
    ]
    proc_dur = run(cmd_dur)
    try:
        dur = float(proc_dur.stdout.strip())
    except:
        dur = 0.0
    return w, h, dur


def srt_timestamp(t: float) -> str:
    if t < 0:
        t = 0
    ms = int(round((t - int(t)) * 1000))
    s = int(t) % 60
    m = (int(t) // 60) % 60
    h = int(t) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ass_timestamp(t: float) -> str:
    if t < 0:
        t = 0
    cs = int(round(t * 100))  # centiseconds
    h = cs // 360000
    cs %= 360000
    m = cs // 6000
    cs %= 6000
    s = cs // 100
    cs = cs % 100
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def sanitize_for_ffmpeg_filter(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def ass_color_rgba(r: int, g: int, b: int, opacity: float) -> str:
    """
    ASS color format is &HAABBGGRR (AA=transparency: 00=opaque, FF=transparent)
    """
    opacity = max(0.0, min(1.0, opacity))
    aa = int(round((1.0 - opacity) * 255))
    return f"&H{aa:02X}{b:02X}{g:02X}{r:02X}"


def hex_to_rgb(s: str) -> Tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) != 6 or any(c not in "0123456789aAbBcCdDeEfF" for c in s):
        raise ValueError(f"Invalid hex color: {s}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return r, g, b


def transcribe_with_faster_whisper(input_path: str, model_size: str, device: str):
    from faster_whisper import WhisperModel

    compute_type = "int8" if device == "cpu" else "float16"
    if device == "auto":
        try:
            model = WhisperModel(model_size, device="cuda", compute_type="float16")
            used = "cuda"
        except Exception:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            used = "cpu"
    else:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        used = device
    print(f"[Transcribe] faster-whisper on {used} model={model_size}")
    segments_out = []
    seg_iter, info = model.transcribe(
        input_path,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    for seg in seg_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {"start": float(w.start), "end": float(w.end), "word": w.word}
                )
        segments_out.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "words": words,
            }
        )
    return segments_out


def transcribe_with_openai_whisper(input_path: str, model_size: str, device: str):
    import whisper, torch

    device_used = (
        "cuda"
        if (device == "auto" and torch.cuda.is_available())
        else (device if device != "auto" else "cpu")
    )
    print(f"[Transcribe] openai-whisper on {device_used} model={model_size}")
    model = whisper.load_model(model_size, device=device_used)
    result = model.transcribe(input_path, verbose=False, word_timestamps=True)
    segments_out = []
    for seg in result.get("segments", []):
        words = []
        if "words" in seg and seg["words"]:
            for w in seg["words"]:
                wtxt = w.get("word") or w.get("text") or ""
                words.append(
                    {"start": float(w["start"]), "end": float(w["end"]), "word": wtxt}
                )
        segments_out.append(
            {
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg.get("text", "").strip(),
                "words": words,
            }
        )
    return segments_out


def transcribe(input_path: str, model_size: str, prefer: str, device: str):
    if prefer == "faster":
        try:
            return transcribe_with_faster_whisper(input_path, model_size, device)
        except ImportError:
            print("faster-whisper not installed; trying openai-whisper.")
            return transcribe_with_openai_whisper(input_path, model_size, device)
    else:
        try:
            return transcribe_with_openai_whisper(input_path, model_size, device)
        except ImportError:
            print("openai-whisper not installed; trying faster-whisper.")
            return transcribe_with_faster_whisper(input_path, model_size, device)


HOOK_WORDS = {
    "why",
    "how",
    "what",
    "secret",
    "mistake",
    "truth",
    "tip",
    "tips",
    "trick",
    "tricks",
    "hack",
    "hacks",
    "warning",
    "beware",
    "avoid",
    "never",
    "always",
    "learned",
    "lesson",
    "surprising",
    "nobody",
    "everyone",
    "best",
    "worst",
    "crazy",
    "insane",
    "million",
    "billion",
    "fast",
    "faster",
    "simple",
    "easy",
    "quick",
    "powerful",
    "?",
    "!",
}
FILLERS = {
    "um",
    "uh",
    "like",
    "you know",
    "sort of",
    "kind of",
    "actually",
    "basically",
}


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float
    words_count: int
    wps: float


def segment_words(segments: List[Dict]) -> List[Dict]:
    out = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            text = seg.get("text", "").strip()
            tokens = [t for t in re.findall(r"\w+|\S", text) if t.strip()]
            dur = max(0.001, seg["end"] - seg["start"])
            if tokens:
                step = dur / len(tokens)
                words = [
                    {
                        "start": seg["start"] + i * step,
                        "end": seg["start"] + (i + 1) * step,
                        "word": tok,
                    }
                    for i, tok in enumerate(tokens)
                ]
        out.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg.get("text", "").strip(),
                "words": words,
            }
        )
    return out


def extract_sentences(segments: List[Dict]) -> List[Dict]:
    sentences = []
    for seg in segment_words(segments):
        words = seg["words"]
        if not words:
            if seg["text"].strip():
                sentences.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                        "words": [],
                    }
                )
            continue
        cur = []
        sent_start = None
        for w in words:
            if sent_start is None:
                sent_start = w["start"]
            cur.append(w)
            token = (w["word"] or "").strip()
            if token.endswith((".", "!", "?")):
                txt = re.sub(r"\s+", " ", " ".join(x["word"] for x in cur)).strip()
                sentences.append(
                    {
                        "start": sent_start,
                        "end": cur[-1]["end"],
                        "text": txt,
                        "words": list(cur),
                    }
                )
                cur = []
                sent_start = None
        if cur:
            txt = re.sub(r"\s+", " ", " ".join(x["word"] for x in cur)).strip()
            sentences.append(
                {
                    "start": sent_start if sent_start is not None else seg["start"],
                    "end": cur[-1]["end"],
                    "text": txt,
                    "words": list(cur),
                }
            )
    sentences = [s for s in sentences if s["end"] > s["start"] and s["text"]]
    sentences.sort(key=lambda s: s["start"])
    return sentences


def score_text(text: str, duration: float) -> float:
    t = text.lower()
    hook_hits = sum(1 for hw in HOOK_WORDS if hw in t)
    filler_hits = sum(t.count(f) for f in FILLERS)
    nwords = len(re.findall(r"\w+", t))
    wps = nwords / max(1e-6, duration)
    density = min(1.5, wps / 3.0)
    punct_bonus = 0.4 if t.endswith((".", "!", "?")) else 0.0
    num_bonus = 0.2 if re.search(r"\d", t) else 0.0
    person_bonus = 0.2 if re.search(r"\b(you|your|i|we)\b", t) else 0.0
    return (
        0.9 * hook_hits
        + 1.2 * density
        + punct_bonus
        + num_bonus
        + person_bonus
        - 0.3 * filler_hits
    )


def get_model():
    if SentenceTransformer is None:
        print(
            "[Hotspots] sentence-transformers not installed; semantic scoring disabled."
        )
        return None
    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print("[Hotspots] Failed to load embedding model:", e)
        return None


def embed_texts(model, texts: List[str]):
    if model is None:
        return np.zeros((len(texts), 384), dtype=float)
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs


def mmr_selection(scores, embeds, k: int, lambda_div: float = 0.7):
    n = scores.shape[0]
    if n == 0:
        return []
    selected = []
    candidates = list(range(n))
    first = int(scores.argmax())
    selected.append(first)
    candidates.remove(first)
    while len(selected) < min(k, n) and candidates:
        best_i = None
        best_val = -1e9
        for i in candidates:
            sim_to_sel = 0.0
            if selected:
                sim_to_sel = max(float(embeds[i] @ embeds[j]) for j in selected)
            val = lambda_div * float(scores[i]) - (1.0 - lambda_div) * sim_to_sel
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        candidates.remove(best_i)
    return selected


def build_sentence_candidates(
    segments: List[Dict], min_sec: int, max_sec: int
) -> List["Candidate"]:
    sents = extract_sentences(segments)
    if not sents:
        return []
    raw = []
    n = len(sents)
    for i in range(n):
        start_t = sents[i]["start"]
        words_col = []
        text_parts = []
        end_t = start_t
        for j in range(i, n):
            s = sents[j]
            if j > i and s["start"] < end_t - 1e-3:
                continue
            words_col.extend(s.get("words", []))
            text_parts.append(s["text"])
            end_t = s["end"]
            dur = end_t - start_t
            if dur < min_sec:
                continue
            if dur > max_sec:
                break
            text = " ".join(text_parts).strip()
            if text.lower().startswith(("and ", "but ", "so ", "then ")):
                continue
            sc = score_text(text, dur)
            raw.append((start_t, end_t, text, words_col, sc))
    if not raw:
        return []
    texts = [r[2] for r in raw]
    model = get_model()
    embeds = embed_texts(model, texts)
    scores = np.array([r[4] for r in raw], dtype=float)
    idxs = mmr_selection(scores, embeds, k=min(50, len(raw)), lambda_div=0.7)
    filtered = [raw[i] for i in idxs]
    out = []
    for st, en, txt, words_col, sc in filtered:
        dur = en - st
        wps = (len(words_col) if words_col else len(re.findall(r"\w+", txt))) / max(
            1e-6, dur
        )
        out.append(
            Candidate(
                start=st,
                end=en,
                text=txt,
                score=sc,
                words_count=len(words_col),
                wps=wps,
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def build_fallback_candidates(
    segments: List[Dict], min_sec: int, max_sec: int, stride_sec: int = 5
) -> List["Candidate"]:
    segs = segment_words(segments)
    if not segs:
        return []
    t0 = segs[0]["start"]
    t1 = segs[-1]["end"]
    words_all = []
    for s in segs:
        words_all.extend(s["words"])
    out = []
    cur = t0
    while cur < t1 - min_sec:
        win_end = min(cur + max_sec, t1)
        wwin = [w for w in words_all if cur <= w["start"] < win_end]
        if len(wwin) >= max(6, int(min_sec * 1.2)):
            text = " ".join(w["word"] for w in wwin)
            t_lower = text.lower()
            hook_hits = sum(1 for hw in HOOK_WORDS if hw in t_lower)
            nwords = len(re.findall(r"\w+", t_lower))
            dur = win_end - cur
            wps = nwords / max(1e-6, dur)
            density = min(1.5, wps / 3.0)
            score = 0.8 * hook_hits + 1.1 * density
            out.append(
                Candidate(
                    start=cur,
                    end=win_end,
                    text=text,
                    score=score,
                    words_count=nwords,
                    wps=wps,
                )
            )
        cur += stride_sec
    out.sort(key=lambda c: c.score, reverse=True)
    return out


def pick_top_n(
    candidates: List["Candidate"], n: int, min_gap: float = 1.0
) -> List["Candidate"]:
    selected = []
    used = []
    for c in sorted(candidates, key=lambda x: x.score, reverse=True):
        ok = True
        for a, b in used:
            if not (c.end + min_gap <= a or c.start - min_gap >= b):
                ok = False
                break
        if ok:
            selected.append(c)
            used.append((c.start, c.end))
        if len(selected) >= n:
            break
    selected.sort(key=lambda x: x.start)
    return selected


def wrap_lines_2(text: str, width: int = 42) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if width is None or width <= 0:
        return t  # let renderer wrap
    if len(t) <= width:
        return t
    half = len(t) // 2
    left = t.rfind(" ", 0, half)
    if left == -1:
        left = t.find(" ", half)
    if left == -1:
        return t
    first = t[:left].strip()
    rest = t[left + 1 :].strip()
    if len(rest) > width:
        half2 = len(rest) // 2
        left2 = rest.rfind(" ", 0, half2)
        if left2 != -1:
            return (
                first + "\n" + rest[:left2].strip() + "\n" + rest[left2 + 1 :].strip()
            )
    return first + "\n" + rest


def enforce_min_gap(events: List[List[float]], min_gap: float) -> List[List[float]]:
    """
    events: list of [start, end, text] (in seconds relative to clip start)
    Ensures each event ends at least min_gap before the next starts.
    """
    if not events:
        return events
    events.sort(key=lambda x: x[0])
    for i in range(1, len(events)):
        prev = events[i - 1]
        cur = events[i]
        if prev[1] > cur[0] - min_gap:
            prev[1] = max(prev[0] + 0.12, cur[0] - min_gap)  # keep readable duration
    # Drop any with inverted or too-short duration
    out = []
    for st, en, txt in events:
        if en - st >= 0.10:
            out.append([st, en, txt])
    return out


def sentences_to_ass_bottom(
    sentences: List[Dict],
    clip_start: float,
    clip_end: float,
    target_w: int,
    target_h: int,
    style: str = "box",
    box_opacity: float = 0.75,
    margin_v: int = 90,
    margin_h: int = 150,
    font: str = "Arial",
    fontsize: int = 96,
    wrap_width: int = 32,
    chunk_gap: float = 0.10,
    colors: Dict[str, str] = None,
) -> str:
    """Bottom-centered captions with non-overlap and theming."""
    if colors is None:
        colors = {
            "primary": "FFFFFF",
            "accent": "FFD400",
            "outline": "000000",
            "box": "000000",
        }

    pad_pre = 0.06
    pad_post = 0.12
    clip_len = max(0.1, clip_end - clip_start)

    entries = []
    for s in sentences:
        if s["end"] < clip_start or s["start"] > clip_end:
            continue
        words = s.get("words") or []
        if words:
            sel = [
                w
                for w in words
                if not (w["end"] <= clip_start or w["start"] >= clip_end)
            ]
            if not sel:
                continue
            st = max(clip_start, sel[0]["start"])
            en = min(clip_end, sel[-1]["end"])
            text = " ".join(w["word"] for w in sel).strip()
        else:
            st = max(clip_start, s["start"])
            en = min(clip_end, s["end"])
            text = s["text"].strip()
        st_rel = max(0.0, (st - clip_start) - pad_pre)
        en_rel = min(clip_len, (en - clip_start) + pad_post)
        if en_rel - st_rel < 0.20 or not text:
            continue
        entries.append([st_rel, en_rel, wrap_lines_2(text, width=wrap_width)])

    entries = enforce_min_gap(entries, min_gap=chunk_gap)

    primary = ass_color_rgba(*hex_to_rgb(colors["primary"]), 1.0)
    outline = ass_color_rgba(*hex_to_rgb(colors["outline"]), 1.0)
    back_box = ass_color_rgba(*hex_to_rgb(colors["box"]), box_opacity)

    if style == "box":
        border_style = 3
        outline_w = 1
        shadow = 0
        back_col = back_box
    else:
        border_style = 1
        outline_w = 4
        shadow = 2
        back_col = ass_color_rgba(0, 0, 0, 0.5)

    header = f"""[Script Info]
; Script generated by create_shorts.py
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes
WrapStyle: 2
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},{primary},{primary},{outline},{back_col},0,0,0,0,100,100,0,0,{border_style},{outline_w},{shadow},2,{margin_h},{margin_h},{margin_v},0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for st, en, txt in entries:
        safe_txt = txt.replace("\n", r"\N").replace("{", r"\{").replace("}", r"\}")
        lines.append(
            f"Dialogue: 0,{ass_timestamp(st)},{ass_timestamp(en)},Default,,0,0,0,,{safe_txt}"
        )
    return header + "\n".join(lines) + "\n"


def sentences_to_ass_tiktok(
    sentences: List[Dict],
    clip_start: float,
    clip_end: float,
    target_w: int,
    target_h: int,
    font: str = "Arial",
    fontsize: int = 108,
    margin_h: int = 200,
    center_offset: int = 0,
    box_opacity: float = 0.85,
    chunk_max_words: int = 3,
    chunk_target_sec: float = 0.8,
    chunk_gap: float = 0.10,
    colors: Dict[str, str] = None,
) -> str:
    """
    TikTok-style center captions:
    - 1–3 word chunks with karaoke highlight (\k)
    - Non-overlapping: enforce minimum gap between chunks
    - Theming via colors dict (primary/accent/outline/box in hex)
    """
    if colors is None:
        colors = {
            "primary": "FFFFFF",
            "accent": "FFD400",
            "outline": "000000",
            "box": "000000",
        }

    clip_len = max(0.1, clip_end - clip_start)

    # Flatten words in clip window
    words_all = []
    for s in sentences:
        ws = s.get("words") or []
        for w in ws:
            if w["end"] <= clip_start or w["start"] >= clip_end:
                continue
            start = max(clip_start, float(w["start"]))
            end = min(clip_end, float(w["end"]))
            if end - start <= 0.02:
                continue
            token = (w["word"] or "").strip()
            if not token:
                continue
            words_all.append({"start": start, "end": end, "word": token})
    words_all.sort(key=lambda x: x["start"])

    # Style colors
    primary = ass_color_rgba(*hex_to_rgb(colors["primary"]), 1.0)  # base text
    karaoke = ass_color_rgba(*hex_to_rgb(colors["accent"]), 1.0)  # highlight
    outline = ass_color_rgba(*hex_to_rgb(colors["outline"]), 1.0)  # outline
    back_box = ass_color_rgba(*hex_to_rgb(colors["box"]), box_opacity)  # background box

    header = f"""[Script Info]
; Script generated by create_shorts.py
ScriptType: v4.00+
PlayResX: {target_w}
PlayResY: {target_h}
ScaledBorderAndShadow: yes
WrapStyle: 2
Collisions: Normal

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,{font},{fontsize},{primary},{karaoke},{outline},{back_box},0,0,0,0,100,100,0,0,3,1,0,5,{margin_h},{margin_h},0,0

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    if not words_all:
        # empty stub keeps renderer stable
        stub = f"Dialogue: 0,{ass_timestamp(0)},{ass_timestamp(min(clip_len,0.8))},TikTok,,0,0,0,,{{\\an5\\pos(540,{960+center_offset})}}"
        return header + stub + "\n"

    # build chunks by pauses/punctuation/limits
    chunks = []
    cur = []
    cur_start = None
    last_end = None
    for w in words_all:
        if cur_start is None:
            cur_start = w["start"]
            last_end = w["end"]
            cur = [w]
            continue
        pause = w["start"] - last_end
        punct_break = cur and (cur[-1]["word"].endswith((".", "!", "?", ",")))
        dur = last_end - cur_start
        if (
            (pause > 0.35)
            or punct_break
            or (len(cur) >= chunk_max_words)
            or (dur >= chunk_target_sec)
        ):
            chunks.append({"start": cur_start, "end": last_end, "words": cur})
            cur_start = w["start"]
            cur = [w]
        else:
            cur.append(w)
        last_end = w["end"]
    if cur:
        chunks.append({"start": cur_start, "end": last_end, "words": cur})

    # convert chunks to events with karaoke and enforce min gap
    events = []  # [st, en, txt]
    for ch in chunks:
        st_rel = max(0.0, ch["start"] - clip_start - 0.02)
        en_rel = min(clip_len, ch["end"] - clip_start + 0.08)
        if en_rel - st_rel < 0.12:
            continue
        parts = [
            r"{\an5\pos(540," + str(960 + center_offset) + r")}"
        ]  # center position
        for w in ch["words"]:
            dur_cs = int(round((w["end"] - w["start"]) * 100))
            if dur_cs <= 0:
                dur_cs = 5
            token = (w["word"] or "").strip()
            token = token.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            parts.append(r"{\k" + str(dur_cs) + "}" + token + " ")
        txt = "".join(parts).strip()
        events.append([st_rel, en_rel, txt])

    events = enforce_min_gap(events, min_gap=chunk_gap)

    lines = []
    for st, en, txt in events:
        lines.append(
            f"Dialogue: 0,{ass_timestamp(st)},{ass_timestamp(en)},TikTok,,0,0,0,,{txt}"
        )

    return header + "\n".join(lines) + ("\n" if lines else "")


def detect_face_center(
    input_path: str, start: float, end: float, sample_count: int = 12
):
    try:
        import cv2, mediapipe as mp
    except ImportError:
        print("[Face] mediapipe/opencv not installed; using center crop.")
        return None
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("[Face] Failed to open video.")
        return None
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mp_fd = mp.solutions.face_detection
    fd = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.4)
    centers = []
    import numpy as _np

    times = _np.linspace(
        start, end if end > start else start + 1.0, num=min(sample_count, 20)
    )
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = fd.process(rgb)
        if res.detections:
            det = max(res.detections, key=lambda d: d.score[0] if d.score else 0)
            bb = det.location_data.relative_bounding_box
            cx = bb.xmin + bb.width * 0.5
            cy = bb.ymin + bb.height * 0.4
            centers.append((cx, cy))
    cap.release()
    if not centers:
        print("[Face] No face detected; using center crop.")
        return None
    cx = float(np.median([c[0] for c in centers]))
    cy = float(np.median([c[1] for c in centers]))
    return cx, cy, frame_w, frame_h


def compute_vertical_crop(
    face_center, frame_w: int, frame_h: int, aspect: float = 9 / 16
):
    crop_w_land = int(round(frame_h * aspect))
    crop_h_port = int(round(frame_w / aspect))
    if frame_w >= frame_h:
        crop_w = min(frame_w, crop_w_land)
        crop_h = frame_h
        if face_center:
            cx, cy, fw, fh = face_center
            x_center = int(round(cx * fw))
            x0 = max(0, min(frame_w - crop_w, x_center - crop_w // 2))
            y0 = 0
        else:
            x0 = (frame_w - crop_w) // 2
            y0 = 0
    else:
        crop_w = frame_w
        crop_h = min(frame_h, crop_h_port)
        if face_center:
            cx, cy, fw, fh = face_center
            y_center = int(round(cy * fh))
            y0 = max(0, min(frame_h - crop_h, y_center - crop_h // 2))
            x0 = 0
        else:
            x0 = 0
            y0 = (frame_h - crop_h) // 2
    return crop_w, crop_h, x0, y0


def export_clip(
    input_path: str,
    out_path: str,
    start: float,
    end: float,
    crop_rect,
    ass_path: str,
    target_w: int = 1080,
    target_h: int = 1920,
    crf: int = 22,
    preset: str = "veryfast",
):
    which_or_die("ffmpeg")
    if not os.path.exists(ass_path):
        raise FileNotFoundError(f"ASS not found: {ass_path}")
    duration = max(0.1, end - start)
    crop_w, crop_h, crop_x, crop_y = crop_rect

    vf = f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y},scale={target_w}:{target_h},ass='{sanitize_for_ffmpeg_filter(ass_path)}'"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        input_path,
        "-t",
        f"{duration:.3f}",
        "-vf",
        vf,
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ]
    print("➤", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="Auto shorts with TikTok-style or bottom captions."
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-shorts", type=int, default=10)
    ap.add_argument("--min-sec", type=int, default=15)
    ap.add_argument("--max-sec", type=int, default=58)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--prefer", choices=["faster", "openai"], default="faster")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # Caption controls (shared)
    ap.add_argument("--caption-preset", choices=["bottom", "tiktok"], default="bottom")
    ap.add_argument("--box-opacity", type=float, default=0.8)
    ap.add_argument("--margin-h", type=int, default=180)
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--fontsize", type=int, default=96)
    ap.add_argument(
        "--chunk-gap",
        type=float,
        default=0.10,
        help="Minimum gap (seconds) between consecutive caption events",
    )

    # Bottom-preset specifics
    ap.add_argument("--caption-style", choices=["box", "outline"], default="box")
    ap.add_argument("--margin-v", type=int, default=90)
    ap.add_argument("--wrap-width", type=int, default=32)

    # TikTok-preset specifics
    ap.add_argument(
        "--center-offset",
        type=int,
        default=0,
        help="Vertical offset from center in pixels (+down, -up)",
    )
    ap.add_argument(
        "--chunk-words",
        type=int,
        default=3,
        help="Max words per chunk for TikTok captions",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=0.8,
        help="Target seconds per chunk for TikTok captions",
    )

    # Theme / colors
    ap.add_argument("--theme", choices=["default", "white-blue"], default="default")
    ap.add_argument(
        "--color-primary", default=None, help="Base text color hex, e.g. FFFFFF"
    )
    ap.add_argument(
        "--color-accent",
        default=None,
        help="Highlight color hex, e.g. FFD400 or 0A2A6F",
    )
    ap.add_argument(
        "--color-outline", default=None, help="Outline color hex, e.g. 000000"
    )
    ap.add_argument(
        "--box-tint", default=None, help="Box tint hex for background, e.g. 081A40"
    )

    args = ap.parse_args()

    # theme resolution
    colors = {
        "primary": "FFFFFF",
        "accent": "FFD400",
        "outline": "000000",
        "box": "000000",
    }  # default (white/yellow)
    if args.theme == "white-blue":
        colors = {
            "primary": "FFFFFF",
            "accent": "0A2A6F",
            "outline": "000000",
            "box": "081A40",
        }
    # manual overrides
    if args.color_primary:
        colors["primary"] = args.color_primary
    if args.color_accent:
        colors["accent"] = args.color_accent
    if args.color_outline:
        colors["outline"] = args.color_outline
    if args.box_tint:
        colors["box"] = args.box_tint

    input_path = args.input
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    w, h, dur = ffprobe_stream_dims(input_path)
    print(f"[Probe] {w}x{h}, duration {dur:.1f}s")

    try:
        segments = transcribe(input_path, args.model, args.prefer, args.device)
    except Exception as e:
        print("Transcription failed:", e)
        sys.exit(1)
    if not segments:
        print("No transcription segments produced.")
        sys.exit(1)
    with open(
        os.path.join(outdir, "transcript_segments.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    # candidates
    cands = build_sentence_candidates(segments, args.min_sec, args.max_sec)
    if len(cands) < args.num_shorts:
        print(
            f"[Info] Only {len(cands)} sentence-level candidates; adding fallback windows…"
        )
        fb = build_fallback_candidates(
            segments, args.min_sec, args.max_sec, stride_sec=max(3, args.min_sec // 2)
        )
        cands = (cands or []) + fb

    if not cands:
        print("No candidates found. Try adjusting --min-sec/--max-sec.")
        sys.exit(1)
    picks = pick_top_n(cands, args.num_shorts, min_gap=args.min_gap)
    if not picks:
        print("No non-overlapping picks available.")
        sys.exit(1)

    sentences_all = extract_sentences(segments)
    manifest = {
        "input": input_path,
        "width": w,
        "height": h,
        "duration": dur,
        "clips": [],
    }

    for i, c in enumerate(picks, start=1):
        clip_start, clip_end = c.start, c.end
        face_center = detect_face_center(
            input_path, clip_start, clip_end, sample_count=14
        )
        crop = (
            compute_vertical_crop(face_center, w, h)
            if face_center
            else compute_vertical_crop(None, w, h)
        )

        if args.caption_preset == "tiktok":
            ass_text = sentences_to_ass_tiktok(
                sentences_all,
                clip_start,
                clip_end,
                target_w=1080,
                target_h=1920,
                font=args.font,
                fontsize=args.fontsize,
                margin_h=args.margin_h,
                center_offset=args.center_offset,
                box_opacity=args.box_opacity,
                chunk_max_words=args.chunk_words,
                chunk_target_sec=args.chunk_sec,
                chunk_gap=args.chunk_gap,
                colors=colors,
            )
            style_ref = "TikTok"
        else:
            ass_text = sentences_to_ass_bottom(
                sentences_all,
                clip_start,
                clip_end,
                target_w=1080,
                target_h=1920,
                style=args.caption_style,
                box_opacity=args.box_opacity,
                margin_v=args.margin_v,
                margin_h=args.margin_h,
                font=args.font,
                fontsize=args.fontsize,
                wrap_width=args.wrap_width,
                chunk_gap=args.chunk_gap,
                colors=colors,
            )
            style_ref = "Default"

        ass_path = os.path.join(outdir, f"short_{i:02d}.ass")
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_text)

        out_mp4 = os.path.join(outdir, f"short_{i:02d}.mp4")
        try:
            export_clip(
                input_path,
                out_mp4,
                clip_start,
                clip_end,
                crop_rect=crop,
                ass_path=ass_path,
            )
        except subprocess.CalledProcessError as e:
            print(f"[Export] Clip {i} failed: {e}")
            continue

        manifest["clips"].append(
            {
                "index": i,
                "start": clip_start,
                "end": clip_end,
                "score": c.score,
                "words_count": c.words_count,
                "wps": c.wps,
                "ass": os.path.abspath(ass_path),
                "mp4": os.path.abspath(out_mp4),
                "style": style_ref,
                "crop": {"w": crop[0], "h": crop[1], "x": crop[2], "y": crop[3]},
            }
        )
        print(
            f"[Done] short_{i:02d}.mp4  {srt_timestamp(clip_start)} → {srt_timestamp(clip_end)}  ({style_ref})"
        )

    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("All done.")


if __name__ == "__main__":
    main()
