"""Token-aware, structure-aware chunking with overlap and citation metadata."""
import re

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:  # offline / encoding download failed -> rough estimate
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3) + 1

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 &/,\-:]{3,70}$")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _heading_of(paragraph: str):
    line = paragraph.strip()
    if "\n" in line:
        return None
    m = _HEADING.match(line)
    if m:
        return m.group(1).strip()
    if _ALLCAPS.match(line):
        return line.title()
    return None


def _split_long(text: str, limit: int):
    """Split an oversized paragraph into sentences, then words if still too long."""
    pieces = []
    for sent in _SENT.split(text):
        if count_tokens(sent) <= limit:
            pieces.append(sent)
            continue
        words, cur = sent.split(), []
        for w in words:
            cur.append(w)
            if count_tokens(" ".join(cur)) >= limit:
                pieces.append(" ".join(cur))
                cur = []
        if cur:
            pieces.append(" ".join(cur))
    return pieces


def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    """Return [{text, position, section, tokens}]. Sliding window over paragraph/sentence units."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    overlap = min(overlap, size // 2)

    units, section = [], "Introduction"
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        h = _heading_of(para)
        if h:
            section = h
        for piece in (_split_long(para, size) if count_tokens(para) > size else [para]):
            units.append({"text": piece, "tokens": count_tokens(piece), "section": section})

    chunks, i = [], 0
    while i < len(units):
        j, total = i, 0
        while j < len(units) and (total + units[j]["tokens"] <= size or j == i):
            total += units[j]["tokens"]
            j += 1
        body = "\n\n".join(u["text"] for u in units[i:j])
        first = next((u for u in units[i:j] if _heading_of(u["text"]) is None), units[i])
        chunks.append({"text": body, "position": len(chunks), "section": first["section"],
                       "tokens": total})
        if j >= len(units):
            break
        # step back to build the overlap, but always make forward progress
        back, acc = j, 0
        while back > i + 1 and acc + units[back - 1]["tokens"] <= overlap:
            back -= 1
            acc += units[back]["tokens"]
        i = max(back, i + 1)
    return chunks
