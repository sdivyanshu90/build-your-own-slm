"""A from-scratch, byte-level Byte-Pair-Encoding (BPE) tokenizer.

Design goals
------------
* **Lossless** — operating on UTF-8 *bytes* means any input round-trips
  exactly, including emoji and arbitrary binary; there is no out-of-vocabulary
  failure mode.
* **GPT-2 compatible scheme** — the same byte<->unicode remapping and regex
  pre-tokenization GPT-2 uses, so the implementation is familiar and battle
  tested in spirit.
* **Self-contained** — no external tokenizer dependency; trains, saves, and
  loads from a single JSON file.

The training algorithm uses an *incremental* pair-count index so that growing
the vocabulary to thousands of merges stays fast even on multi-megabyte
corpora (only the words affected by a merge are re-scanned).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

import regex as re

__all__ = ["BPETokenizer"]

# GPT-2 pre-tokenization pattern: keeps contractions, runs of letters, digits,
# punctuation and whitespace as separate "words" so merges never cross these
# natural boundaries.
_GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_FORMAT_VERSION = 1


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """Reversible map from the 256 byte values to printable unicode chars.

    Control and whitespace bytes are shifted into a private unicode range so the
    tokenizer never has to handle invisible/forbidden characters as token text.
    This is the exact mapping introduced by GPT-2.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


def _get_pairs(symbols: list[str]) -> set[tuple[str, str]]:
    """Return the set of adjacent symbol pairs in a word."""
    return set(pairwise(symbols))


class BPETokenizer:
    """Byte-level BPE tokenizer with train / encode / decode / save / load.

    Attributes:
        vocab: Mapping from token text (in byte-unicode space) to integer id.
        merges: Ordered list of learned merges; index encodes merge priority.
        special_tokens: Mapping from special token text to its id.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: dict[str, int] | None = None,
        pattern: str = _GPT2_SPLIT_PATTERN,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or {}
        self.pattern = pattern

        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}
        self._ranks = {pair: i for i, pair in enumerate(merges)}
        self._id_to_token = {i: t for t, i in vocab.items()}
        self._id_to_token.update({i: t for t, i in self.special_tokens.items()})
        self._compiled_pattern = re.compile(self.pattern)
        self._special_pattern = (
            re.compile("(" + "|".join(re.escape(t) for t in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._bpe_cache: dict[str, list[str]] = {}

    # -- construction -------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def token_to_id(self, token: str) -> int | None:
        if token in self.special_tokens:
            return self.special_tokens[token]
        return self.vocab.get(token)

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: list[str] | None = None,
        pattern: str = _GPT2_SPLIT_PATTERN,
    ) -> BPETokenizer:
        """Learn a BPE vocabulary from a corpus.

        Args:
            texts: Iterable of raw text chunks (e.g. lines or documents).
            vocab_size: Target vocabulary size *including* the 256 base byte
                tokens and any special tokens.
            special_tokens: Atomic tokens (e.g. ``"<|endoftext|>"``) that are
                never split and are assigned the final ids.
            pattern: Pre-tokenization regex.

        Returns:
            A trained :class:`BPETokenizer`.
        """
        special_tokens = special_tokens or []
        byte_encoder = _bytes_to_unicode()
        compiled = re.compile(pattern)

        num_merges = vocab_size - 256 - len(special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size={vocab_size} too small for 256 base bytes + "
                f"{len(special_tokens)} special tokens."
            )

        # 1. Pre-tokenize the whole corpus into word-frequency counts. Working
        #    on unique words (not raw token positions) is what keeps training
        #    tractable on real corpora.
        word_freqs: Counter[str] = Counter()
        for text in texts:
            for match in compiled.findall(text):
                token_bytes = match.encode("utf-8")
                word = "".join(byte_encoder[b] for b in token_bytes)
                word_freqs[word] += 1

        # 2. Represent each word as a mutable symbol list and build the
        #    incremental pair-count index.
        words: list[list[str]] = []
        freqs: list[int] = []
        for word, freq in word_freqs.items():
            words.append(list(word))
            freqs.append(freq)

        pair_counts: Counter[tuple[str, str]] = Counter()
        pair_to_words: dict[tuple[str, str], set[int]] = defaultdict(set)
        for idx, symbols in enumerate(words):
            for pair in pairwise(symbols):
                pair_counts[pair] += freqs[idx]
                pair_to_words[pair].add(idx)

        merges: list[tuple[str, str]] = []
        for _ in range(num_merges):
            if not pair_counts:
                break
            # Most frequent pair; deterministic lexicographic tie-break.
            best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
            if pair_counts[best] <= 0:
                break
            merges.append(best)
            a, b = best
            merged = a + b

            for idx in list(pair_to_words.get(best, ())):
                symbols = words[idx]
                freq = freqs[idx]
                # Withdraw this word's current pair contributions.
                for pair in pairwise(symbols):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        pair_counts.pop(pair, None)
                        pair_to_words.pop(pair, None)
                    else:
                        pair_to_words[pair].discard(idx)
                # Apply the merge in-place.
                new_symbols: list[str] = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == a and symbols[j + 1] == b:
                        new_symbols.append(merged)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                words[idx] = new_symbols
                # Re-add contributions for the rewritten word.
                for pair in pairwise(new_symbols):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(idx)

        # 3. Assemble the final vocabulary: 256 base bytes, then merges, then
        #    special tokens (which occupy the highest ids).
        vocab: dict[str, int] = {byte_encoder[b]: b for b in range(256)}
        next_id = 256
        for a, b in merges:
            vocab[a + b] = next_id
            next_id += 1
        specials = {tok: next_id + i for i, tok in enumerate(special_tokens)}

        return cls(vocab=vocab, merges=merges, special_tokens=specials, pattern=pattern)

    # -- encoding -----------------------------------------------------------
    def _bpe(self, word: str) -> list[str]:
        """Apply learned merges to a single byte-unicode word, with caching."""
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        symbols = list(word)
        if len(symbols) < 2:
            self._bpe_cache[word] = symbols
            return symbols

        while True:
            pairs = _get_pairs(symbols)
            candidate = min(pairs, key=lambda p: self._ranks.get(p, float("inf")))
            if candidate not in self._ranks:
                break
            a, b = candidate
            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            if len(symbols) == 1:
                break

        self._bpe_cache[word] = symbols
        return symbols

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, treating any special-token text as ordinary bytes."""
        ids: list[int] = []
        for match in self._compiled_pattern.findall(text):
            token_bytes = match.encode("utf-8")
            word = "".join(self._byte_encoder[b] for b in token_bytes)
            for piece in self._bpe(word):
                ids.append(self.vocab[piece])
        return ids

    def encode(self, text: str, *, allowed_special: bool = True) -> list[int]:
        """Encode text into token ids.

        Args:
            text: Input string.
            allowed_special: When ``True`` (default), recognised special tokens
                in the text are emitted as their atomic id. When ``False`` they
                are encoded as ordinary bytes.
        """
        if not allowed_special or self._special_pattern is None:
            return self.encode_ordinary(text)

        ids: list[int] = []
        for chunk in self._special_pattern.split(text):
            if not chunk:
                continue
            if chunk in self.special_tokens:
                ids.append(self.special_tokens[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode token ids back to text (UTF-8, replacing invalid bytes)."""
        pieces: list[str] = []
        for i in ids:
            token = self._id_to_token.get(int(i))
            if token is None:
                continue
            pieces.append(token)
        text = "".join(pieces)
        # Special tokens are plain text and map through cleanly; byte tokens map
        # back to their original bytes via the byte decoder.
        byte_values = bytearray()
        for ch in text:
            if ch in self._byte_decoder:
                byte_values.append(self._byte_decoder[ch])
            else:
                # Special-token characters that are not in byte space.
                byte_values.extend(ch.encode("utf-8"))
        return byte_values.decode("utf-8", errors="replace")

    def id_to_bytes(self, idx: int) -> bytes:
        """Return the raw bytes a single token id expands to.

        Used for *incremental* decoding during streaming generation, where the
        caller buffers bytes and flushes only complete UTF-8 sequences.
        """
        token = self._id_to_token.get(int(idx))
        if token is None:
            return b""
        if idx in self._id_to_token and token in self.special_tokens:
            return token.encode("utf-8")
        out = bytearray()
        for ch in token:
            if ch in self._byte_decoder:
                out.append(self._byte_decoder[ch])
            else:
                out.extend(ch.encode("utf-8"))
        return bytes(out)

    @property
    def eot_id(self) -> int | None:
        """Id of the end-of-text token if one was configured."""
        return self.special_tokens.get("<|endoftext|>")

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialise the tokenizer to a single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "pattern": self.pattern,
            "vocab": self.vocab,
            "merges": [list(m) for m in self.merges],
            "special_tokens": self.special_tokens,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously written by :meth:`save`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = payload.get("version")
        if version != _FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer format version: {version}")
        return cls(
            vocab={str(k): int(v) for k, v in payload["vocab"].items()},
            merges=[tuple(m) for m in payload["merges"]],
            special_tokens={str(k): int(v) for k, v in payload["special_tokens"].items()},
            pattern=payload["pattern"],
        )
