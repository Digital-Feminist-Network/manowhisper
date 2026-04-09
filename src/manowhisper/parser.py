import os
from pathlib import Path

import webvtt
from alive_progress import alive_bar

try:
    import nltk

    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    import nltk

    nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize

# Maximum character length for a single chunk when no sentence boundary is found.
MAX_CHUNK_CHARS = 500


def extract_show_name(input_path):
    show_name = Path(input_path).parent.name
    return show_name


def _split_caption_text(text: str) -> list[str]:
    """
    Split a single caption block into sentence-sized chunks.

    1. Try NLTK sentence tokenization first.
    2. If a resulting chunk is still too long (no punctuation-based boundary
       was found), split it further into MAX_CHUNK_CHARS-sized pieces so the
       classifier never receives a wall of text.
    """
    raw_sentences = sent_tokenize(text)
    chunks = []
    for s in raw_sentences:
        if len(s) <= MAX_CHUNK_CHARS:
            chunks.append(s)
        else:
            # Hard-split overly long segments on word boundaries.
            words = s.split()
            current = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > MAX_CHUNK_CHARS and current:
                    chunks.append(" ".join(current))
                    current = [word]
                    current_len = len(word)
                else:
                    current.append(word)
                    current_len += len(word) + 1
            if current:
                chunks.append(" ".join(current))
    return chunks


def extract_sentences_timestamps(input_path, window: int = 1):
    """
    Parse WebVTT files to extract sentences, filenames, and timestamps.

    Each caption block is first split into sentence-sized chunks (handling
    transcripts with no punctuation / long runs of text).

    A sliding context window is then applied: for each anchor sentence the
    `window` sentences before and after it (within the same file) are
    prepended/appended so the classifier receives richer context.  The
    returned sentence strings are the *windowed* versions, but the filenames
    and timestamps still correspond to the anchor sentence.

    Parameters
    ----------
    input_path : str | Path
        Path to a single .vtt file or a directory of .vtt files.
    window : int
        Number of sentences to include on each side of the anchor (default 1,
        giving a three-sentence window).
    """
    if os.path.isdir(input_path):
        vtt_files = sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".vtt")
        )
    elif os.path.isfile(input_path) and input_path.endswith(".vtt"):
        vtt_files = [input_path]
    else:
        raise ValueError("Input must be a directory or a .vtt file.")

    all_sentences: list[str] = []
    all_filenames: list[str] = []
    all_timestamps: list[str] = []

    with alive_bar(len(vtt_files), title="Parsing WebVTT files") as bar:
        for filepath in vtt_files:
            filename = os.path.basename(filepath)

            # --- Phase 1: expand every caption into sentence-sized chunks ----
            file_sentences: list[str] = []
            file_timestamps: list[str] = []

            for caption in webvtt.read(filepath):
                text = caption.text.strip().replace("\n", " ")
                chunks = _split_caption_text(text)
                for chunk in chunks:
                    file_sentences.append(chunk)
                    file_timestamps.append(caption.start)

            # --- Phase 2: build windowed context for each anchor sentence ----
            n = len(file_sentences)
            for i, (sentence, ts) in enumerate(zip(file_sentences, file_timestamps)):
                prev_context = file_sentences[max(0, i - window) : i]
                next_context = file_sentences[i + 1 : min(n, i + window + 1)]
                windowed = " ".join(prev_context + [sentence] + next_context)
                all_sentences.append(windowed)
                all_filenames.append(filename)
                all_timestamps.append(ts)

            bar()

    return all_sentences, all_filenames, all_timestamps


def extract_fulltext(input_path):
    """
    Parse WebVTT files to extract full text.
    Handles both a directory of WebVTT files and a single WebVTT file.
    """
    if os.path.isdir(input_path):
        vtt_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith(".vtt")
        ]
    elif os.path.isfile(input_path) and input_path.endswith(".vtt"):
        vtt_files = [input_path]
    else:
        raise ValueError("Input must be a directory or a .vtt file.")

    transcript = []
    for filepath in vtt_files:
        for caption in webvtt.read(filepath):
            transcript.append(caption.text)

    return " ".join(transcript)
