"""Bakes ipa_everywhere's live lexicon (g2p/ag2p/lexicon/lexicon.db) into the lexicon.bin the G2P loads.
Only lex.k / lex.v are rebuilt; spaCy tables, vocab and NFC sections are carried over from a base lexicon.bin.
python3 tools/bake_lexicon.py BASE_lexicon.bin OUT_lexicon.bin [lexicon.db]"""
import json, sqlite3, struct, sys
from pathlib import Path

DB = Path.home() / "Documents/ipa_everywhere/g2p/ag2p/lexicon/lexicon.db"
STATUS = {"resolved": "r", "ambiguous": "a", "contextual": "c", "weak_strong": "w"}
MAGIC = b"KKG2P\x01\0\0"


def read(path):
    d = Path(path).read_bytes()
    assert d[:8] == MAGIC, "bad magic"
    p = 8
    def u32():
        nonlocal p
        p += 4
        return struct.unpack_from("<I", d, p - 4)[0]
    def s():
        nonlocal p
        n = u32()
        p += n
        return d[p - n:p]
    return {s().decode(): [s() for _ in range(u32())] for _ in range(u32())}


def write(path, secs):
    out = bytearray(MAGIC) + struct.pack("<I", len(secs))
    for name, items in secs.items():
        out += struct.pack("<I", len(name)) + name.encode() + struct.pack("<I", len(items))
        for x in items:
            out += struct.pack("<I", len(x)) + x
    Path(path).write_bytes(bytes(out))


def rows(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conds = {}
    for w, cond, ipa in c.execute("SELECT word, condition, ipa FROM variant WHERE lang='en-US' ORDER BY id"):
        conds.setdefault(w, []).append((cond, ipa))
    for w, status, ipa, variants, source, reviewed in c.execute(
            "SELECT word, status, ipa, variants, source, reviewed FROM lexicon WHERE lang='en-US'"):
        opts = [v["ipa"] for v in json.loads(variants or "[]")]
        default = opts[0] if opts else (conds[w][0][1] if w in conds else "")
        flag = "1" if source == "misaki-dict" and not reviewed else "0"
        cs = "\x1e".join(f"{k}\x1d{v}" for k, v in conds.get(w, []))
        yield w, f"{STATUS[status]}{flag}\x1f{ipa or ''}\x1f{default}\x1f{chr(0x1e).join(opts)}\x1f{cs}"


def main():
    base, out = sys.argv[1], sys.argv[2]
    secs = read(base)
    lex = sorted((w.encode(), v.encode()) for w, v in rows(sys.argv[3] if len(sys.argv) > 3 else DB))
    old = dict(zip(secs["lex.k"], secs["lex.v"]))
    new = dict(lex)
    changed = sum(1 for k, v in new.items() if k in old and old[k] != v)
    print(f"entries {len(old)} -> {len(new)}: added {len(new.keys() - old.keys())} removed {len(old.keys() - new.keys())} changed {changed}")
    secs["lex.k"], secs["lex.v"] = [k for k, _ in lex], [v for _, v in lex]
    write(out, secs)


if __name__ == "__main__":
    main()
