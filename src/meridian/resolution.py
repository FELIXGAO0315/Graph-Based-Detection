"""Normalisation and entity resolution.

Every match keeps its method and a confidence. Only unambiguous matches are
accepted; ambiguous ones are kept as `candidate` rows and never merged.
"""
from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd
from unidecode import unidecode

from .data import Bundle

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
# Multi-token legal forms first so "co ltd" is removed as one unit.
LEGAL_FORMS = [
    "sp z o o", "s a r l", "s de r l", "s a de c v", "pte ltd", "pty ltd", "sdn bhd", "co ltd", "company limited",
    "s p a", "s r l", "s a s", "s a", "b v", "n v", "d o o", "a s", "o o o",
    "ltd", "limited", "inc", "incorporated", "corp", "corporation", "llc", "llp", "lp", "plc", "gmbh", "ag",
    "sa", "sas", "sarl", "srl", "bv", "nv", "spa", "oy", "ab", "as", "kk", "ltda", "sl", "kft", "zrt", "ooo",
    "ao", "doo", "se", "co", "company", "pte", "pty", "bhd", "tbk", "pt", "cv", "sc", "sae", "jsc", "pjsc",
    "llc", "fzco", "fze", "wll", "ou", "uab", "sia", "eood", "ad", "dd", "sro", "as", "aps", "asa",
]
_LEGAL_TOKENS = sorted({tuple(f.split()) for f in LEGAL_FORMS}, key=len, reverse=True)
# Duplicate-record markers seen in the vendor master.
_SUFFIX_RE = re.compile(r"\((?:branch|use this one|old|duplicate|dup|do not use|\d+)\)", re.I)
_PUNCT_RE = re.compile(r"[^0-9a-z]+")
# Honorifics on individual names.
_TITLES = {"dr", "drg", "prof", "mr", "mrs", "ms", "mx", "sr", "sra", "dra", "ir", "h", "hj", "dott", "ing"}


def translit(s: str) -> str:
    return unidecode(s or "")


def name_tokens(name: str) -> list[str]:
    s = _SUFFIX_RE.sub(" ", translit(name).lower())
    return _PUNCT_RE.sub(" ", s).split()


def strip_legal_form(tokens: list[str]) -> tuple[list[str], str]:
    """Remove trailing legal forms; return (core tokens, removed legal form)."""
    removed = []
    changed = True
    while changed and tokens:
        changed = False
        for form in _LEGAL_TOKENS:
            n = len(form)
            if len(tokens) > n and tuple(tokens[-n:]) == form:
                removed = list(form) + removed
                tokens = tokens[:-n]
                changed = True
                break
    return tokens, " ".join(removed)


def name_key(name: str) -> str:
    """Name without punctuation, script, case, legal form or duplicate markers."""
    core, _ = strip_legal_form(name_tokens(name))
    return " ".join(core)


def person_key(name: str) -> str:
    """Order-insensitive person name key without honorifics."""
    toks = [t for t in name_tokens(name) if t not in _TITLES]
    return " ".join(sorted(toks))


# ---------------------------------------------------------------------------
# Addresses and contacts
# ---------------------------------------------------------------------------
_ADDR_ABBREV = {
    "ste": "suite", "st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard",
    "bd": "boulevard", "jl": "jalan", "jln": "jalan", "fl": "floor", "flr": "floor", "apt": "apartment",
    "no": "", "nr": "", "piso": "floor", "etage": "floor", "bldg": "building", "dr": "drive",
    "ln": "lane", "hwy": "highway", "sq": "square", "pl": "place", "ct": "court", "cres": "crescent",
    "ul": "ulica", "str": "strasse", "c": "calle",
}


def address_tokens(raw: str) -> list[str]:
    s = _PUNCT_RE.sub(" ", translit(raw).lower())
    out = []
    for tok in s.split():
        tok = _ADDR_ABBREV.get(tok, tok)
        if tok:
            out.append(tok)
    return out


def address_key(raw: str, country: str | None = None) -> str:
    """Order-insensitive key: sorted tokens after expanding abbreviations.

    Unit numbers stay in the key (keep door, building and suite).
    Field order differs across records ("Suite 307, Camino ..." vs
    "CAMINO ... STE 307"), so tokens are sorted.
    """
    toks = sorted(address_tokens(raw))
    return (country or "").upper() + ":" + " ".join(toks)


def phone_key(phone: str | None) -> str | None:
    if not phone or not isinstance(phone, str):
        return None
    main = re.split(r"(?:x|ext\.?)\s*\d+$", phone.strip(), flags=re.I)[0]
    digits = re.sub(r"\D", "", main)
    if len(digits) < 7:
        return None
    return digits[-9:]  # national significant part; tolerant of +CC / trunk-0 variants


def normalise_addresses(b: Bundle) -> pd.DataFrame:
    a = b.tp_address_history.copy()
    a["addr_key"] = [address_key(r, c) for r, c in zip(a["raw_address"], a["address_country"])]
    return a


def normalise_contacts(b: Bundle) -> pd.DataFrame:
    tp = b.third_parties[["record_id", "phone", "email_domain"]].copy()
    tp["phone_key"] = tp["phone"].map(phone_key)
    tp["domain_key"] = tp["email_domain"].str.lower().str.strip()
    return tp


# ---------------------------------------------------------------------------
# Vendor <-> registry matching
# ---------------------------------------------------------------------------
def match_registry(b: Bundle) -> pd.DataFrame:
    """Link vendor records to registry R entity statements.

    Order: registry number, then LEI, then transliterated name within the same
    jurisdiction. A name match is accepted only if it is unique on both sides
    within the jurisdiction; otherwise every pair is kept as a candidate.
    Returns one row per (record_id, statement_id) with method, confidence, status.
    """
    tp = b.third_parties[b.third_parties["party_type"] == "organization"]
    re_ = b.reg_entities
    rows = []
    done: set[str] = set()

    by_reg = tp.dropna(subset=["registry_no"]).merge(
        re_.dropna(subset=["registry_no"])[["registry_no", "statement_id"]], on="registry_no")
    for r in by_reg.itertuples():
        rows.append((r.record_id, r.statement_id, "registry_no", 1.0, "accepted"))
        done.add(r.record_id)

    by_lei = tp[~tp["record_id"].isin(done)].dropna(subset=["lei"]).merge(
        re_.dropna(subset=["lei"])[["lei", "statement_id"]], on="lei")
    for r in by_lei.itertuples():
        rows.append((r.record_id, r.statement_id, "lei", 1.0, "accepted"))
        done.add(r.record_id)

    ent_keys = re_.assign(key=re_["name"].map(name_key))
    ent_idx = defaultdict(list)
    for e in ent_keys.itertuples():
        if e.key:
            ent_idx[(e.jurisdiction, e.key)].append(e.statement_id)
    rest = tp[~tp["record_id"].isin(done)].assign(key=lambda d: d["legal_name"].map(name_key))
    vend_idx = defaultdict(list)
    for v in rest.itertuples():
        vend_idx[(v.country, v.key)].append(v.record_id)
    for (ctry, key), recs in vend_idx.items():
        ents = ent_idx.get((ctry, key), [])
        if not ents:
            continue
        # Duplicate vendor records of one entity legitimately share a registry entity.
        accepted = len(ents) == 1
        for rec in recs:
            for sid in ents:
                rows.append((rec, sid, "name_jurisdiction", 0.9 if accepted else 0.5,
                             "accepted" if accepted else "candidate"))
    return pd.DataFrame(rows, columns=["record_id", "statement_id", "method", "confidence", "status"])


def match_persons(b: Bundle) -> pd.DataFrame:
    """Link individual experts to registry natural persons by name and nationality."""
    tp = b.third_parties[b.third_parties["party_type"] == "individual"]
    pers = b.reg_persons.assign(key=b.reg_persons["full_name"].map(person_key))
    idx = defaultdict(list)
    for p in pers.itertuples():
        idx[(p.nationality, p.key)].append(p.statement_id)
    rows = []
    for v in tp.itertuples():
        hits = idx.get((v.country, person_key(v.legal_name)), [])
        for sid in hits:
            rows.append((v.record_id, sid, "person_name_nationality", 0.9 if len(hits) == 1 else 0.5,
                         "accepted" if len(hits) == 1 else "candidate"))
    return pd.DataFrame(rows, columns=["record_id", "statement_id", "method", "confidence", "status"])


# ---------------------------------------------------------------------------
# Duplicate vendor records -> ResolvedEntity
# ---------------------------------------------------------------------------
def resolve_duplicates(b: Bundle, name_merge: str = "corroborated") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group vendor-master records that are the same real third party.

    Merge evidence:
      - same tax_id, registry number or LEI (confidence 1.0)
      - same name key in the same country, for organisations, when no
        identifier conflicts (both present and different) and the records
        also share a normalised address or phone (confidence 0.8)
    Same-name pairs without that corroboration are kept as IDENTITY_CANDIDATE
    and not merged: on the starter data they are mostly sibling companies of
    one declared group, some with different registry numbers.
    `name_merge` sets the strictness for sensitivity runs: "none" (identifiers
    only), "corroborated" (default), "any" (same name alone, unless identifiers conflict).
    Shared address, sponsor or owner alone is never enough.
    Returns (entity map record_id -> entity_id, pair evidence table).
    """
    tp = b.third_parties.copy()
    tp["key"] = tp["legal_name"].map(name_key)
    tp["phone_key"] = tp["phone"].map(phone_key)
    addrs = normalise_addresses(b).groupby("record_id")["addr_key"].apply(set).to_dict()
    info = tp.set_index("record_id")
    parent = {r: r for r in tp["record_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, c):
        ra, rb = find(a), find(c)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    pairs = []
    for col in ["tax_id", "registry_no", "lei"]:
        for _, g in tp.dropna(subset=[col]).groupby(col)["record_id"]:
            recs = sorted(set(g))
            for other in recs[1:]:
                pairs.append((recs[0], other, col, 1.0, "SAME_AS"))
                union(recs[0], other)

    def conflict(a, c) -> bool:
        for col in ["registry_no", "lei"]:
            va, vc = info.at[a, col], info.at[c, col]
            if isinstance(va, str) and isinstance(vc, str) and va != vc:
                return True
        return False

    orgs = tp[(tp["party_type"] == "organization") & (tp["key"] != "")]
    for _, g in orgs.groupby(["country", "key"])["record_id"]:
        recs = sorted(set(g))
        for i, a in enumerate(recs):
            for c in recs[i + 1:]:
                if find(a) == find(c):
                    continue
                shared_addr = bool(addrs.get(a, set()) & addrs.get(c, set()))
                pa, pc = info.at[a, "phone_key"], info.at[c, "phone_key"]
                shared_phone = isinstance(pa, str) and pa == pc
                if name_merge == "none":
                    ok = False
                elif name_merge == "any":
                    ok = not conflict(a, c)
                else:
                    ok = not conflict(a, c) and (shared_addr or shared_phone)
                if ok:
                    pairs.append((a, c, "name_country+address/phone", 0.8, "SAME_AS"))
                    union(a, c)
                else:
                    why = "identifier conflict" if conflict(a, c) else "no shared address/phone"
                    pairs.append((a, c, f"name_country ({why})", 0.4, "IDENTITY_CANDIDATE"))

    emap = pd.DataFrame({"record_id": tp["record_id"]})
    emap["entity_id"] = emap["record_id"].map(lambda r: "ENT-" + find(r).removeprefix("TP-"))
    emap["n_records"] = emap.groupby("entity_id")["record_id"].transform("size")
    return emap, pd.DataFrame(pairs, columns=["record_a", "record_b", "reason", "confidence", "relation"])
