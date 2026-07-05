"""Join key between the two tennis data sources, which use different name
formats: tennis-data.co.uk (historical results/odds, feeds the Elo engine)
writes "Surname F." (e.g. "Djokovic N."); OddsPapi (live BetMGM odds)
returns "Surname, First" (e.g. "Djokovic, Novak"). Normalizing OddsPapi's
format to match tennis-data.co.uk's is what lets a live fixture be looked up
in the Elo ratings dict.

Known, disclosed limitation: this only handles the common single-first-name
case. Multi-word first names (e.g. "Juan Martin" -> tennis-data.co.uk's
"J.M." double-initial convention) won't match and fall back to BASE_RATING
-- rare enough (a handful of players) that it's called out in the UI rather
than hand-coded around, matching this project's existing policy on small,
un-backtestable edge cases (see e.g. the promotion/relegation gap in the
club-football addendum)."""


def normalize_oddspapi_name(name: str) -> str:
    if "," not in name:
        return name.strip()
    surname, _, first = name.partition(",")
    surname = surname.strip()
    first = first.strip()
    if not first:
        return surname
    return f"{surname} {first[0]}."


def resolve_player(name: str, ratings: dict):
    """Look up a normalized name in the Elo ratings dict, tolerating case
    differences in name particles -- confirmed directly: tennis-data.co.uk
    capitalizes "de"/"van"/"del" etc. ("De Minaur A.", "Van De Zandschulp
    B."), but OddsPapi/BetMGM sends them in natural lowercase ("de Minaur,
    Alex"), so an exact-match lookup silently misses real, well-ranked
    players. Returns (matched_name_or_input, was_matched) -- the caller
    still falls back to BASE_RATING (via engine.get()'s default) when
    was_matched is False, and should disclose that rather than hide it."""
    if name in ratings:
        return name, True
    lower = name.lower()
    for key in ratings:
        if key.lower() == lower:
            return key, True
    return name, False
