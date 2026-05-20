import math
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

DB_PATH = "ev_scanner_oddsapi_io.db"
BASE_URL = "https://api.odds-api.io/v3"
DEFAULT_SPORTS = ["basketball", "baseball", "american-football", "ice-hockey", "football", "tennis"]
DEFAULT_MARKET_NAMES = ["ML", "Spread", "Totals"]
DEFAULT_TARGET_BOOK = "FanDuel"
DEFAULT_SHARP_BOOKS = ["Pinnacle"]

st.set_page_config(page_title="EV Scanner Pro Max — Odds-API.io", layout="wide")

# --------------------------- Database ---------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            sport TEXT,
            league TEXT,
            event_id TEXT,
            event_name TEXT,
            market TEXT,
            selection TEXT,
            sportsbook TEXT,
            decimal_odds REAL,
            american_odds INTEGER,
            point REAL,
            implied REAL,
            fair_prob REAL,
            fair_odds INTEGER,
            ev_percent REAL,
            edge_percent REAL,
            confidence_score INTEGER
        )
        """
    )
    con.commit()
    con.close()


def save_snapshots(df: pd.DataFrame):
    if df.empty:
        return
    cols = [
        "ts", "sport", "league", "event_id", "event_name", "market", "selection",
        "sportsbook", "decimal_odds", "american_odds", "point", "implied", "fair_prob",
        "fair_odds", "ev_percent", "edge_percent", "confidence_score"
    ]
    con = sqlite3.connect(DB_PATH)
    df[cols].to_sql("odds_snapshots", con, if_exists="append", index=False)
    con.close()


def load_history() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM odds_snapshots ORDER BY ts ASC", con)
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df

# --------------------------- Odds Math ---------------------------
def decimal_to_american(decimal_odds: float) -> Optional[int]:
    if decimal_odds is None or decimal_odds <= 1:
        return None
    if decimal_odds >= 2:
        return int(round((decimal_odds - 1) * 100))
    return int(round(-100 / (decimal_odds - 1)))


def american_to_decimal(odds: int) -> float:
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def american_to_implied(odds: int) -> float:
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)


def decimal_to_implied(decimal_odds: float) -> float:
    return 1 / decimal_odds if decimal_odds and decimal_odds > 1 else 0


def prob_to_american(prob: float) -> Optional[int]:
    if prob <= 0 or prob >= 1:
        return None
    if prob >= 0.5:
        return int(round(-100 * prob / (1 - prob)))
    return int(round(100 * (1 - prob) / prob))


def ev_percent(true_prob: float, american_odds: int) -> float:
    return ((true_prob * american_to_decimal(american_odds)) - 1) * 100


def kelly_fraction(true_prob: float, american_odds: int) -> float:
    dec = american_to_decimal(american_odds)
    b = dec - 1
    q = 1 - true_prob
    k = ((b * true_prob) - q) / b if b else 0
    return max(0, k)


def no_vig_probs(selection_to_american: Dict[str, int]) -> Dict[str, float]:
    implied = {k: american_to_implied(v) for k, v in selection_to_american.items() if v is not None}
    total = sum(implied.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in implied.items()}


def confidence_score(ev: float, books_used: int, sharp_available: bool, market: str, line_move: float = 0.0) -> int:
    score = 0
    score += min(max(ev, 0) * 8, 40)
    score += min(books_used * 4, 24)
    score += 18 if sharp_available else 0
    score += 10 if market in ["ML", "Spread", "Totals"] else 4
    score += min(abs(line_move) * 2, 8)
    return int(round(min(score, 100)))

# --------------------------- API helpers ---------------------------
def api_get(path: str, params: Dict[str, Any], timeout: int = 30) -> Tuple[Any, Optional[str], Dict[str, str]]:
    url = f"{BASE_URL}{path}"
    r = requests.get(url, params=params, timeout=timeout)
    usage = {
        "limit": r.headers.get("x-ratelimit-limit") or r.headers.get("X-RateLimit-Limit") or "?",
        "remaining": r.headers.get("x-ratelimit-remaining") or r.headers.get("X-RateLimit-Remaining") or "?",
        "reset": r.headers.get("x-ratelimit-reset") or r.headers.get("X-RateLimit-Reset") or "?",
    }
    if r.status_code != 200:
        return None, f"{r.status_code}: {r.text[:800]}", usage
    try:
        return r.json(), None, usage
    except Exception as e:
        return None, f"Could not parse JSON: {e}", usage


def fetch_leagues(api_key: str, sport: str) -> Tuple[List[Dict], Optional[str], Dict[str, str]]:
    data, err, usage = api_get("/leagues", {"apiKey": api_key, "sport": sport, "all": "true"})
    if err:
        return [], err, usage
    return data if isinstance(data, list) else [], None, usage


def fetch_events(api_key: str, sport: str, league_slug: Optional[str], bookmaker: Optional[str], limit: int = 25) -> Tuple[List[Dict], Optional[str], Dict[str, str]]:
    params = {"apiKey": api_key, "sport": sport, "status": "pending", "limit": limit}
    if league_slug:
        params["league"] = league_slug
    if bookmaker:
        params["bookmaker"] = bookmaker
    data, err, usage = api_get("/events", params)
    if err:
        return [], err, usage
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else [], None, usage


def fetch_odds(api_key: str, event_id: str, bookmakers: List[str]) -> Tuple[Optional[Dict], Optional[str], Dict[str, str]]:
    params = {"apiKey": api_key, "eventId": event_id, "bookmakers": ",".join(bookmakers)}
    data, err, usage = api_get("/odds", params)
    if err:
        return None, err, usage
    return data if isinstance(data, dict) else None, None, usage


def fetch_value_bets(api_key: str, bookmaker: str, sport: Optional[str] = None) -> Tuple[List[Dict], Optional[str], Dict[str, str]]:
    params = {"apiKey": api_key, "bookmaker": bookmaker, "includeEventDetails": "true"}
    if sport:
        params["sport"] = sport
    data, err, usage = api_get("/value-bets", params)
    if err:
        return [], err, usage
    return data if isinstance(data, list) else [], None, usage


def fetch_arbitrage(api_key: str, bookmakers: List[str]) -> Tuple[List[Dict], Optional[str], Dict[str, str]]:
    data, err, usage = api_get("/arbitrage-bets", {"apiKey": api_key, "bookmakers": ",".join(bookmakers)})
    if err:
        return [], err, usage
    return data if isinstance(data, list) else [], None, usage

# --------------------------- Parsing Odds-API.io shape ---------------------------
def normalize_decimal(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def mk_selection_key(selection: str, point: Optional[float]) -> str:
    return f"{selection}|{'' if point is None else point}"


def parse_market_odds(market: Dict) -> List[Dict]:
    rows = []
    market_name = market.get("key") or market.get("name", "Unknown")

    odds_list = market.get("odds", [])
    if isinstance(odds_list, dict):
        odds_list = [odds_list]

    for item in odds_list:
        if not isinstance(item, dict):
            continue

        point = item.get("hdp") or item.get("point") or item.get("line")

        try:
            point = float(point) if point is not None else None
        except Exception:
            point = None

        player_name = (
            item.get("participant")
            or item.get("player")
            or item.get("name")
            or item.get("description")
            or item.get("label")
            or ""
        )

        for key, val in item.items():
            if key in ["hdp", "point", "line", "updatedAt", "suspended", "participant", "player", "name", "description", "label"]:
                continue

            dec = normalize_decimal(val)
            if dec is None or dec <= 1:
                continue

            side = str(key).replace("_", " ").title()

            rows.append({
                "market": market_name,
                "selection": f"{player_name} {side} {point} {market_name}".strip(),
                "point": point,
                "decimal_odds": dec
            })

    return rows


def odds_response_to_rows(odds_data: Dict, sport: str, target_book: str, sharp_books: List[str], bankroll: float, kelly_mult: float, market_filter: List[str]) -> pd.DataFrame:
    if not odds_data:
        return pd.DataFrame()
    ts = datetime.now(timezone.utc).isoformat()
    event_id = str(odds_data.get("id", ""))
    home = odds_data.get("home", "Home")
    away = odds_data.get("away", "Away")
    event_name = f"{away} @ {home}"
    league = odds_data.get("league", {})
    league_name = league.get("name") if isinstance(league, dict) else str(league)
    bookmakers = odds_data.get("bookmakers", {}) or {}
    if not isinstance(bookmakers, dict):
        return pd.DataFrame()

    offers_by_market_selection: Dict[Tuple[str, str], List[Dict]] = {}
    flat_offers = []

    for book_name, markets in bookmakers.items():
        if not isinstance(markets, list):
            continue
        for market in markets:
            market_name = market.get("key") or market.get("name", "Unknown")
            # Temporarily do not filter markets until we confirm the API market names
# if market_filter and market_name not in market_filter:
#     continue
            for parsed in parse_market_odds(market):
                american = decimal_to_american(parsed["decimal_odds"])
                if american is None:
                    continue
                skey = mk_selection_key(parsed["selection"], parsed["point"])
                offer = {
                    "sportsbook": book_name,
                    "market": market_name,
                    "selection": parsed["selection"],
                    "point": parsed["point"],
                    "decimal_odds": parsed["decimal_odds"],
                    "american_odds": american,
                    "selection_key": skey,
                }
                flat_offers.append(offer)
                offers_by_market_selection.setdefault((market_name, skey), []).append(offer)

    if not flat_offers:
        return pd.DataFrame()

    rows = []
    market_groups = sorted(
    set((o["market"], o["point"]) for o in flat_offers),
    key=lambda x: (str(x[0]), -999999 if x[1] is None else float(x[1]))
)
    for market_name, point_group in market_groups:
        market_offers = [
            o for o in flat_offers
            if o["market"] == market_name and o["point"] == point_group
        ]
        selection_keys = sorted(set(o["selection"] for o in market_offers))

        consensus_avg = {}
        sharp_avg = {}
        for skey in selection_keys:
            offers = [o for o in market_offers if o["selection"] == skey]
            non_target = [o for o in offers if o["sportsbook"].lower() != target_book.lower()] or offers
            avg_imp = sum(decimal_to_implied(o["decimal_odds"]) for o in non_target) / max(len(non_target), 1)
            consensus_avg[skey] = prob_to_american(avg_imp) or non_target[0]["american_odds"]

            sharp = [o for o in offers if o["sportsbook"].lower() in [s.lower() for s in sharp_books]]
            if sharp:
                avg_imp_sharp = sum(decimal_to_implied(o["decimal_odds"]) for o in sharp) / len(sharp)
                sharp_avg[skey] = prob_to_american(avg_imp_sharp) or sharp[0]["american_odds"]

        consensus_novig = no_vig_probs(consensus_avg) if len(consensus_avg) >= 2 else {}
        sharp_novig = no_vig_probs(sharp_avg) if len(sharp_avg) >= 2 else {}

        for offer in market_offers:
            skey = offer["selection"]
            true_prob = sharp_novig.get(skey) or consensus_novig.get(skey)
            if not true_prob:
                continue
            books_used = len(set(o["sportsbook"] for o in market_offers if o["selection_key"] == skey))
            sharp_available = bool(sharp_novig) and any(o["sportsbook"].lower() in [s.lower() for s in sharp_books] for o in market_offers)
            evp = ev_percent(true_prob, offer["american_odds"])
            edge = (true_prob - decimal_to_implied(offer["decimal_odds"])) * 100
            kelly = kelly_fraction(true_prob, offer["american_odds"]) * kelly_mult
            rows.append({
                "ts": ts,
                "sport": sport,
                "league": league_name,
                "event_id": event_id,
                "event_name": event_name,
                "market": offer["market"],
                "selection": offer["selection"],
                "point": offer["point"],
                "sportsbook": offer["sportsbook"],
                "decimal_odds": round(offer["decimal_odds"], 3),
                "american_odds": offer["american_odds"],
                "implied": round(decimal_to_implied(offer["decimal_odds"]), 4),
                "fair_prob": round(true_prob, 4),
                "fair_odds": prob_to_american(true_prob),
                "ev_percent": round(evp, 2),
                "edge_percent": round(edge, 2),
                "kelly_fraction": round(kelly * 100, 2),
                "suggested_stake": round(bankroll * kelly, 2),
                "books_used": books_used,
                "sharp_available": sharp_available,
                "confidence_score": confidence_score(evp, books_used, sharp_available, offer["market"]),
                "fair_source": "Sharp books" if skey in sharp_novig else "Consensus",
            })
    return pd.DataFrame(rows)

# --------------------------- UI Styling ---------------------------
def style_ev_table(df: pd.DataFrame):
    def color_ev(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 3:
            return "background-color: #0b6b3a; color: white; font-weight: bold"
        if v >= 0:
            return "background-color: #8a6d00; color: white"
        return "background-color: #7a1f1f; color: white"

    def color_conf(val):
        try:
            v = float(val)
        except Exception:
            return ""
        if v >= 70:
            return "background-color: #0b6b3a; color: white"
        if v >= 45:
            return "background-color: #8a6d00; color: white"
        return "background-color: #3a3a3a; color: white"

    styled = df.style
    if "ev_percent" in df.columns:
        styled = styled.map(color_ev, subset=["ev_percent"])
    if "confidence_score" in df.columns:
        styled = styled.map(color_conf, subset=["confidence_score"])
    return styled

# --------------------------- App ---------------------------
init_db()
st.title("EV Scanner Pro Max — Odds-API.io Edition")
st.caption("Built for Odds-API.io keys. Uses /leagues → /events → /odds, plus optional value-bet and arbitrage endpoint tabs.")

try:
    secret_key = st.secrets.get("ODDS_API_IO_KEY", "") or st.secrets.get("ODDS_API_KEY", "")
except Exception:
    secret_key = ""

with st.sidebar:
    st.header("Settings")
    if secret_key:
        st.success("Using API key from Streamlit Secrets.")
        api_key = secret_key
    else:
        api_key = st.text_input("Odds-API.io Key", type="password")
        st.caption("For deployment, add ODDS_API_IO_KEY in Streamlit Secrets.")

    st.subheader("Credit Saver")
    live = st.toggle("Live auto-refresh", value=False)
    interval = st.selectbox("Refresh every", options=[30, 60, 300], index=1, format_func=lambda x: "5 min" if x == 300 else f"{x} sec")
    if live:
        st_autorefresh(interval=interval * 1000, key="live_refresh")

    sports = st.multiselect("Sports", DEFAULT_SPORTS, default=["basketball"])
    league_keyword = st.text_input("League keyword filter", value="nba", help="For NBA, keep this as nba. The app finds matching basketball leagues automatically.")
    event_limit = st.number_input("Max events per league", min_value=1, max_value=50, value=10, step=1)
    max_leagues = st.number_input("Max matching leagues", min_value=1, max_value=10, value=2, step=1)

    st.subheader("Sportsbooks")
    target_book = st.text_input("Target sportsbook", value=DEFAULT_TARGET_BOOK)
    sharp_books_text = st.text_input("Sharp books, comma-separated", value=", ".join(DEFAULT_SHARP_BOOKS))
    extra_books_text = st.text_input("Extra comparison books, comma-separated", value="DraftKings, BetMGM")
    sharp_books = [x.strip() for x in sharp_books_text.split(",") if x.strip()]
    extra_books = [x.strip() for x in extra_books_text.split(",") if x.strip()]
    bookmakers = []
    for b in [target_book] + sharp_books + extra_books:
        if b and b.lower() not in [x.lower() for x in bookmakers]:
            bookmakers.append(b)

    MARKET_KEY_MAP = {
    "ML": "h2h",
    "Spread": "spreads",
    "Totals": "totals",
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "3PT Made": "player_threes",
    "PRA": "player_points_rebounds_assists",
    "Steals": "player_steals",
    "Blocks": "player_blocks",
    "Blocks + Steals": "player_blocks_steals",
    "Turnovers": "player_turnovers",
    "Points + Rebounds": "player_points_rebounds",
    "Points + Assists": "player_points_assists",
    "Rebounds + Assists": "player_rebounds_assists",
    "Double Double": "player_double_double",
    "Triple Double": "player_triple_double",
}

    markets = st.multiselect(
    "Markets",
    list(MARKET_KEY_MAP.keys()),
    default=["ML"]
)

    st.subheader("Serious Play Filters")
    min_ev = st.number_input("Minimum EV %", value=3.0, step=0.5)
    min_books = st.number_input("Minimum books used", min_value=1, value=2, step=1)
    require_sharp = st.toggle("Sharp book required", value=True)
    only_target = st.toggle("Only show target sportsbook", value=True)

    st.subheader("Bankroll")
    bankroll = st.number_input("Bankroll", min_value=0.0, value=100.0, step=10.0)
    kelly_mult = st.select_slider("Kelly multiplier", options=[0.1, 0.25, 0.5, 1.0], value=0.25)
    save_history = st.toggle("Save odds snapshots", value=True)

run = st.button("Run Odds-API.io Scan", type="primary") or live

if not api_key:
    st.info("Paste your Odds-API.io key or add ODDS_API_IO_KEY in Streamlit Secrets.")
elif run:
    frames = []
    errors = []
    usage_rows = []

    with st.status("Scanning Odds-API.io...", expanded=False) as status:
        for sport in sports:
            leagues, err, usage = fetch_leagues(api_key, sport)
            usage_rows.append({"step": "leagues", "sport": sport, **usage})
            if err:
                errors.append(f"{sport} leagues: {err}")
                continue
            if league_keyword.strip():
                lk = league_keyword.strip().lower()
                leagues = [l for l in leagues if lk in str(l.get("name", "")).lower() or lk in str(l.get("slug", "")).lower()]
            leagues = leagues[: int(max_leagues)]
            if not leagues:
                errors.append(f"No leagues matched '{league_keyword}' for sport '{sport}'. Try clearing the league keyword or using basketball + nba.")
                continue
            for league in leagues:
                slug = league.get("slug")
                lname = league.get("name", slug)
                events, err, usage = fetch_events(api_key, sport, slug, target_book, int(event_limit))
                usage_rows.append({"step": "events", "sport": sport, "league": lname, **usage})
                if err:
                    errors.append(f"{sport}/{lname} events: {err}")
                    continue
                for ev in events:
                    event_id = str(ev.get("id", ""))
                    if not event_id:
                        continue
                    odds_data, err, usage = fetch_odds(api_key, event_id, bookmakers)
                    usage_rows.append({"step": "odds", "sport": sport, "league": lname, "event_id": event_id, **usage})
                    if err:
                        errors.append(f"Event {event_id} odds: {err}")
                        continue
                    selected_market_keys = [MARKET_KEY_MAP[m] for m in markets]

                    rows = odds_response_to_rows(odds_data,sport,target_book,sharp_books,bankroll,kelly_mult,selected_market_keys)
                    if not rows.empty:
                        frames.append(rows)
        status.update(label="Scan complete", state="complete")

    if usage_rows:
        last_usage = usage_rows[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rate limit", last_usage.get("limit", "?"))
        c2.metric("Requests remaining", last_usage.get("remaining", "?"))
        c3.metric("Reset", last_usage.get("reset", "?"))
        with st.expander("Usage / rate-limit details"):
            st.dataframe(pd.DataFrame(usage_rows), use_container_width=True, hide_index=True)

    if errors:
        st.warning("\n".join(errors[:12]))
        if len(errors) > 12:
            st.caption(f"Showing 12 of {len(errors)} warnings/errors.")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        st.error("No odds rows came back. Check that your Odds-API.io selected bookmakers include the names in the sidebar, and try Sport=basketball, League keyword=nba, Market=ML.")
    else:
        if only_target:
            df = df[df["sportsbook"].str.lower() == target_book.lower()]
        if save_history and not df.empty:
            save_snapshots(df)

        df = df.sort_values(["ev_percent", "confidence_score"], ascending=False)
        serious = df[df["ev_percent"] >= min_ev]
        serious = serious[serious["books_used"] >= min_books]
        if require_sharp:
            serious = serious[serious["sharp_available"] == True]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows scanned", len(df))
        c2.metric("Serious plays", len(serious))
        c3.metric("Best EV %", f"{df['ev_percent'].max():.2f}%")
        c4.metric("Best confidence", int(df["confidence_score"].max()))

        tabs = st.tabs(["Serious +EV Feed","High Probability","Correlation Builder","Provider Value Bets","Provider Arbitrage","History / CLV",
        "Raw Odds"])
        display_cols = ["confidence_score", "sport", "league", "event_name", "market", "selection", "point", "sportsbook", "decimal_odds", "american_odds", "fair_prob", "fair_odds", "ev_percent", "edge_percent", "books_used", "sharp_available", "kelly_fraction", "suggested_stake", "fair_source"]

        with tabs[1]:
            st.subheader("High Probability Bets")
            
            high_prob = df[df["fair_prob"] >= 0.80].copy()
            high_prob = high_prob.sort_values(by="fair_prob", ascending=False).head(25)

            if high_prob.empty:
                st.info("No high probability bets found.")
            else:
                st.dataframe(style_ev_table(high_prob[display_cols]), use_container_width=True, hide_index=True)
                
                st.download_button(
                    "Download serious +EV CSV",
                    serious.to_csv(index=False).encode("utf-8"),
                    "serious_plus_ev_plays.csv",
                    "text/csv"
                )
        with tabs[2]:
            st.subheader("Correlation Builder")

    if df.empty:
        st.info("Run a scan first.")
    else:
        event_choice = st.selectbox(
            "Choose game",
            sorted(df["event_name"].dropna().unique())
        )

        game_df = df[df["event_name"] == event_choice].copy()

        anchor_choice = st.selectbox(
            "Choose your first leg",
            game_df["selection"].astype(str)
        )

        anchor_market = anchor_choice.lower()

        def simple_correlation_reason(anchor_market, candidate_market):
            a = anchor_market.lower()
            c = candidate_market.lower()

            if "player" in a and "totals" in c:
                return "Player production can correlate with high-scoring games."

            if "totals" in a and "player" in c:
                return "Higher totals can support player overs."

            if "player" in a and "ml" in c:
                return "Player prop overs can correlate with team wins."

            if "ml" in a and "player" in c:
                return "Team wins can support star player production."

            return ""

        def correlation_strength(anchor_market, candidate_market):
            a = anchor_market.lower()
            c = candidate_market.lower()

            if "player" in a and "totals" in c:
                return "Strong"

           

    return "Weak"
    with tabs[3]:
            st.caption("Uses Odds-API.io's /value-bets endpoint when your plan/bookmaker supports it.")
            if st.button("Fetch provider value bets"):
                vals, err, usage = fetch_value_bets(api_key, target_book, sports[0] if sports else None)
                if err:
                    st.error(err)
                elif vals:
                    st.dataframe(pd.json_normalize(vals), use_container_width=True)
                else:
                    st.info("No provider value bets returned.")

    with tabs[4]:
            st.caption("Uses Odds-API.io's /arbitrage-bets endpoint when available for your plan/bookmakers.")
            if st.button("Fetch provider arbitrage"):
                arbs, err, usage = fetch_arbitrage(api_key, bookmakers)
                if err:
                    st.error(err)
                elif arbs:
                    st.dataframe(pd.json_normalize(arbs), use_container_width=True)
                else:
                    st.info("No provider arbitrage returned.")

    with tabs[5]:
            hist = load_history()
            if hist.empty:
                st.info("No history saved yet. Run scans with 'Save odds snapshots' enabled.")
            else:
                event_choice = st.selectbox("Event", sorted(hist["event_name"].dropna().unique()))
                h = hist[hist["event_name"] == event_choice].copy()
                market_choice = st.selectbox("Market", sorted(h["market"].dropna().unique()))
                h = h[h["market"] == market_choice]
                selection_choice = st.selectbox("Selection", sorted(h["selection"].dropna().unique()))
                h = h[h["selection"] == selection_choice]
                h["ts_dt"] = pd.to_datetime(h["ts"], errors="coerce")
                summary = h.sort_values("ts_dt").groupby("sportsbook").agg(
                    opening_american_odds=("american_odds", "first"),
                    current_american_odds=("american_odds", "last"),
                    opening_ev=("ev_percent", "first"),
                    current_ev=("ev_percent", "last"),
                    first_seen=("ts", "first"),
                    latest_seen=("ts", "last"),
                ).reset_index()
                st.dataframe(summary, use_container_width=True, hide_index=True)
                fig = px.line(h, x="ts_dt", y="american_odds", color="sportsbook", title="Line Movement")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[6]:
            st.dataframe(style_ev_table(df[display_cols]), use_container_width=True, hide_index=True)
            st.download_button("Download raw odds CSV", df.to_csv(index=False).encode("utf-8"), "raw_odds.csv", "text/csv")
else:
    st.write("Choose settings and run a scan. For NBA: Sport = basketball, League keyword = nba, Target = FanDuel.")
    st.info("This version is for keys from odds-api.io/dashboard/settings, not The-Odds-API.com keys.")
