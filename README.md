# EV Scanner Pro Max — Odds-API.io Edition

This version is built for keys from https://odds-api.io/dashboard/settings.

## Streamlit Secrets
Use either:

```toml
ODDS_API_IO_KEY = "your_odds_api_io_key"
```

or paste your key in the sidebar.

## Run locally
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Notes
- Odds-API.io uses sport slugs like `basketball` instead of `basketball_nba`.
- NBA is filtered through the league keyword box, default `nba`.
- Bookmaker names are usually display names like `FanDuel`, `DraftKings`, `Pinnacle`.
