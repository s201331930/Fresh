"""
Test data availability for the expanded 120+ ticker universe.
"""
import yfinance as yf
import pandas as pd
import json

CANDIDATES = [
    # === US LARGE CAP STOCKS (20) ===
    {"name": "AAPL",  "ticker": "AAPL",  "market": "US", "label": "Apple", "class": "US Stock"},
    {"name": "MSFT",  "ticker": "MSFT",  "market": "US", "label": "Microsoft", "class": "US Stock"},
    {"name": "AMZN",  "ticker": "AMZN",  "market": "US", "label": "Amazon", "class": "US Stock"},
    {"name": "GOOG",  "ticker": "GOOG",  "market": "US", "label": "Alphabet", "class": "US Stock"},
    {"name": "META",  "ticker": "META",  "market": "US", "label": "Meta Platforms", "class": "US Stock"},
    {"name": "NVDA",  "ticker": "NVDA",  "market": "US", "label": "NVIDIA", "class": "US Stock"},
    {"name": "TSLA",  "ticker": "TSLA",  "market": "US", "label": "Tesla", "class": "US Stock"},
    {"name": "JPM",   "ticker": "JPM",   "market": "US", "label": "JP Morgan", "class": "US Stock"},
    {"name": "JNJ",   "ticker": "JNJ",   "market": "US", "label": "Johnson & Johnson", "class": "US Stock"},
    {"name": "PG",    "ticker": "PG",    "market": "US", "label": "Procter & Gamble", "class": "US Stock"},
    {"name": "UNH",   "ticker": "UNH",   "market": "US", "label": "UnitedHealth", "class": "US Stock"},
    {"name": "V",     "ticker": "V",     "market": "US", "label": "Visa", "class": "US Stock"},
    {"name": "MA",    "ticker": "MA",    "market": "US", "label": "Mastercard", "class": "US Stock"},
    {"name": "HD",    "ticker": "HD",    "market": "US", "label": "Home Depot", "class": "US Stock"},
    {"name": "KO",    "ticker": "KO",    "market": "US", "label": "Coca-Cola", "class": "US Stock"},
    {"name": "PEP",   "ticker": "PEP",   "market": "US", "label": "PepsiCo", "class": "US Stock"},
    {"name": "WMT",   "ticker": "WMT",   "market": "US", "label": "Walmart", "class": "US Stock"},
    {"name": "DIS",   "ticker": "DIS",   "market": "US", "label": "Disney", "class": "US Stock"},
    {"name": "NFLX",  "ticker": "NFLX",  "market": "US", "label": "Netflix", "class": "US Stock"},
    {"name": "BA",    "ticker": "BA",    "market": "US", "label": "Boeing", "class": "US Stock"},

    # === US SECTOR ETFs (3 new, already have XLK/XLF/XLE/XLP/XLV/XLU) ===
    {"name": "XLI",   "ticker": "XLI",   "market": "US", "label": "Industrials Sector", "class": "Sector ETF"},
    {"name": "XLB",   "ticker": "XLB",   "market": "US", "label": "Materials Sector", "class": "Sector ETF"},
    {"name": "XLRE",  "ticker": "XLRE",  "market": "US", "label": "Real Estate Sector", "class": "Sector ETF"},

    # === US BROAD MARKET ETFs (2 new) ===
    {"name": "DIA",   "ticker": "DIA",   "market": "US", "label": "Dow Jones ETF", "class": "Broad ETF"},
    {"name": "MDY",   "ticker": "MDY",   "market": "US", "label": "S&P MidCap 400", "class": "Broad ETF"},

    # === INTERNATIONAL ETFs (10 new) ===
    {"name": "EWG",   "ticker": "EWG",   "market": "US", "label": "Germany ETF", "class": "Intl ETF"},
    {"name": "EWU",   "ticker": "EWU",   "market": "US", "label": "UK ETF", "class": "Intl ETF"},
    {"name": "EWA",   "ticker": "EWA",   "market": "US", "label": "Australia ETF", "class": "Intl ETF"},
    {"name": "EWC",   "ticker": "EWC",   "market": "US", "label": "Canada ETF", "class": "Intl ETF"},
    {"name": "EWT",   "ticker": "EWT",   "market": "US", "label": "Taiwan ETF", "class": "Intl ETF"},
    {"name": "EWY",   "ticker": "EWY",   "market": "US", "label": "South Korea ETF", "class": "Intl ETF"},
    {"name": "EWS",   "ticker": "EWS",   "market": "US", "label": "Singapore ETF", "class": "Intl ETF"},
    {"name": "VWO",   "ticker": "VWO",   "market": "US", "label": "Vanguard EM ETF", "class": "Intl ETF"},
    {"name": "INDA",  "ticker": "INDA",  "market": "US", "label": "India ETF", "class": "Intl ETF"},
    {"name": "FXI",   "ticker": "FXI",   "market": "US", "label": "China Large-Cap ETF", "class": "Intl ETF"},

    # === COMMODITY ETFs (10) ===
    {"name": "SLV",   "ticker": "SLV",   "market": "US", "label": "Silver ETF", "class": "Commodity"},
    {"name": "USO",   "ticker": "USO",   "market": "US", "label": "US Oil Fund", "class": "Commodity"},
    {"name": "DBA",   "ticker": "DBA",   "market": "US", "label": "Agriculture ETF", "class": "Commodity"},
    {"name": "DBC",   "ticker": "DBC",   "market": "US", "label": "Commodity Index", "class": "Commodity"},
    {"name": "PPLT",  "ticker": "PPLT",  "market": "US", "label": "Platinum ETF", "class": "Commodity"},
    {"name": "PALL",  "ticker": "PALL",  "market": "US", "label": "Palladium ETF", "class": "Commodity"},
    {"name": "UNG",   "ticker": "UNG",   "market": "US", "label": "Natural Gas ETF", "class": "Commodity"},
    {"name": "CPER",  "ticker": "CPER",  "market": "US", "label": "Copper ETF", "class": "Commodity"},
    {"name": "NIB",   "ticker": "NIB",   "market": "US", "label": "Cocoa ETF", "class": "Commodity"},
    {"name": "JO",    "ticker": "JO",    "market": "US", "label": "Coffee ETF", "class": "Commodity"},

    # === CURRENCY ETFs (8) ===
    {"name": "FXE",   "ticker": "FXE",   "market": "US", "label": "Euro Currency ETF", "class": "Currency"},
    {"name": "FXY",   "ticker": "FXY",   "market": "US", "label": "Japanese Yen ETF", "class": "Currency"},
    {"name": "FXB",   "ticker": "FXB",   "market": "US", "label": "British Pound ETF", "class": "Currency"},
    {"name": "FXA",   "ticker": "FXA",   "market": "US", "label": "Australian Dollar ETF", "class": "Currency"},
    {"name": "FXC",   "ticker": "FXC",   "market": "US", "label": "Canadian Dollar ETF", "class": "Currency"},
    {"name": "UUP",   "ticker": "UUP",   "market": "US", "label": "US Dollar Bull ETF", "class": "Currency"},
    {"name": "FXF",   "ticker": "FXF",   "market": "US", "label": "Swiss Franc ETF", "class": "Currency"},
    {"name": "CEW",   "ticker": "CEW",   "market": "US", "label": "EM Currency ETF", "class": "Currency"},

    # === FOREX PAIRS (6) ===
    {"name": "EURUSD", "ticker": "EURUSD=X", "market": "US", "label": "EUR/USD", "class": "Forex"},
    {"name": "GBPUSD", "ticker": "GBPUSD=X", "market": "US", "label": "GBP/USD", "class": "Forex"},
    {"name": "USDJPY", "ticker": "JPY=X",    "market": "US", "label": "USD/JPY", "class": "Forex"},
    {"name": "AUDUSD", "ticker": "AUDUSD=X", "market": "US", "label": "AUD/USD", "class": "Forex"},
    {"name": "USDCHF", "ticker": "CHF=X",    "market": "US", "label": "USD/CHF", "class": "Forex"},
    {"name": "USDCAD", "ticker": "CAD=X",    "market": "US", "label": "USD/CAD", "class": "Forex"},

    # === BOND ETFs (8) ===
    {"name": "IEF",   "ticker": "IEF",   "market": "US", "label": "7-10 Year Treasury", "class": "Bond ETF"},
    {"name": "SHY",   "ticker": "SHY",   "market": "US", "label": "1-3 Year Treasury", "class": "Bond ETF"},
    {"name": "HYG",   "ticker": "HYG",   "market": "US", "label": "High Yield Corporate", "class": "Bond ETF"},
    {"name": "LQD",   "ticker": "LQD",   "market": "US", "label": "IG Corporate Bond", "class": "Bond ETF"},
    {"name": "AGG",   "ticker": "AGG",   "market": "US", "label": "US Aggregate Bond", "class": "Bond ETF"},
    {"name": "BND",   "ticker": "BND",   "market": "US", "label": "Total Bond Market", "class": "Bond ETF"},
    {"name": "MUB",   "ticker": "MUB",   "market": "US", "label": "Municipal Bond", "class": "Bond ETF"},
    {"name": "TIP",   "ticker": "TIP",   "market": "US", "label": "TIPS Bond ETF", "class": "Bond ETF"},

    # === REIT ETFs (4) ===
    {"name": "VNQ",   "ticker": "VNQ",   "market": "US", "label": "US REIT ETF", "class": "REIT"},
    {"name": "IYR",   "ticker": "IYR",   "market": "US", "label": "iShares US REIT", "class": "REIT"},
    {"name": "REM",   "ticker": "REM",   "market": "US", "label": "Mortgage REIT ETF", "class": "REIT"},
    {"name": "VNQI",  "ticker": "VNQI",  "market": "US", "label": "Intl REIT ETF", "class": "REIT"},

    # === TASI STOCKS (6 new) ===
    {"name": "Riyad_Bank",    "ticker": "1010.SR", "market": "TASI", "label": "Riyad Bank", "class": "TASI Stock"},
    {"name": "Alinma",        "ticker": "1150.SR", "market": "TASI", "label": "Alinma Bank", "class": "TASI Stock"},
    {"name": "Saudi_Kayan",   "ticker": "2350.SR", "market": "TASI", "label": "Saudi Kayan", "class": "TASI Stock"},
    {"name": "Tawuniya",      "ticker": "8010.SR", "market": "TASI", "label": "Tawuniya Insurance", "class": "TASI Stock"},
    {"name": "Dar_Al_Arkan",  "ticker": "4300.SR", "market": "TASI", "label": "Dar Al Arkan", "class": "TASI Stock"},
    {"name": "Jarir",         "ticker": "4190.SR", "market": "TASI", "label": "Jarir Marketing", "class": "TASI Stock"},

    # === INTERNATIONAL STOCKS (8) ===
    {"name": "BABA",  "ticker": "BABA",  "market": "US", "label": "Alibaba", "class": "Intl Stock"},
    {"name": "TSM",   "ticker": "TSM",   "market": "US", "label": "TSMC", "class": "Intl Stock"},
    {"name": "NVO",   "ticker": "NVO",   "market": "US", "label": "Novo Nordisk", "class": "Intl Stock"},
    {"name": "TM",    "ticker": "TM",    "market": "US", "label": "Toyota Motor", "class": "Intl Stock"},
    {"name": "SHEL",  "ticker": "SHEL",  "market": "US", "label": "Shell plc", "class": "Intl Stock"},
    {"name": "ASML",  "ticker": "ASML",  "market": "US", "label": "ASML Holdings", "class": "Intl Stock"},
    {"name": "SAP",   "ticker": "SAP",   "market": "US", "label": "SAP SE", "class": "Intl Stock"},
    {"name": "RIO",   "ticker": "RIO",   "market": "US", "label": "Rio Tinto", "class": "Intl Stock"},

    # === THEMATIC / SPECIALTY ETFs (5) ===
    {"name": "ARKK",  "ticker": "ARKK",  "market": "US", "label": "ARK Innovation", "class": "Thematic ETF"},
    {"name": "XBI",   "ticker": "XBI",   "market": "US", "label": "Biotech ETF", "class": "Thematic ETF"},
    {"name": "HACK",  "ticker": "HACK",  "market": "US", "label": "Cybersecurity ETF", "class": "Thematic ETF"},
    {"name": "TAN",   "ticker": "TAN",   "market": "US", "label": "Solar ETF", "class": "Thematic ETF"},
    {"name": "KWEB",  "ticker": "KWEB",  "market": "US", "label": "China Internet ETF", "class": "Thematic ETF"},
]

MIN_WEEKS = 200

def test_all():
    results = []
    by_class = {}
    for asset in CANDIDATES:
        name = asset["name"]
        ticker = asset["ticker"]
        mkt = asset["market"]
        cls = asset["class"]
        try:
            tk = yf.Ticker(ticker)
            daily = tk.history(start="2000-01-01", auto_adjust=False)
            if daily.empty:
                print(f"  {name:18s} ({ticker:12s}) {cls:15s}: NO DATA")
                continue
            rule = "W-THU" if mkt == "TASI" else "W-FRI"
            weekly = daily["Close"].resample(rule).last().dropna()
            n = len(weekly)
            start = daily.index.min().strftime("%Y-%m-%d")
            end = daily.index.max().strftime("%Y-%m-%d")
            has_vol = "Volume" in daily.columns and daily["Volume"].sum() > 0
            ok = n >= MIN_WEEKS
            status = "OK" if ok else "SHORT"
            print(f"  {name:18s} ({ticker:12s}) {cls:15s}: {n:5d} wks  {start} to {end}  vol={'Y' if has_vol else 'N'}  [{status}]")
            if ok:
                asset["n_weeks"] = n
                asset["has_volume"] = has_vol
                results.append(asset)
                by_class.setdefault(cls, []).append(name)
        except Exception as e:
            print(f"  {name:18s} ({ticker:12s}): ERROR - {e}")

    print(f"\n{'='*60}")
    print(f"Qualifying assets (>= {MIN_WEEKS} weeks): {len(results)}")
    for cls, names in sorted(by_class.items()):
        print(f"  {cls:18s}: {len(names):3d} ({', '.join(names[:5])}{'...' if len(names)>5 else ''})")

    with open("study/expanded_universe.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to study/expanded_universe.json")

if __name__ == "__main__":
    test_all()
