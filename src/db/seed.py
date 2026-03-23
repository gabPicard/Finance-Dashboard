"""Seed the database with real market and corporation data.

Running this module directly or calling ``seed()`` is idempotent — calling it
twice will not create duplicate rows.
"""

from __future__ import annotations

from .database import init_db, session_scope
from .models import Corporation, Market

# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

_MARKETS: list[dict] = [
    {"name": "S&P 500", "exchange": "NYSE/NASDAQ", "region": "US", "currency": "USD"},
    {"name": "NASDAQ", "exchange": "NASDAQ", "region": "US", "currency": "USD"},
    {"name": "CAC 40", "exchange": "Euronext Paris", "region": "France", "currency": "EUR"},
]

_CORPORATIONS: list[dict] = [
    # ── S&P 500 ──────────────────────────────────────────────────────────
    {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 3_000_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1980-12-12",
    },
    {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "industry": "Software—Infrastructure",
        "market_cap": 3_100_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1986-03-13",
    },
    {
        "ticker": "GOOGL",
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "market_cap": 2_100_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "2004-08-19",
    },
    {
        "ticker": "AMZN",
        "name": "Amazon.com, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Internet Retail",
        "market_cap": 1_900_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1997-05-15",
    },
    {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 2_800_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1999-01-22",
    },
    {
        "ticker": "META",
        "name": "Meta Platforms, Inc.",
        "sector": "Communication Services",
        "industry": "Internet Content & Information",
        "market_cap": 1_400_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "2012-05-18",
    },
    {
        "ticker": "TSLA",
        "name": "Tesla, Inc.",
        "sector": "Consumer Cyclical",
        "industry": "Auto Manufacturers",
        "market_cap": 800_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "2010-06-29",
    },
    {
        "ticker": "JPM",
        "name": "JPMorgan Chase & Co.",
        "sector": "Financial Services",
        "industry": "Banks—Diversified",
        "market_cap": 590_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1969-03-05",
    },
    {
        "ticker": "JNJ",
        "name": "Johnson & Johnson",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers—General",
        "market_cap": 380_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "1944-09-25",
    },
    {
        "ticker": "V",
        "name": "Visa Inc.",
        "sector": "Financial Services",
        "industry": "Credit Services",
        "market_cap": 540_000_000_000.0,
        "market_name": "S&P 500",
        "ipo_date": "2008-03-19",
    },
    # ── NASDAQ ───────────────────────────────────────────────────────────
    {
        "ticker": "ASML",
        "name": "ASML Holding N.V.",
        "sector": "Technology",
        "industry": "Semiconductor Equipment & Materials",
        "market_cap": 350_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1995-03-27",
    },
    {
        "ticker": "ADBE",
        "name": "Adobe Inc.",
        "sector": "Technology",
        "industry": "Software—Application",
        "market_cap": 210_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1986-08-20",
    },
    {
        "ticker": "NFLX",
        "name": "Netflix, Inc.",
        "sector": "Communication Services",
        "industry": "Entertainment",
        "market_cap": 270_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "2002-05-23",
    },
    {
        "ticker": "AMD",
        "name": "Advanced Micro Devices, Inc.",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 260_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1972-09-27",
    },
    {
        "ticker": "INTC",
        "name": "Intel Corporation",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 90_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1971-10-13",
    },
    {
        "ticker": "CSCO",
        "name": "Cisco Systems, Inc.",
        "sector": "Technology",
        "industry": "Communication Equipment",
        "market_cap": 220_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1990-02-16",
    },
    {
        "ticker": "QCOM",
        "name": "Qualcomm Incorporated",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 170_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1991-12-13",
    },
    {
        "ticker": "AVGO",
        "name": "Broadcom Inc.",
        "sector": "Technology",
        "industry": "Semiconductors",
        "market_cap": 750_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "2009-08-06",
    },
    {
        "ticker": "COST",
        "name": "Costco Wholesale Corporation",
        "sector": "Consumer Defensive",
        "industry": "Discount Stores",
        "market_cap": 380_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1985-12-05",
    },
    {
        "ticker": "SBUX",
        "name": "Starbucks Corporation",
        "sector": "Consumer Cyclical",
        "industry": "Restaurants",
        "market_cap": 80_000_000_000.0,
        "market_name": "NASDAQ",
        "ipo_date": "1992-06-26",
    },
    # ── CAC 40 ───────────────────────────────────────────────────────────
    {
        "ticker": "MC.PA",
        "name": "LVMH Moët Hennessy Louis Vuitton SE",
        "sector": "Consumer Cyclical",
        "industry": "Luxury Goods",
        "market_cap": 320_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1989-01-09",
    },
    {
        "ticker": "OR.PA",
        "name": "L'Oréal S.A.",
        "sector": "Consumer Defensive",
        "industry": "Household & Personal Products",
        "market_cap": 190_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1963-01-01",
    },
    {
        "ticker": "TTE.PA",
        "name": "TotalEnergies SE",
        "sector": "Energy",
        "industry": "Oil & Gas Integrated",
        "market_cap": 140_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1985-01-01",
    },
    {
        "ticker": "SAN.PA",
        "name": "Sanofi S.A.",
        "sector": "Healthcare",
        "industry": "Drug Manufacturers—General",
        "market_cap": 130_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1999-01-01",
    },
    {
        "ticker": "AIR.PA",
        "name": "Airbus SE",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "market_cap": 110_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "2000-07-10",
    },
    {
        "ticker": "BNP.PA",
        "name": "BNP Paribas S.A.",
        "sector": "Financial Services",
        "industry": "Banks—Diversified",
        "market_cap": 70_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1993-10-01",
    },
    {
        "ticker": "KER.PA",
        "name": "Kering S.A.",
        "sector": "Consumer Cyclical",
        "industry": "Luxury Goods",
        "market_cap": 35_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1988-01-01",
    },
    {
        "ticker": "RI.PA",
        "name": "Pernod Ricard S.A.",
        "sector": "Consumer Defensive",
        "industry": "Beverages—Wineries & Distilleries",
        "market_cap": 28_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1975-01-01",
    },
    {
        "ticker": "HO.PA",
        "name": "Thales S.A.",
        "sector": "Industrials",
        "industry": "Aerospace & Defense",
        "market_cap": 25_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1994-01-01",
    },
    {
        "ticker": "ORA.PA",
        "name": "Orange S.A.",
        "sector": "Communication Services",
        "industry": "Telecom Services",
        "market_cap": 22_000_000_000.0,
        "market_name": "CAC 40",
        "ipo_date": "1997-10-01",
    },
]


def seed() -> None:
    """Populate Market and Corporation tables with real data.

    Idempotent: existing rows are left unchanged.
    """
    init_db()

    with session_scope() as session:
        # --- markets ---
        market_map: dict[str, int] = {}
        for m in _MARKETS:
            existing = session.query(Market).filter_by(name=m["name"]).first()
            if existing is None:
                market = Market(
                    name=m["name"],
                    exchange=m["exchange"],
                    region=m["region"],
                    currency=m["currency"],
                )
                session.add(market)
                session.flush()  # get the generated id
                market_map[m["name"]] = market.id
            else:
                market_map[m["name"]] = existing.id

        # --- corporations ---
        for c in _CORPORATIONS:
            existing = session.query(Corporation).filter_by(ticker=c["ticker"]).first()
            if existing is not None:
                continue
            corp = Corporation(
                ticker=c["ticker"],
                name=c["name"],
                sector=c.get("sector"),
                industry=c.get("industry"),
                market_cap=c.get("market_cap"),
                market_id=market_map.get(c.get("market_name", "")),
                description=c.get("description"),
                ipo_date=c.get("ipo_date"),
            )
            session.add(corp)


if __name__ == "__main__":
    seed()
    print("Database seeded successfully.")
