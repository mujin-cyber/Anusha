"""
NSE Stock Analysis API - Streaming + Analysis
Data source: Yahoo Finance via yfinance

LOCAL:
    pip install -r requirements.txt
    uvicorn nse_api:app --reload --port 8000

TEST:
    http://127.0.0.1:8000/
    http://127.0.0.1:8000/quote/RELIANCE
    http://127.0.0.1:8000/analyze/RELIANCE
    http://127.0.0.1:8000/chart/RELIANCE?months=6

IMPORTANT:
- Yahoo Finance/yfinance is NOT an official NSE real-time feed.
- The quote endpoint starts a yfinance WebSocket subscription for a symbol
  and caches the latest streamed price.
- If the stream has not produced a value yet, /quote falls back to Yahoo's
  1-minute history endpoint.
- /analyze is intentionally separate and should NOT be polled every second.
"""

from datetime import datetime, timezone
from typing import Optional
import asyncio
import math
import threading
import time

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="NSE Stock Analysis API",
    version="2.0.0",
    description="Indian stock quotes, streaming quotes and technical/fundamental analysis using yfinance."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# GLOBAL STREAM CACHE
# ================================================================

# Example:
# {
#   "RELIANCE.NS": {
#       "price": 1215.50,
#       "previous_close": 1219.20,
#       ...
#   }
# }
STREAM_CACHE = {}

# Symbols for which a WebSocket listener has already been started.
STREAM_THREADS = {}

STREAM_LOCK = threading.Lock()


# ================================================================
# GENERAL HELPERS
# ================================================================

def clean_symbol(symbol: str) -> str:
    symbol = symbol.upper().strip()

    if not symbol:
        raise HTTPException(
            status_code=400,
            detail="Stock symbol is required."
        )

    return symbol[:-3] if symbol.endswith(".NS") else symbol


def yf_symbol(symbol: str) -> str:
    return f"{clean_symbol(symbol)}.NS"


def safe_float(value, digits: int = 2) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return round(value, digits)

    except Exception:
        return None


def safe_int(value) -> Optional[int]:
    try:
        if value is None or pd.isna(value):
            return None

        return int(value)

    except Exception:
        return None


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ================================================================
# STREAM MESSAGE HELPERS
# ================================================================

def first_value(message: dict, *keys):
    """
    Return the first non-null value from a Yahoo WebSocket message.

    We support both camelCase and snake_case variants because the exact
    decoded message shape can vary between yfinance versions.
    """
    for key in keys:
        if key in message and message[key] is not None:
            return message[key]

    return None


def process_stream_message(message: dict):
    """
    Process one Yahoo Finance WebSocket message.

    Typical Yahoo fields include:
        id
        price
        time
        currency
        exchange
        quoteType
        marketHours
        changePercent
        change
        dayVolume
        dayHigh
        dayLow
        openPrice
        previousClose
        marketPrice
    """

    if not isinstance(message, dict):
        return

    symbol = first_value(
        message,
        "id",
        "symbol",
        "ticker",
    )

    if not symbol:
        return

    symbol = str(symbol).upper()

    # We only use NSE symbols in this application.
    if not symbol.endswith(".NS"):
        return

    price = first_value(
        message,
        "price",
        "market_price",
        "marketPrice",
        "regular_market_price",
        "regularMarketPrice",
    )

    if price is None:
        return

    price = safe_float(price)

    if price is None:
        return

    previous_close = safe_float(
        first_value(
            message,
            "previous_close",
            "previousClose",
            "regular_market_previous_close",
            "regularMarketPreviousClose",
        )
    )

    change = safe_float(
        first_value(
            message,
            "change",
        )
    )

    change_percent = safe_float(
        first_value(
            message,
            "change_percent",
            "changePercent",
        )
    )

    # If Yahoo does not send change fields, calculate them.
    if change is None and previous_close not in (None, 0):
        change = safe_float(price - previous_close)

    if change_percent is None and previous_close not in (None, 0):
        change_percent = safe_float(
            (price - previous_close) / previous_close * 100
        )

    direction = "flat"

    if change is not None:
        if change > 0:
            direction = "up"
        elif change < 0:
            direction = "down"

    stream_time = first_value(
        message,
        "time",
        "timestamp",
    )

    # Yahoo generally supplies epoch seconds.
    market_timestamp = None

    if stream_time is not None:
        try:
            ts = float(stream_time)

            # Convert milliseconds if necessary.
            if ts > 10_000_000_000:
                ts = ts / 1000

            market_timestamp = datetime.fromtimestamp(
                ts,
                tz=timezone.utc
            ).isoformat()

        except Exception:
            market_timestamp = str(stream_time)

    with STREAM_LOCK:
        STREAM_CACHE[symbol] = {
            "symbol": symbol[:-3] if symbol.endswith(".NS") else symbol,
            "ticker": symbol,
            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "direction": direction,

            "open": safe_float(
                first_value(message, "open_price", "openPrice", "open", "regular_market_open", "regularMarketOpen")
            ),

            "high": safe_float(
                first_value(message, "day_high", "dayHigh", "high", "regular_market_day_high", "regularMarketDayHigh")
            ),

            "low": safe_float(
                first_value(message, "day_low", "dayLow", "low", "regular_market_day_low", "regularMarketDayLow")
            ),

            "volume": safe_int(
                first_value(message, "day_volume", "dayVolume", "volume", "regular_market_volume", "regularMarketVolume")
            ),

            "market_data_timestamp": market_timestamp,
            "api_timestamp": now_utc(),

            "source": "Yahoo Finance via yfinance WebSocket",
            "data_type": "websocket_stream",

            "note": (
                "Streaming data from Yahoo Finance via yfinance. "
                "Not an official NSE real-time feed."
            ),
        }

        print(
            f"[STREAM] Price received: {symbol} = {price}"
        )


# ================================================================
# WEBSOCKET LISTENER
# ================================================================

async def async_websocket_worker(ticker_symbol: str):
    """
    Async Yahoo Finance WebSocket listener for one ticker.

    The async client is used because it has been verified to receive
    RELIANCE.NS ticks correctly in the local environment.
    """

    while True:
        try:
            print(f"[STREAM] Connecting to Yahoo AsyncWebSocket: {ticker_symbol}")

            async with yf.AsyncWebSocket(verbose=False) as ws:
                await ws.subscribe([ticker_symbol])

                print(f"[STREAM] Subscribed: {ticker_symbol}")

                # This blocks while Yahoo sends live messages.
                # process_stream_message updates STREAM_CACHE.
                await ws.listen(process_stream_message)

        except asyncio.CancelledError:
            print(f"[STREAM] Cancelled: {ticker_symbol}")
            raise

        except Exception as exc:
            print(f"[STREAM] {ticker_symbol} connection error: {exc}")

        # Small delay before reconnecting if listen() exits.
        await asyncio.sleep(3)


def websocket_worker(ticker_symbol: str):
    """
    Run the async WebSocket listener in its own dedicated thread/event loop.
    """

    try:
        asyncio.run(async_websocket_worker(ticker_symbol))
    except Exception as exc:
        print(f"[STREAM] {ticker_symbol} worker stopped: {exc}")


def ensure_stream(ticker_symbol: str):
    """
    Start a WebSocket listener for ticker_symbol only once.
    """

    with STREAM_LOCK:

        existing = STREAM_THREADS.get(ticker_symbol)

        if existing is not None and existing.is_alive():
            return

        thread = threading.Thread(
            target=websocket_worker,
            args=(ticker_symbol,),
            daemon=True,
            name=f"yfinance-stream-{ticker_symbol}",
        )

        STREAM_THREADS[ticker_symbol] = thread

        thread.start()


# ================================================================
# HISTORICAL DATA
# ================================================================

def get_history(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:

    try:
        df = yf.Ticker(ticker).history(
            period=period,
            interval=interval,
            auto_adjust=False,
            prepost=False,
            actions=True,
        )

        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for {ticker}."
            )

        df = df[~df.index.duplicated(keep="last")]
        df = df.dropna(subset=["Close"])

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No usable market data found for {ticker}."
            )

        return df

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Yahoo Finance error for {ticker}: {exc}"
        )


# ================================================================
# TECHNICAL INDICATORS
# ================================================================

def sma(series, period):
    return series.rolling(period).mean()


def ema(series, period):
    return series.ewm(
        span=period,
        adjust=False
    ).mean()


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result = 100 - (100 / (1 + rs))

    result = result.mask(
        (avg_loss == 0) & (avg_gain > 0),
        100
    )

    result = result.mask(
        (avg_gain == 0) & (avg_loss > 0),
        0
    )

    return result


def roc(series, period=12):
    return series.pct_change(period) * 100


def macd(series):

    fast = ema(series, 12)
    slow = ema(series, 26)

    line = fast - slow
    signal = ema(line, 9)
    histogram = line - signal

    return line, signal, histogram


def adx(high, low, close, period=14):

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0
        ),
        index=high.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0
        ),
        index=high.index
    )

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr.replace(0, np.nan)
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr.replace(0, np.nan)
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    )

    return dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()


def bollinger(close, period=20, multiplier=2):

    middle = close.rolling(period).mean()
    std = close.rolling(period).std()

    upper = middle + multiplier * std
    lower = middle - multiplier * std

    percent_b = (
        (close - lower)
        / (upper - lower).replace(0, np.nan)
    )

    bandwidth_pct = (
        (upper - lower)
        / middle.replace(0, np.nan)
        * 100
    )

    return (
        middle,
        upper,
        lower,
        percent_b,
        bandwidth_pct
    )


# ================================================================
# FUNDAMENTALS
# ================================================================

def get_fundamentals(ticker_obj, df):

    result = {
        "market_cap": None,
        "pe_ratio": None,
        "forward_pe": None,
        "peg_ratio": None,
        "price_to_book": None,
        "eps": None,
        "dividend_yield": None,
        "dividend_rate": None,
        "roe": None,
        "roa": None,
        "profit_margin": None,
        "operating_margin": None,
        "debt_to_equity": None,
        "current_ratio": None,
        "book_value": None,
        "enterprise_value": None,
        "enterprise_to_ebitda": None,
        "beta": None,
        "52_week_high": None,
        "52_week_low": None,
        "currency": None,
    }

    # fast_info
    try:
        fast = ticker_obj.fast_info

        for key, output_key in [
            ("market_cap", "market_cap"),
            ("currency", "currency"),
            ("year_high", "52_week_high"),
            ("year_low", "52_week_low"),
        ]:

            try:
                value = fast.get(key)
            except Exception:
                try:
                    value = fast[key]
                except Exception:
                    value = None

            if output_key == "currency":
                if value is not None:
                    result[output_key] = value
            else:
                result[output_key] = safe_float(value)

    except Exception:
        pass

    # info
    try:
        info = ticker_obj.info or {}

        mapping = {
            "trailingPE": "pe_ratio",
            "forwardPE": "forward_pe",
            "pegRatio": "peg_ratio",
            "priceToBook": "price_to_book",
            "trailingEps": "eps",
            "dividendYield": "dividend_yield",
            "dividendRate": "dividend_rate",
            "returnOnEquity": "roe",
            "returnOnAssets": "roa",
            "profitMargins": "profit_margin",
            "operatingMargins": "operating_margin",
            "debtToEquity": "debt_to_equity",
            "currentRatio": "current_ratio",
            "bookValue": "book_value",
            "enterpriseValue": "enterprise_value",
            "enterpriseToEbitda": "enterprise_to_ebitda",
            "beta": "beta",
            "fiftyTwoWeekHigh": "52_week_high",
            "fiftyTwoWeekLow": "52_week_low",
            "marketCap": "market_cap",
            "currency": "currency",
        }

        for source_key, output_key in mapping.items():

            value = info.get(source_key)

            if value is None:
                continue

            if output_key == "currency":
                result[output_key] = value
            else:
                result[output_key] = safe_float(value)

    except Exception:
        pass

    # History fallback for 52-week high/low.
    try:

        year = df.tail(252)

        if result["52_week_high"] is None:
            result["52_week_high"] = safe_float(
                year["High"].max()
            )

        if result["52_week_low"] is None:
            result["52_week_low"] = safe_float(
                year["Low"].min()
            )

    except Exception:
        pass

    return result


def get_company_name(ticker_obj, symbol):

    try:

        info = ticker_obj.info or {}

        return (
            info.get("longName")
            or info.get("shortName")
            or symbol
        )

    except Exception:

        return symbol


# ================================================================
# ROOT
# ================================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "api": "NSE Stock Analysis API",
        "version": "2.0.0",
        "data_source": "Yahoo Finance via yfinance",

        "quote_mode": "WebSocket live cache + 1-minute fallback",

        "endpoints": {
            "quote": "/quote/{symbol}",
            "analyze": "/analyze/{symbol}",
            "chart": "/chart/{symbol}?months=6",
            "quotes": "/quotes?symbols=RELIANCE,TCS,INFY",
            "watchlist": "/watchlist?symbols=RELIANCE,TCS,INFY"
        },

        "note": (
            "Yahoo Finance/yfinance is not an official NSE real-time feed."
        )
    }


# ================================================================
# QUOTE
# ================================================================

@app.get("/quote/{symbol}")
def quote(symbol: str):

    """
    Lightweight quote endpoint.

    First:
        Starts/uses a yfinance WebSocket stream.

    Then:
        Returns the newest cached streamed price.

    If no streamed value is available yet:
        Falls back to 1-minute Yahoo history.

    Your frontend can call this every 1 second.
    """

    base = clean_symbol(symbol)
    ticker_symbol = yf_symbol(base)

    # Make sure streaming has started.
    ensure_stream(ticker_symbol)

    # ------------------------------------------------------------
    # 1. Try WebSocket cache
    # ------------------------------------------------------------

    with STREAM_LOCK:
        cached = STREAM_CACHE.get(ticker_symbol)

        if cached is not None:
            return cached

    # ------------------------------------------------------------
    # 2. Fallback to 1-minute history
    # ------------------------------------------------------------

    try:

        ticker = yf.Ticker(ticker_symbol)

        intraday = ticker.history(
            period="1d",
            interval="1m",
            auto_adjust=False,
            prepost=False,
            actions=False
        )

        if intraday is None or intraday.empty:

            raise HTTPException(
                status_code=404,
                detail=f"No quote data available for {base}."
            )

        intraday = intraday.dropna(
            subset=["Close"]
        )

        if intraday.empty:

            raise HTTPException(
                status_code=404,
                detail=f"No valid quote data available for {base}."
            )

        last = intraday.iloc[-1]

        price = safe_float(last["Close"])

        previous_close = None

        try:

            daily = ticker.history(
                period="5d",
                interval="1d",
                auto_adjust=False,
                prepost=False,
                actions=False
            )

            daily = daily.dropna(
                subset=["Close"]
            )

            if len(daily) >= 2:
                previous_close = safe_float(
                    daily["Close"].iloc[-2]
                )

        except Exception:
            pass

        change = None
        change_percent = None

        if price is not None and previous_close not in (None, 0):

            change = safe_float(
                price - previous_close
            )

            change_percent = safe_float(
                change / previous_close * 100
            )

        direction = "flat"

        if change is not None:

            if change > 0:
                direction = "up"

            elif change < 0:
                direction = "down"

        ts = intraday.index[-1]

        return {
            "symbol": base,
            "ticker": ticker_symbol,

            "price": price,
            "previous_close": previous_close,
            "change": change,
            "change_percent": change_percent,
            "direction": direction,

            "open": safe_float(last["Open"]),
            "high": safe_float(last["High"]),
            "low": safe_float(last["Low"]),
            "volume": safe_int(last["Volume"]),

            "market_data_timestamp": (
                ts.isoformat()
                if hasattr(ts, "isoformat")
                else str(ts)
            ),

            "api_timestamp": now_utc(),

            "source": "Yahoo Finance via yfinance 1m fallback",
            "data_type": "1m_intraday_fallback",

            "note": (
                "WebSocket stream has not supplied a value yet; "
                "using Yahoo 1-minute history."
            )
        }

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=502,
            detail=f"Unable to fetch quote for {ticker_symbol}: {exc}"
        )


# ================================================================
# MULTIPLE QUOTES
# ================================================================

@app.get("/quotes")
def quotes(
    symbols: str = Query(
        ...,
        description="Comma-separated NSE symbols"
    )
):

    requested = [
        clean_symbol(x)
        for x in symbols.split(",")
        if x.strip()
    ]

    if not requested:

        raise HTTPException(
            status_code=400,
            detail="Provide at least one symbol."
        )

    results = []

    for symbol in requested:

        try:
            results.append(
                quote(symbol)
            )

        except Exception as exc:

            results.append({
                "symbol": symbol,
                "error": str(exc)
            })

    return {
        "timestamp": now_utc(),
        "count": len(results),
        "data": results
    }


# ================================================================
# ANALYZE
# ================================================================

@app.get("/analyze/{symbol}")
def analyze(symbol: str):

    """
    Full technical + fundamental analysis.

    IMPORTANT:
    Do NOT call this every second.

    Use /quote/{symbol} for frequent price updates.
    """

    base = clean_symbol(symbol)
    ticker_symbol = yf_symbol(base)

    ticker = yf.Ticker(ticker_symbol)

    df = get_history(
        ticker_symbol,
        period="2y",
        interval="1d"
    )

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Indicators
    ma20 = sma(close, 20)
    ma50 = sma(close, 50)
    ma200 = sma(close, 200)

    rsi14 = rsi(close, 14)

    roc12 = roc(close, 12)

    macd_line, macd_signal, macd_hist = macd(close)

    adx14 = adx(
        high,
        low,
        close,
        14
    )

    bb_middle, bb_upper, bb_lower, bb_percent_b, bb_bandwidth = (
        bollinger(close)
    )

    current_price = safe_float(
        close.iloc[-1]
    )

    current_ma50 = safe_float(
        ma50.iloc[-1]
    )

    current_ma200 = safe_float(
        ma200.iloc[-1]
    )

    # MA trend
    if current_ma50 is None or current_ma200 is None:

        ma_trend = None

    elif (
        current_price > current_ma50
        and current_ma50 > current_ma200
    ):

        ma_trend = "Bullish"

    elif (
        current_price < current_ma50
        and current_ma50 < current_ma200
    ):

        ma_trend = "Bearish"

    else:

        ma_trend = "Neutral"

    # 52-week values
    year = df.tail(252)

    high52 = safe_float(
        year["High"].max()
    )

    low52 = safe_float(
        year["Low"].min()
    )

    pct_of_52w_high = (
        safe_float(
            current_price / high52 * 100
        )
        if current_price is not None
        and high52 not in (None, 0)
        else None
    )

    # Daily change
    previous_close = (
        safe_float(
            close.iloc[-2]
        )
        if len(df) >= 2
        else None
    )

    change = (
        safe_float(
            current_price - previous_close
        )
        if current_price is not None
        and previous_close is not None
        else None
    )

    change_percent = (
        safe_float(
            change / previous_close * 100
        )
        if change is not None
        and previous_close not in (None, 0)
        else None
    )

    latest = df.iloc[-1]

    return {

        "symbol": base,

        "ticker": ticker_symbol,

        "company_name": get_company_name(
            ticker,
            base
        ),

        "timestamp": now_utc(),

        "price": {

            "current": current_price,

            "previous_close": previous_close,

            "change": change,

            "change_percent": change_percent,

            "open": safe_float(
                latest["Open"]
            ),

            "high": safe_float(
                latest["High"]
            ),

            "low": safe_float(
                latest["Low"]
            ),

            "volume": safe_int(
                latest["Volume"]
            )
        },

        "technical": {

            "current_price": current_price,

            "rsi_14": safe_float(
                rsi14.iloc[-1]
            ),

            "roc_12": safe_float(
                roc12.iloc[-1]
            ),

            "macd_line": safe_float(
                macd_line.iloc[-1]
            ),

            "macd_signal": safe_float(
                macd_signal.iloc[-1]
            ),

            "macd_histogram": safe_float(
                macd_hist.iloc[-1]
            ),

            "ma_20": safe_float(
                ma20.iloc[-1]
            ),

            "ma_50": current_ma50,

            "ma_200": current_ma200,

            "ma_trend": ma_trend,

            "adx_14": safe_float(
                adx14.iloc[-1]
            ),

            "52_week_high": high52,

            "52_week_low": low52,

            "pct_of_52w_high": pct_of_52w_high,

            "bollinger_middle": safe_float(
                bb_middle.iloc[-1]
            ),

            "bollinger_upper": safe_float(
                bb_upper.iloc[-1]
            ),

            "bollinger_lower": safe_float(
                bb_lower.iloc[-1]
            ),

            "bollinger_percent_b": safe_float(
                bb_percent_b.iloc[-1]
            ),

            "bollinger_bandwidth_pct": safe_float(
                bb_bandwidth.iloc[-1]
            )
        },

        "fundamentals": get_fundamentals(
            ticker,
            df
        ),

        "data_source": "Yahoo Finance via yfinance",

        "analysis_note": (
            "Technical indicators are calculated from Yahoo Finance "
            "daily OHLCV data. Fundamental fields depend on Yahoo "
            "Finance availability and are returned as null when unavailable."
        )
    }


# ================================================================
# CHART
# ================================================================

@app.get("/chart/{symbol}")
def chart(
    symbol: str,
    months: int = Query(
        6,
        ge=1,
        le=24
    )
):

    base = clean_symbol(symbol)
    ticker_symbol = yf_symbol(base)

    df = get_history(
        ticker_symbol,
        period="2y",
        interval="1d"
    )

    requested_days = max(
        22,
        months * 22
    )

    view = df.tail(
        requested_days
    )

    ma50 = sma(
        df["Close"],
        50
    )

    ma200 = sma(
        df["Close"],
        200
    )

    rsi14 = rsi(
        df["Close"],
        14
    )

    macd_line, macd_signal, _ = macd(
        df["Close"]
    )

    rows = []

    for idx in view.index:

        rows.append({

            "date": (
                idx.date().isoformat()
                if hasattr(idx, "date")
                else str(idx)
            ),

            "open": safe_float(
                view.loc[idx, "Open"]
            ),

            "high": safe_float(
                view.loc[idx, "High"]
            ),

            "low": safe_float(
                view.loc[idx, "Low"]
            ),

            "close": safe_float(
                view.loc[idx, "Close"]
            ),

            "volume": safe_int(
                view.loc[idx, "Volume"]
            ),

            "rsi": safe_float(
                rsi14.loc[idx]
            ),

            "macd_line": safe_float(
                macd_line.loc[idx]
            ),

            "macd_signal": safe_float(
                macd_signal.loc[idx]
            ),

            "ma_50": safe_float(
                ma50.loc[idx]
            ),

            "ma_200": safe_float(
                ma200.loc[idx]
            )
        })

    return {

        "symbol": base,

        "ticker": ticker_symbol,

        "months": months,

        "timestamp": now_utc(),

        "data": rows,

        "source": "Yahoo Finance via yfinance"
    }


# ================================================================
# WATCHLIST
# ================================================================

@app.get("/watchlist")
def watchlist(
    symbols: str = Query(
        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK"
    )
):

    requested = [
        clean_symbol(x)
        for x in symbols.split(",")
        if x.strip()
    ]

    result = []

    for symbol in requested:

        try:

            q = quote(symbol)

            result.append({

                "symbol": q["symbol"],

                "price": q["price"],

                "change": q["change"],

                "change_percent": q["change_percent"],

                "direction": q["direction"],

                "volume": q["volume"],

                "market_data_timestamp": (
                    q["market_data_timestamp"]
                )
            })

        except Exception as exc:

            result.append({
                "symbol": symbol,
                "error": str(exc)
            })

    return {
        "timestamp": now_utc(),
        "data": result
    }


# ================================================================
# LOCAL RUN
# ================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "nse_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
