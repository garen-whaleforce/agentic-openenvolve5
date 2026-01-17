#!/usr/bin/env python3
"""
Paper Trading Database Schema and Operations

Versioned, auditable paper trading with:
- Unique constraints to prevent duplicate signals
- Status tracking (SIGNAL -> ORDERED -> OPEN -> CLOSED)
- Version tracking for reproducibility
- Price fill and P&L calculation
"""

import os
import sqlite3
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, List, Any
from contextlib import contextmanager

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent / "paper_trading.db"


def get_db_path() -> Path:
    """Get paper trading database path from env or default."""
    return Path(os.getenv("PAPER_TRADING_DB", DEFAULT_DB_PATH))


@contextmanager
def get_connection():
    """Get database connection with row factory."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize paper trading database schema."""
    with get_connection() as conn:
        conn.executescript("""
            -- Paper trading signals table
            CREATE TABLE IF NOT EXISTS paper_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Signal identification (unique key)
                signal_date DATE NOT NULL,
                symbol TEXT NOT NULL,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                strategy_version TEXT NOT NULL,

                -- Version tracking (critical for reproducibility)
                model_main TEXT,
                model_helper TEXT,
                prompt_version TEXT,
                code_commit TEXT,
                transcript_hash TEXT,  -- MD5 of transcript content

                -- Analysis output
                trade_long BOOLEAN NOT NULL DEFAULT 0,
                trade_long_tier TEXT,  -- D7_CORE, D6_STRICT, or NULL
                direction_score INTEGER,
                confidence REAL,
                prediction TEXT,  -- UP/DOWN
                risk_code TEXT,
                long_block_reason TEXT,

                -- Market data at signal time
                eps_surprise REAL,
                earnings_day_return REAL,
                pre_earnings_5d_return REAL,

                -- LongEligible breakdown
                computed_positives INTEGER,
                computed_vetoes INTEGER,
                priced_in_risk TEXT,

                -- Execution tracking
                status TEXT NOT NULL DEFAULT 'SIGNAL',  -- SIGNAL, ORDERED, OPEN, CLOSED, CANCELLED, ERROR
                entry_date DATE,
                entry_price REAL,
                exit_date DATE,
                exit_price REAL,
                holding_days INTEGER,

                -- P&L
                gross_return_pct REAL,
                net_return_pct REAL,  -- After slippage/cost

                -- Metadata
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_msg TEXT,
                notes TEXT,

                -- Raw JSON for full audit trail
                raw_result_json TEXT,

                -- Unique constraint: one signal per symbol/quarter/strategy per day
                UNIQUE(signal_date, symbol, year, quarter, strategy_version)
            );

            -- Index for common queries
            CREATE INDEX IF NOT EXISTS idx_paper_signals_date ON paper_signals(signal_date);
            CREATE INDEX IF NOT EXISTS idx_paper_signals_status ON paper_signals(status);
            CREATE INDEX IF NOT EXISTS idx_paper_signals_trade_long ON paper_signals(trade_long);
            CREATE INDEX IF NOT EXISTS idx_paper_signals_symbol ON paper_signals(symbol);

            -- Daily run log for auditing
            CREATE TABLE IF NOT EXISTS paper_run_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date DATE NOT NULL,
                run_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy_version TEXT NOT NULL,
                symbols_processed INTEGER,
                signals_generated INTEGER,
                trade_long_count INTEGER,
                errors_count INTEGER,
                duration_seconds REAL,
                env_snapshot TEXT,  -- JSON
                notes TEXT,

                UNIQUE(run_date, strategy_version)
            );

            -- Price history for paper fills
            CREATE TABLE IF NOT EXISTS paper_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                source TEXT,  -- e.g., 'fmp', 'yahoo', 'manual'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                UNIQUE(symbol, date)
            );
            CREATE INDEX IF NOT EXISTS idx_paper_prices_symbol_date ON paper_prices(symbol, date);
        """)
        print(f"Paper trading database initialized: {get_db_path()}")


def insert_signal(
    signal_date: date,
    symbol: str,
    year: int,
    quarter: int,
    strategy_version: str,
    result: Dict[str, Any],
    transcript_content: Optional[str] = None,
    model_main: Optional[str] = None,
    model_helper: Optional[str] = None,
    code_commit: Optional[str] = None,
) -> int:
    """Insert a new paper trading signal. Returns signal ID."""

    # Calculate transcript hash for reproducibility
    transcript_hash = None
    if transcript_content:
        transcript_hash = hashlib.md5(transcript_content.encode()).hexdigest()

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO paper_signals (
                signal_date, symbol, year, quarter, strategy_version,
                model_main, model_helper, code_commit, transcript_hash,
                trade_long, trade_long_tier, direction_score, confidence, prediction,
                risk_code, long_block_reason,
                eps_surprise, earnings_day_return, pre_earnings_5d_return,
                computed_positives, computed_vetoes, priced_in_risk,
                status, raw_result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_date.isoformat() if isinstance(signal_date, date) else signal_date,
            symbol, year, quarter, strategy_version,
            model_main, model_helper, code_commit, transcript_hash,
            result.get("trade_long", False),
            result.get("trade_long_tier"),
            result.get("le_DirectionScore") or result.get("direction_score"),
            result.get("confidence"),
            result.get("prediction"),
            result.get("risk_code"),
            result.get("long_block_reason"),
            result.get("eps_surprise"),
            result.get("earnings_day_return"),
            result.get("pre_earnings_5d_return"),
            result.get("computed_positives"),
            result.get("computed_vetoes"),
            result.get("le_PricedInRisk"),
            "SIGNAL",
            str(result) if result else None,
        ))
        return cursor.lastrowid


def update_signal_status(signal_id: int, status: str, **kwargs):
    """Update signal status and optional fields."""
    with get_connection() as conn:
        # Build dynamic update
        fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
        values = [status]

        for key in ["entry_date", "entry_price", "exit_date", "exit_price",
                    "holding_days", "gross_return_pct", "net_return_pct", "error_msg", "notes"]:
            if key in kwargs and kwargs[key] is not None:
                fields.append(f"{key} = ?")
                val = kwargs[key]
                if isinstance(val, date):
                    val = val.isoformat()
                values.append(val)

        values.append(signal_id)
        conn.execute(f"UPDATE paper_signals SET {', '.join(fields)} WHERE id = ?", values)


def fill_entry_price(signal_id: int, entry_date: date, entry_price: float, slippage_bps: float = 10):
    """Fill entry price for a signal."""
    # Apply slippage (assume buying, so price goes up slightly)
    adjusted_price = entry_price * (1 + slippage_bps / 10000)
    update_signal_status(
        signal_id,
        status="OPEN",
        entry_date=entry_date,
        entry_price=adjusted_price,
    )


def fill_exit_price(signal_id: int, exit_date: date, exit_price: float, slippage_bps: float = 10):
    """Fill exit price and calculate returns."""
    with get_connection() as conn:
        row = conn.execute("SELECT entry_price, entry_date FROM paper_signals WHERE id = ?", (signal_id,)).fetchone()
        if not row or not row["entry_price"]:
            raise ValueError(f"Signal {signal_id} has no entry price")

        entry_price = row["entry_price"]
        entry_date_str = row["entry_date"]

        # Apply slippage (assume selling, so price goes down slightly)
        adjusted_exit = exit_price * (1 - slippage_bps / 10000)

        # Calculate returns
        gross_return = (exit_price / entry_price - 1) * 100
        net_return = (adjusted_exit / entry_price - 1) * 100

        # Calculate holding days
        entry_dt = datetime.strptime(entry_date_str, "%Y-%m-%d").date()
        holding_days = (exit_date - entry_dt).days

        update_signal_status(
            signal_id,
            status="CLOSED",
            exit_date=exit_date,
            exit_price=adjusted_exit,
            holding_days=holding_days,
            gross_return_pct=gross_return,
            net_return_pct=net_return,
        )


def get_open_positions() -> List[Dict]:
    """Get all open positions."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM paper_signals
            WHERE status = 'OPEN'
            ORDER BY entry_date
        """).fetchall()
        return [dict(row) for row in rows]


def get_signals_for_date(signal_date: date, trade_long_only: bool = True) -> List[Dict]:
    """Get signals for a specific date."""
    with get_connection() as conn:
        query = "SELECT * FROM paper_signals WHERE signal_date = ?"
        params = [signal_date.isoformat() if isinstance(signal_date, date) else signal_date]

        if trade_long_only:
            query += " AND trade_long = 1"

        query += " ORDER BY direction_score DESC"

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_closed_trades(limit: int = 100) -> List[Dict]:
    """Get recent closed trades."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM paper_signals
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [dict(row) for row in rows]


def get_performance_summary() -> Dict:
    """Get overall paper trading performance summary."""
    with get_connection() as conn:
        # Closed trades stats
        closed = conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN net_return_pct > 0 THEN 1 ELSE 0 END) as wins,
                AVG(net_return_pct) as avg_return,
                SUM(net_return_pct) as total_return,
                MIN(net_return_pct) as worst_loss,
                MAX(net_return_pct) as best_gain,
                AVG(holding_days) as avg_holding
            FROM paper_signals
            WHERE status = 'CLOSED'
        """).fetchone()

        # Open positions
        open_count = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status = 'OPEN'").fetchone()[0]

        # Pending signals
        pending = conn.execute("SELECT COUNT(*) FROM paper_signals WHERE status = 'SIGNAL' AND trade_long = 1").fetchone()[0]

        total = closed["total_trades"] or 0
        wins = closed["wins"] or 0

        return {
            "closed_trades": total,
            "win_rate": wins / total if total > 0 else 0,
            "avg_return_pct": closed["avg_return"] or 0,
            "total_return_pct": closed["total_return"] or 0,
            "worst_loss_pct": closed["worst_loss"] or 0,
            "best_gain_pct": closed["best_gain"] or 0,
            "avg_holding_days": closed["avg_holding"] or 0,
            "open_positions": open_count,
            "pending_signals": pending,
        }


def log_daily_run(
    run_date: date,
    strategy_version: str,
    symbols_processed: int,
    signals_generated: int,
    trade_long_count: int,
    errors_count: int,
    duration_seconds: float,
    env_snapshot: Optional[Dict] = None,
):
    """Log a daily paper trading run."""
    import json
    with get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO paper_run_log (
                run_date, strategy_version, symbols_processed, signals_generated,
                trade_long_count, errors_count, duration_seconds, env_snapshot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_date.isoformat() if isinstance(run_date, date) else run_date,
            strategy_version,
            symbols_processed,
            signals_generated,
            trade_long_count,
            errors_count,
            duration_seconds,
            json.dumps(env_snapshot) if env_snapshot else None,
        ))


if __name__ == "__main__":
    # Initialize database if run directly
    init_db()

    # Print summary
    summary = get_performance_summary()
    print("\nPaper Trading Summary:")
    print(f"  Closed trades: {summary['closed_trades']}")
    print(f"  Win rate: {summary['win_rate']*100:.1f}%")
    print(f"  Avg return: {summary['avg_return_pct']:.2f}%")
    print(f"  Open positions: {summary['open_positions']}")
    print(f"  Pending signals: {summary['pending_signals']}")
