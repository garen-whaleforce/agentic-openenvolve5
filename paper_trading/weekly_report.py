#!/usr/bin/env python3
"""
Paper Trading Weekly Report

Generates comprehensive weekly report with:
- Performance summary
- Win rate and Wilson CI
- Tier breakdown
- Drift indicators
- Open positions mark-to-market

Usage:
    python weekly_report.py

    # Output to specific file
    python weekly_report.py --output report.md
"""

import argparse
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List

# Add project paths
script_dir = Path(__file__).parent
project_dir = script_dir.parent
sys.path.insert(0, str(project_dir))

from dotenv import load_dotenv
load_dotenv(project_dir / ".env")

import paper_trading_db as db


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 0.0
    try:
        from scipy import stats
        p = successes / n
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator
        margin = z * ((p * (1-p) + z**2 / (4*n)) / n) ** 0.5 / denominator
        return max(0, center - margin), min(1, center + margin), (min(1, center + margin) - max(0, center - margin)) / 2
    except ImportError:
        # Fallback without scipy
        p = successes / n
        return p - 0.1, p + 0.1, 0.1


def get_weekly_signals(weeks_back: int = 1) -> List[Dict]:
    """Get signals from the past N weeks."""
    cutoff = date.today() - timedelta(weeks=weeks_back)

    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM paper_signals
            WHERE signal_date >= ?
            ORDER BY signal_date DESC
        """, (cutoff.isoformat(),)).fetchall()
        return [dict(row) for row in rows]


def get_all_closed_trades() -> List[Dict]:
    """Get all closed trades."""
    with db.get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM paper_signals
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC
        """).fetchall()
        return [dict(row) for row in rows]


def get_open_positions() -> List[Dict]:
    """Get all open positions."""
    return db.get_open_positions()


def generate_report() -> str:
    """Generate the weekly report."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    # Fetch data
    all_closed = get_all_closed_trades()
    open_positions = get_open_positions()
    recent_signals = get_weekly_signals(weeks_back=4)
    summary = db.get_performance_summary()

    # Build report
    lines = [
        f"# Paper Trading Weekly Report",
        f"",
        f"**Generated**: {today.strftime('%Y-%m-%d %H:%M')}",
        f"**Week**: {week_start} to {week_end}",
        f"",
        f"---",
        f"",
        f"## 1. Executive Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Closed Trades | {summary['closed_trades']} |",
        f"| Open Positions | {summary['open_positions']} |",
        f"| Pending Signals | {summary['pending_signals']} |",
    ]

    if summary['closed_trades'] > 0:
        lines.extend([
            f"| **Win Rate** | **{summary['win_rate']*100:.1f}%** |",
            f"| **Avg Return** | **{summary['avg_return_pct']:+.2f}%** |",
            f"| Total Return | {summary['total_return_pct']:+.2f}% |",
            f"| Best Trade | {summary['best_gain_pct']:+.2f}% |",
            f"| Worst Trade | {summary['worst_loss_pct']:+.2f}% |",
        ])

        # Wilson CI
        wins = int(summary['win_rate'] * summary['closed_trades'])
        n = summary['closed_trades']
        lower, upper, hw = wilson_ci(wins, n)
        lines.extend([
            f"",
            f"### Wilson 95% Confidence Interval",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {summary['win_rate']*100:.1f}% ({wins}/{n}) |",
            f"| CI Lower | {lower*100:.1f}% |",
            f"| CI Upper | {upper*100:.1f}% |",
            f"| Half-width | {hw*100:.1f}% |",
        ])

    # Tier breakdown
    if all_closed:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 2. Tier Breakdown",
            f"",
            f"| Tier | Trades | Wins | Win Rate | Avg Return |",
            f"|------|--------|------|----------|------------|",
        ])

        tier_stats = {}
        for trade in all_closed:
            tier = trade.get("trade_long_tier") or "Unknown"
            if tier not in tier_stats:
                tier_stats[tier] = {"trades": 0, "wins": 0, "returns": []}
            tier_stats[tier]["trades"] += 1
            ret = trade.get("net_return_pct") or 0
            tier_stats[tier]["returns"].append(ret)
            if ret > 0:
                tier_stats[tier]["wins"] += 1

        for tier, stats in sorted(tier_stats.items()):
            trades = stats["trades"]
            wins = stats["wins"]
            win_rate = wins / trades * 100 if trades > 0 else 0
            avg_ret = sum(stats["returns"]) / len(stats["returns"]) if stats["returns"] else 0
            lines.append(f"| {tier} | {trades} | {wins} | {win_rate:.1f}% | {avg_ret:+.2f}% |")

    # Open positions
    if open_positions:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 3. Open Positions ({len(open_positions)})",
            f"",
            f"| Symbol | Tier | Entry Date | Entry Price | Days Held |",
            f"|--------|------|------------|-------------|-----------|",
        ])

        for pos in open_positions:
            entry_date = pos.get("entry_date", "")
            if entry_date:
                days = (today - datetime.strptime(entry_date, "%Y-%m-%d").date()).days
            else:
                days = 0
            lines.append(
                f"| {pos['symbol']} | {pos.get('trade_long_tier') or 'N/A'} | "
                f"{entry_date} | ${pos.get('entry_price', 0):.2f} | {days} |"
            )

    # Recent signals (last 4 weeks)
    recent_trade_long = [s for s in recent_signals if s.get("trade_long")]
    if recent_trade_long:
        lines.extend([
            f"",
            f"---",
            f"",
            f"## 4. Recent Signals (4 weeks)",
            f"",
            f"| Date | Symbol | Tier | Direction | Status |",
            f"|------|--------|------|-----------|--------|",
        ])

        for sig in recent_trade_long[:20]:  # Limit to 20
            lines.append(
                f"| {sig['signal_date']} | {sig['symbol']} | "
                f"{sig.get('trade_long_tier') or 'N/A'} | "
                f"{sig.get('direction_score') or '?'} | {sig['status']} |"
            )

    # Drift indicators
    lines.extend([
        f"",
        f"---",
        f"",
        f"## 5. Drift Indicators",
        f"",
    ])

    # Recent win rate (last 20 trades)
    recent_closed = all_closed[:20] if len(all_closed) >= 20 else all_closed
    if recent_closed:
        recent_wins = sum(1 for t in recent_closed if (t.get("net_return_pct") or 0) > 0)
        recent_n = len(recent_closed)
        recent_wr = recent_wins / recent_n * 100 if recent_n > 0 else 0

        lines.extend([
            f"### Recent Performance (Last {recent_n} trades)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Win Rate | {recent_wr:.1f}% ({recent_wins}/{recent_n}) |",
        ])

        recent_returns = [t.get("net_return_pct") or 0 for t in recent_closed]
        recent_avg = sum(recent_returns) / len(recent_returns) if recent_returns else 0
        lines.append(f"| Avg Return | {recent_avg:+.2f}% |")

        # Compare to overall
        if summary['closed_trades'] > recent_n:
            lines.extend([
                f"| Overall Win Rate | {summary['win_rate']*100:.1f}% |",
                f"| Drift (recent - overall) | {recent_wr - summary['win_rate']*100:+.1f}% |",
            ])

    # Risk code distribution
    if recent_signals:
        risk_dist = {}
        for sig in recent_signals:
            risk = sig.get("risk_code") or "unknown"
            risk_dist[risk] = risk_dist.get(risk, 0) + 1

        lines.extend([
            f"",
            f"### Risk Code Distribution (Last 4 weeks)",
            f"",
            f"| Risk Code | Count | % |",
            f"|-----------|-------|---|",
        ])
        total = len(recent_signals)
        for risk, count in sorted(risk_dist.items(), key=lambda x: -x[1]):
            lines.append(f"| {risk} | {count} | {count/total*100:.1f}% |")

    # Footer
    lines.extend([
        f"",
        f"---",
        f"",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Weekly Report")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    args = parser.parse_args()

    # Initialize database
    db.init_db()

    # Generate report
    report = generate_report()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
