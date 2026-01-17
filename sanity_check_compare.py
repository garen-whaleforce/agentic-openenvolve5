#!/usr/bin/env python3
"""
Sanity Check - 比對舊結果
驗證跑出來的 DirectionScore 和 confidence 是否與之前一致
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from analysis_engine import analyze_earnings_async

# 從舊驗證結果選取的樣本 (有 le_DirectionScore 的)
REFERENCE_SAMPLES = [
    {"symbol": "TYL", "year": 2022, "quarter": 1, "expected_direction": 7, "expected_confidence": 0.7},
    {"symbol": "COR", "year": 2021, "quarter": 1, "expected_direction": 6, "expected_confidence": 0.6},
    {"symbol": "CHD", "year": 2019, "quarter": 1, "expected_direction": 6, "expected_confidence": 0.6},
    {"symbol": "DVN", "year": 2018, "quarter": 3, "expected_direction": 7, "expected_confidence": 0.7},
    {"symbol": "BDX", "year": 2022, "quarter": 2, "expected_direction": 7, "expected_confidence": 0.7},
]


async def run_single_test(sample: dict):
    """Run a single test and compare with expected values."""
    start = time.time()
    symbol = sample["symbol"]
    year = sample["year"]
    quarter = sample["quarter"]
    expected_dir = sample["expected_direction"]
    expected_conf = sample["expected_confidence"]

    try:
        result = await analyze_earnings_async(
            symbol=symbol,
            year=year,
            quarter=quarter,
            skip_cache=True,
        )
        elapsed = time.time() - start

        agentic_result = result.get("agentic_result", {})
        confidence = agentic_result.get("confidence")
        long_eligible_json = agentic_result.get("long_eligible_json", {})
        direction_score = long_eligible_json.get("DirectionScore") if long_eligible_json else None
        trade_long = agentic_result.get("trade_long", False)
        trade_long_tier = agentic_result.get("trade_long_tier", "")

        # Compare with expected
        dir_match = direction_score == expected_dir if direction_score is not None else None
        conf_match = confidence == expected_conf if confidence is not None else None

        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "expected_direction": expected_dir,
            "actual_direction": direction_score,
            "direction_match": dir_match,
            "expected_confidence": expected_conf,
            "actual_confidence": confidence,
            "confidence_match": conf_match,
            "trade_long": trade_long,
            "trade_long_tier": trade_long_tier,
            "time_seconds": elapsed,
            "success": True,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "symbol": symbol,
            "year": year,
            "quarter": quarter,
            "expected_direction": expected_dir,
            "actual_direction": None,
            "direction_match": None,
            "expected_confidence": expected_conf,
            "actual_confidence": None,
            "confidence_match": None,
            "trade_long": None,
            "trade_long_tier": None,
            "time_seconds": elapsed,
            "success": False,
            "error": str(e)[:100],
        }


async def main():
    print("=" * 70)
    print("🔬 SANITY CHECK - 比對舊驗證結果")
    print("=" * 70)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📦 樣本數: {len(REFERENCE_SAMPLES)}")
    print("=" * 70)

    print("\n📋 參考樣本 (來自 validation_results.csv):")
    for s in REFERENCE_SAMPLES:
        print(f"  {s['symbol']} {s['year']}Q{s['quarter']}: Dir={s['expected_direction']}, Conf={s['expected_confidence']}")

    print("\n🚀 開始測試...\n")
    print("-" * 70)

    results = []
    total_start = time.time()

    for i, sample in enumerate(REFERENCE_SAMPLES, 1):
        symbol = sample["symbol"]
        year = sample["year"]
        quarter = sample["quarter"]

        print(f"\n[{i}/{len(REFERENCE_SAMPLES)}] {symbol} {year}Q{quarter}")
        print(f"  Expected: Dir={sample['expected_direction']}, Conf={sample['expected_confidence']}")

        result = await run_single_test(sample)
        results.append(result)

        if result["success"]:
            dir_status = "✅" if result["direction_match"] else "❌" if result["direction_match"] is False else "❓"
            conf_status = "✅" if result["confidence_match"] else "❌" if result["confidence_match"] is False else "❓"

            print(f"  Actual:   Dir={result['actual_direction']}, Conf={result['actual_confidence']}")
            print(f"  Match:    Direction {dir_status} | Confidence {conf_status}")
            print(f"  TradeLong: {result['trade_long']} ({result['trade_long_tier']})")
            print(f"  Time: {result['time_seconds']:.1f}s")
        else:
            print(f"  ❌ ERROR: {result['error']}")

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("📊 SANITY CHECK 比對結果")
    print("=" * 70)

    df = pd.DataFrame(results)
    successful = df[df["success"] == True]

    print(f"\n📈 執行統計:")
    print(f"  總樣本: {len(df)} | 成功: {len(successful)} | 失敗: {len(df) - len(successful)}")
    print(f"  總時間: {total_time:.1f}s ({total_time/len(df):.1f}s/樣本)")

    if len(successful) > 0:
        dir_matches = successful["direction_match"].sum()
        conf_matches = successful["confidence_match"].sum()
        dir_total = successful["direction_match"].notna().sum()
        conf_total = successful["confidence_match"].notna().sum()

        print(f"\n🎯 一致性比對:")
        print(f"  DirectionScore: {dir_matches}/{dir_total} 一致 ({dir_matches/dir_total*100:.0f}%)" if dir_total > 0 else "  DirectionScore: N/A")
        print(f"  Confidence: {conf_matches}/{conf_total} 一致 ({conf_matches/conf_total*100:.0f}%)" if conf_total > 0 else "  Confidence: N/A")

        # Detail table
        print("\n📋 詳細比對:")
        print(f"  {'Symbol':<8} {'Q':<8} {'Exp Dir':<8} {'Act Dir':<8} {'Match':<6} {'Exp Conf':<10} {'Act Conf':<10} {'Match':<6}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*6}")

        for _, row in successful.iterrows():
            dir_m = "✅" if row["direction_match"] else "❌" if row["direction_match"] is False else "❓"
            conf_m = "✅" if row["confidence_match"] else "❌" if row["confidence_match"] is False else "❓"
            print(f"  {row['symbol']:<8} {row['year']}Q{row['quarter']:<5} {row['expected_direction']:<8} {str(row['actual_direction']):<8} {dir_m:<6} {row['expected_confidence']:<10} {str(row['actual_confidence']):<10} {conf_m:<6}")

    print("\n" + "=" * 70)
    if len(successful) > 0:
        all_match = (dir_matches == dir_total) and (conf_matches == conf_total) if dir_total > 0 and conf_total > 0 else False
        if all_match:
            print("✅ SANITY CHECK 通過 - 結果與舊驗證一致")
        else:
            print("⚠️ SANITY CHECK 有差異 - 請檢查原因")
    else:
        print("❌ SANITY CHECK 失敗 - 無法取得結果")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
