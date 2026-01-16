# Validation 4000 Samples Supplementary Statistics

**Date**: 2026-01-02
**Source**: `validation_2017_2025_4000_20260102_201558.csv`
**Version**: v2.2 (Lookahead Assertions Enabled)

---

## 1. Headline Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 4000 |
| Success Rate | 100% (4000/4000) |
| Valid Samples | 3999 |
| Overall Accuracy | 55.0% (2201/3999) |
| **Long Trades** | **327** |
| **Long Win Rate** | **81.7% (267/327)** |
| **Long Avg Return** | **+8.46%** |
| Long Median Return | +7.47% |

### Tier Breakdown

| Tier | Trades | Win Rate | Avg Return |
|------|--------|----------|------------|
| D7_CORE | 221 | 84.2% | +9.41% |
| D6_STRICT | 106 | 76.4% | +6.47% |

---

## 2. Quarterly Distribution

### Summary
- **Min trades/quarter**: 2
- **Median trades/quarter**: 9
- **P75 trades/quarter**: 12
- **Max trades/quarter**: 23
- **Quarters with trades**: 35/35 (complete coverage)

### Worst Quarters (>=6 trades)

| Quarter | Trades | Win Rate | Avg Return | Min Return |
|---------|--------|----------|------------|------------|
| 2022Q1 | 23 | 43.5% | -2.38% | -21.49% |
| 2019Q2 | 8 | 50.0% | +0.08% | -19.48% |
| 2024Q4 | 6 | 50.0% | -0.62% | -13.90% |
| 2022Q4 | 9 | 66.7% | +5.59% | -5.22% |
| 2021Q3 | 13 | 69.2% | +4.11% | -12.25% |

---

## 3. Loss Tail Statistics

### Win/Loss Breakdown
| Metric | Value |
|--------|-------|
| Winners | 267 (81.7%) |
| Losers | 60 (18.3%) |

### Return Distribution
| Metric | Value |
|--------|-------|
| Avg Win | +11.65% |
| Avg Loss | -5.78% |
| Overall Avg | +8.46% |
| Median | +7.47% |
| Std Dev | 11.45% |

### Tail Risk
| Percentile | Return |
|------------|--------|
| P05 | -7.22% |
| P10 | -4.09% |
| P25 | +1.32% |
| Min (worst) | -21.49% |
| Max (best) | +87.66% |

### Risk-Adjusted Metrics
| Metric | Value |
|--------|-------|
| Sum of Wins | +3111.87% |
| Sum of Losses | -346.55% |
| **Profit Factor** | **8.98** |
| Net Return | +2765.33% |

---

## 4. Wilson Confidence Interval Analysis

### Current (n=327)
| Metric | Value |
|--------|-------|
| Win Rate | 81.7% |
| Wilson 95% CI | [77.1%, 85.5%] |
| Half-width | 4.2% |

### CI Projection (assuming 81.7% win rate maintained)

| Trades | Lower Bound | Upper Bound | Half-width |
|--------|-------------|-------------|------------|
| 327 | 77.1% | 85.5% | 4.2% |
| 400 | 77.4% | 85.0% | 3.8% |
| 500 | 78.0% | 84.8% | 3.4% |
| 600 | 78.2% | 84.4% | 3.1% |
| 700 | 78.5% | 84.3% | 2.9% |
| 800 | 78.8% | 84.2% | 2.7% |

### D7_CORE Only (Current n=221)
| Metric | Value |
|--------|-------|
| Win Rate | 84.2% |
| Wilson 95% CI | [78.8%, 88.4%] |

### D7_CORE CI Projection (for LB>=80% target)

| D7 Trades | Lower Bound | LB>=80%? |
|-----------|-------------|----------|
| 221 | 78.8% | NO |
| 300 | 79.4% | NO |
| 350 | 79.8% | NO |
| 400 | 80.1% | **YES** |
| 500 | 80.5% | **YES** |

**Key Insight**: To achieve Wilson LB >= 80% for D7_CORE, need ~400 D7 trades (requires ~8000 samples).

---

## 5. 2019 Attribution Analysis

**Year Summary**: 28 trades, 67.9% win rate, +3.77% avg return

### By Tier

| Tier | Trades | Win Rate | Avg Return | Total Contribution |
|------|--------|----------|------------|-------------------|
| D7_CORE | 18 | 77.8% | +7.28% | +130.98% |
| D6_STRICT | 10 | 50.0% | -2.53% | -25.31% |

### Worst Sector×Tier Combinations

| Sector | Tier | Trades | Win Rate | Total | Min |
|--------|------|--------|----------|-------|-----|
| Financial Services | D6_STRICT | 2 | 0% | -16.10% | -10.67% |
| Technology | D6_STRICT | 4 | 50% | -15.98% | -10.96% |
| Energy | D7_CORE | 1 | 0% | -7.63% | -7.63% |

**Conclusion**: 2019 drag is primarily from D6_STRICT in Financial Services and Technology sectors.

---

## 6. 2022 Attribution Analysis

**Year Summary**: 50 trades, 62.0% win rate, +2.88% avg return

### By Tier

| Tier | Trades | Win Rate | Avg Return | Total Contribution |
|------|--------|----------|------------|-------------------|
| D7_CORE | 32 | 65.6% | +3.22% | +102.89% |
| D6_STRICT | 18 | 55.6% | +2.28% | +41.06% |

### Worst Sector×Tier Combinations

| Sector | Tier | Trades | Win Rate | Total | Min |
|--------|------|--------|----------|-------|-----|
| Real Estate | D7_CORE | 5 | 60% | -26.88% | -21.49% |
| Communication Services | D6_STRICT | 2 | 0% | -14.58% | -9.35% |
| Healthcare | D6_STRICT | 3 | 33% | -9.32% | -17.37% |
| Consumer Cyclical | D6_STRICT | 2 | 0% | -6.47% | -3.65% |

**Key Issue**: 2022Q1 accounts for 23 trades at 43.5% win rate, containing the worst single trade (BXP Real Estate -21.49%).

---

## 7. Worst 10 Long Trades

| Symbol | Year | Quarter | Sector | Tier | Return |
|--------|------|---------|--------|------|--------|
| BXP | 2022 | Q1 | Real Estate | D7_CORE | -21.49% |
| TTD | 2019 | Q2 | Technology | D7_CORE | -19.48% |
| MRNA | 2022 | Q1 | Healthcare | D6_STRICT | -17.37% |
| REG | 2022 | Q1 | Real Estate | D7_CORE | -16.78% |
| OXY | 2024 | Q4 | Energy | D7_CORE | -13.90% |
| ULTA | 2017 | Q1 | Consumer Cyclical | D7_CORE | -12.65% |
| ALB | 2021 | Q3 | Basic Materials | D6_STRICT | -12.25% |
| MGM | 2024 | Q4 | Consumer Cyclical | D7_CORE | -11.67% |
| FICO | 2019 | Q3 | Technology | D6_STRICT | -10.96% |
| BLK | 2019 | Q4 | Financial Services | D6_STRICT | -10.67% |

---

## 8. Key Takeaways

1. **Win Rate is Healthy at 81.7%** - Down from previous 91.7% (1951 samples), indicating v2.2 lookahead fixes are effective.

2. **D6_STRICT is the Drag Source** - 76.4% win rate vs D7_CORE's 84.2%. Consider D7-only strategy for higher confidence.

3. **2022Q1 is the Outlier Quarter** - 23 trades at 43.5% win rate, skewing 2022 results.

4. **Real Estate 2022 Cluster Risk** - Single worst trade (-21.49%) plus sector concentration in Q1 2022.

5. **Wilson CI Needs More Trades** - Current CI [77.1%, 85.5%] has 4.2% half-width. Need ~500-600 trades to narrow to ~3%.

6. **Profit Factor 8.98** - Strong risk-reward profile despite tail losses.

---

*Generated: 2026-01-02*
*Audit Version: v2.2*
