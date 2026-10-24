# Data Sources Register

| ID | Source | File in data/raw | Covers | Status |
|----|--------|------------------|--------|--------|
| S1 | SBP policy rate | sbp_policy_rate.csv | May 2015 - Apr 2026, 39 rate changes | Received |
| S2 | PAMA history | pama_history_2007_2026.xlsx | Jul 2007 - FY2025-26, by model | Received, needs parsing check |
| S3 | PAMA latest | pama_sales_2026_09.pdf | Jul-Sep 2026 vs 2025 | Pending copy (cross-check only) |
| S4 | PBS CPI review | pbs_cpi_review_2026_08.pdf | CPI/SPI/WPI, Nov 2024 - Aug 2026 | Pending copy (needed at CPI step) |
| S5 | Fuel prices | - | Petrol/HSD | Open (manual entry) |
| S6 | PBS CPI history | - | Long CPI series | Open |
| S7 | Tax/duty events | - | Budget 2026-27 SROs | Open (manual log) |
| S8 | Financing rate (KIBOR or lending rate) | - | Needed per team brief | Open |

## Notes
- SBP file records rate changes only; forward-fill to monthly in Stage 2.
- PAMA covers members only. Alto is used as a demand proxy.
- PBS base year is 2015-16.
- Public data is monthly and national. Variant, color and regional forecasts need Suzuki internal data.