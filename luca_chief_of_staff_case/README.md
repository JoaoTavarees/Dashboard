# Luca – Chief of Staff hiring case (submission)

All data is fictional, as stated in the case brief.

## Deliverables (`deliverables/`)

| Step | File | Import to Google |
|---|---|---|
| 1 | `Luca_Step1_Financial_Model.xlsx` – 12-month model, fully formula-driven | Drive → New → File upload → Open with Google Sheets |
| 2 | `Luca_Step2_Financial_Operations_Summary.pptx` (+ PDF) – 5-slide board summary | Drive → Open with Google Slides |
| 3 | `Luca_Step3_Execution_Content_Operations.pptx` (+ PDF) – 4-slide execution plan | Drive → Open with Google Slides |
| 3 (support) | `Luca_Content_Pipeline_Master_Tracker.xlsx` – working tracker template with the Week 1 snapshot | Open with Google Sheets |

Submit to: Frederico@lucaedu.com

## Model structure (Step 1)

`Cover` (KPIs) → `Assumptions` (every input, scenario selector `C6`, loan switch `C7`) → `Headcount` (25 named hires) →
`P&L` → `Cash Flow` (indirect method + balance-sheet check) → `Runway` (with vs. without loan, side by side) →
`Loan` (24-month waterfall in EUR and USD, plus FX sensitivity) → `Scenario Summary`.

Assumptions not given in the brief are labelled on the `Assumptions` tab: opening cash $3.0M, 45-day DSO, 30-day payables,
30% Mexican income tax paid one month in arrears, and $1.5k equipment capex per hire. Overhead grows linearly, +15% over 12 months.

## Headline results (Base case)

- Monthly revenue grows from $1.5M to $2.85M; $25.3M over 12 months. Gross margin goes from 65% to 60%.
- EBITDA is $13.6M (54% margin). Free cash flow is positive every month, and cash grows from $3.0M to $11.3M without the loan.
- Runway is self-funding with or without the loan. In the Stress case (flat revenue, gross margin falling to 55%, DSO rising to
  90 days, plus an illustrative $1.0M/month for the existing cost base), runway is 7.4 months without the loan and 8.3 with it.
- Recommendation: don't draw the €500k term loan. Negotiate an undrawn USD/MXN standby facility instead.

## Rebuild & verify

```bash
pip install -r build/requirements.txt          # LibreOffice is needed only for verification
cd build
python3 build_financial_model.py && python3 build_tracker.py
python3 build_step2_deck.py && python3 build_step3_deck.py
python3 verify_model.py   # recalculates the workbook in LibreOffice for all 3 scenarios × loan on/off
```

`model.py` is a Python reference implementation of the spreadsheet logic. `verify_model.py` checks that the recalculated
workbook matches it to within $0.50 on every key line, that the balance sheet balances, and that no cell has a formula error.
