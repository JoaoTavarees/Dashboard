# Luca – Chief of Staff hiring case (submission)

All data is fictional, as stated in the case brief.

## Deliverables (`deliverables/`)

| Step | File | Import to Google |
|---|---|---|
| 1 | `Luca_Step1_Financial_Model.xlsx` – 12-month model, fully formula-driven | Drive → New → File upload → Open with Google Sheets |
| 2 | `Luca_Step2_Financial_Operations_Summary.pptx` (+ PDF) – 5-slide board summary | Drive → Open with Google Slides |
| 3 | `Luca_Step3_Execution_Content_Operations.pptx` (+ PDF) – 4-slide execution plan | Drive → Open with Google Slides |
| 3 (support) | `Luca_Content_Pipeline_Master_Tracker.xlsx` – tracker template with the Week 1 snapshot | Open with Google Sheets |

Submit to: Frederico@lucaedu.com

## Model (Step 1)

Tabs: `Summary` → `Assumptions` → `Headcount` → `P&L` → `Cash Flow` (indirect method, without and with the loan) →
`Loan` (24-month schedule in EUR and USD) → `Runway` (cash, burn and runway, with vs. without the loan).

Every input comes from the case brief, except:
- **Opening cash of $3.0M.** This is my assumption, needed to calculate the cash position and runway.
- **Hires per month by level** (10 senior, 15 junior across Months 1–6). The case asks you to recommend this.
- **Overhead grows in a straight line**, from $40k to $46k. The brief lets you choose the approach.

Income tax, depreciation, payment delays and capital expenditure are not modelled because the brief does not mention them.

## Headline results

- Monthly revenue grows from $1.5M to $2.85M, or $25.3M over 12 months. Gross margin goes from 65% to 60%.
- Operating profit is $13.6M (54% margin). Luca generates cash every month: $3.0M grows to $16.6M without the loan
  and $17.0M with it.
- Runway is not limited in either case. Recommendation: don't take the €500k loan.

## Rebuild & verify

```bash
pip install -r build/requirements.txt          # LibreOffice is needed only for verification
cd build
python3 build_financial_model.py && python3 build_tracker.py
python3 build_step2_deck.py && python3 build_step3_deck.py
python3 verify_model.py   # recalculates the workbook in LibreOffice and compares it with model.py
```
