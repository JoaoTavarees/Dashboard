"""Recalculates the workbook in LibreOffice for every scenario / loan switch and compares it
with the Python reference model. Exits non-zero on any mismatch or formula error."""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from openpyxl import load_workbook

import model

SRC = Path(__file__).resolve().parents[1] / "deliverables" / "Luca_Step1_Financial_Model.xlsx"
MC = [chr(ord("C") + i) for i in range(12)]
P_ROWS = dict(revenue=8, gm=10, ebitda=24, ebit=27)
CF_CLOSE, CF_CHK = 26, 44
RW_CASH = dict(nl=16, wl=32)


def recalc(sel, loan_on, tmp):
    wb = load_workbook(SRC)
    wb["Assumptions"]["C6"] = sel
    wb["Assumptions"]["C7"] = loan_on
    src = Path(tmp) / f"in_{sel}_{loan_on}.xlsx"
    wb.save(src)
    out = Path(tmp) / "out"
    subprocess.run(["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", str(out), str(src)],
                   check=True, capture_output=True)
    return load_workbook(out / src.name, data_only=True)


def main():
    errors = 0
    tmp = tempfile.mkdtemp()
    try:
        for sel, name in enumerate(["Base", "Downside", "Stress"], start=1):
            ref = model.run(model.SCENARIOS[name])
            for loan_on in (1, 0):
                wb = recalc(sel, loan_on, tmp)
                for ws in wb.worksheets:
                    for row in ws.iter_rows():
                        for c in row:
                            if isinstance(c.value, str) and re.match(r"^(#[A-Z/0-9]+[!?]|#N/A|Err:)", c.value):
                                print(f"  formula error {ws.title}!{c.coordinate}: {c.value}")
                                errors += 1
                P, C, R = wb["P&L"], wb["Cash Flow"], wb["Runway"]
                checks = []
                for key, row in P_ROWS.items():
                    checks += [(f"P&L {key} M{m+1}", P[f"{col}{row}"].value, ref[key][m]) for m, col in enumerate(MC)]
                ni_key = "ni_wl" if loan_on else "ni_nl"
                checks += [(f"P&L NI M{m+1}", P[f"{col}31"].value, ref[ni_key][m]) for m, col in enumerate(MC)]
                cash_key = "cash_wl" if loan_on else "cash_nl"
                checks += [(f"CF close M{m+1}", C[f"{col}{CF_CLOSE}"].value, ref[cash_key][m])
                           for m, col in enumerate(MC)]
                for k in ("nl", "wl"):
                    checks += [(f"Runway {k} M{m+1}", R[f"{col}{RW_CASH[k]}"].value, ref[f"cash_{k}"][m])
                               for m, col in enumerate(MC)]
                bad = [(lbl, a, b) for lbl, a, b in checks if a is None or abs(a - b) > 0.5]
                chk = [C[f"{col}{CF_CHK}"].value for col in MC] + [R["C54"].value]
                status = "OK" if not bad and all(abs(x or 0) < 0.01 for x in chk) else "FAIL"
                print(f"{name:9s} loan={loan_on}: {len(checks)} values compared, {len(bad)} mismatches, "
                      f"BS check max={max(abs(x or 0) for x in chk):.2f}, runway NL={R['C48'].value} "
                      f"WL={R['D48'].value} -> {status}")
                for b in bad[:5]:
                    print("   ", b)
                errors += len(bad) + (status == "FAIL")
    finally:
        shutil.rmtree(tmp)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
