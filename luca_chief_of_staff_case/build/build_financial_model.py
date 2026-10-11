"""Builds the lean Luca_Step1_Financial_Model.xlsx with live formulas (Google Sheets compatible)."""
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter as L

import model

OUT = Path(__file__).resolve().parents[1] / "deliverables" / "Luca_Step1_Financial_Model.xlsx"

NAVY, TEAL, GREY = "1F2A44", "0F9D8A", "6B7280"
F_TITLE = Font(name="Arial", size=16, bold=True, color=NAVY)
F_SUB = Font(name="Arial", size=10, italic=True, color=GREY)
F_HDR = Font(name="Arial", size=10, bold=True, color="FFFFFF")
F_SEC = Font(name="Arial", size=10, bold=True, color=NAVY)
F_B = Font(name="Arial", size=10, bold=True)
F_N = Font(name="Arial", size=10)
F_IN = Font(name="Arial", size=10, color="0000FF")
FILL_HDR = PatternFill("solid", fgColor=NAVY)
FILL_SEC = PatternFill("solid", fgColor="E5E7EB")
FILL_IN = PatternFill("solid", fgColor="FFF7D6")
FILL_TOT = PatternFill("solid", fgColor="F3F4F6")
THIN = Side(style="thin", color="9CA3AF")
TOP = Border(top=THIN)
TOPBOT = Border(top=THIN, bottom=Side(style="double", color="9CA3AF"))

USD = '#,##0;(#,##0);"–"'
EUR = '€#,##0;(€#,##0);"–"'
PCT = '0.0%'
INT = '0;(0);"–"'

M = model.MONTHS
MC = [L(3 + i) for i in range(M)]          # month columns C..N
FIRST, LAST, TOT = MC[0], MC[-1], L(3 + M)  # C, N, O


def style_range(ws, ref, font=None, fill=None, fmt=None, border=None, align=None):
    for row in ws[ref]:
        for c in row:
            if font: c.font = font
            if fill: c.fill = fill
            if fmt: c.number_format = fmt
            if border: c.border = border
            if align: c.alignment = align


def title(ws, text, sub):
    ws["A1"], ws["A2"] = text, sub
    ws["A1"].font, ws["A2"].font = F_TITLE, F_SUB
    ws.sheet_view.showGridLines = False


def month_header(ws, row=4, opening_label=None):
    ws.cell(row=row, column=1, value="USD")
    if opening_label:
        ws.cell(row=row, column=2, value=opening_label)
    for i, col in enumerate(MC):
        ws[f"{col}{row}"] = i + 1
        ws[f"{col}{row}"].number_format = '"Month "0'
    ws[f"{TOT}{row}"] = "12-month total"
    style_range(ws, f"A{row}:{TOT}{row}", font=F_HDR, fill=FILL_HDR, align=Alignment(horizontal="center"))
    ws.cell(row=row, column=1).alignment = Alignment(horizontal="left")
    ws.freeze_panes = ws[f"C{row + 1}"]
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 12
    for col in MC:
        ws.column_dimensions[col].width = 11.5
    ws.column_dimensions[TOT].width = 14


def section(ws, row, text):
    ws.cell(row=row, column=1, value=text)
    style_range(ws, f"A{row}:{TOT}{row}", font=F_SEC, fill=FILL_SEC)


def line(ws, row, label, fn, fmt=USD, total="sum", bold=False, border=None, opening=None, fill=None, font=None):
    """fn(col, prev_col) -> formula for that month; prev_col is None for Month 1."""
    ws.cell(row=row, column=1, value=label).font = F_B if bold else F_N
    if opening is not None:
        ws[f"B{row}"] = opening
        ws[f"B{row}"].number_format = fmt
    for i, col in enumerate(MC):
        c = ws[f"{col}{row}"]
        c.value = fn(col, MC[i - 1] if i else None)
        c.number_format = fmt
        c.font = font or (F_B if bold else F_N)
        if fill:
            c.fill = fill
    if total == "sum":
        ws[f"{TOT}{row}"] = f"=SUM({FIRST}{row}:{LAST}{row})"
    elif total == "last":
        ws[f"{TOT}{row}"] = f"={LAST}{row}"
    elif total:
        ws[f"{TOT}{row}"] = total
    ws[f"{TOT}{row}"].number_format = fmt
    ws[f"{TOT}{row}"].font = F_B
    ws[f"{TOT}{row}"].fill = FILL_TOT
    if border:
        style_range(ws, f"A{row}:{TOT}{row}", border=border)
    return row


wb = Workbook()
wb.calculation.fullCalcOnLoad = True
SM = wb.active
SM.title = "Summary"
A = wb.create_sheet("Assumptions")
H = wb.create_sheet("Headcount")
P = wb.create_sheet("P&L")
C = wb.create_sheet("Cash Flow")
LN = wb.create_sheet("Loan")
R = wb.create_sheet("Runway")

# ---------------------------------------------------------------- Assumptions
title(A, "Assumptions",
      "Blue cells on yellow are inputs; everything else in the workbook is a formula. "
      "All values come from the case brief except where the Source column says otherwise.")
A.column_dimensions["A"].width = 3
A.column_dimensions["B"].width = 44
A.column_dimensions["C"].width = 14
A.column_dimensions["D"].width = 80
ref = {}


def a_hdr(row, text):
    A[f"B{row}"], A[f"C{row}"], A[f"D{row}"] = text, "Value", "Source"
    style_range(A, f"B{row}:D{row}", font=F_HDR, fill=FILL_HDR)


def a_in(row, key, label, value, fmt, src="Case brief."):
    A[f"B{row}"] = label
    c = A[f"C{row}"]
    c.value, c.font, c.fill, c.number_format = value, F_IN, FILL_IN, fmt
    A[f"D{row}"] = src
    A[f"D{row}"].font = F_SUB
    ref[key] = f"Assumptions!$C${row}"


a_hdr(4, "Revenue")
a_in(5, "rev0", "Starting monthly revenue (Month 1)", model.START_REVENUE, USD)
a_in(6, "g", "Growth, month over month (compound)", model.GROWTH, PCT)
a_in(7, "gm1", "Gross margin, Month 1", model.GM_START, PCT)
a_in(8, "gm12", "Gross margin, Month 12", model.GM_END, PCT,
     "Case brief. Compresses in equal steps between Month 1 and Month 12 (about −0.45 points a month).")

a_hdr(10, "Hiring")
a_in(11, "jr", "Junior salary (per month)", model.JUNIOR, USD)
a_in(12, "sr", "Senior salary (per month)", model.SENIOR, USD)
a_in(13, "hire_cost", "Hiring cost (one-off, per hire)", model.HIRING_COST, USD,
     "Case brief. Booked in the month the person starts.")
a_in(14, "ben", "Benefits (% of total salary)", model.BENEFITS, PCT)
A["B15"] = "The 25 hires and their start months are set on the Headcount tab."
A["B15"].font = F_SUB

a_hdr(17, "Operating expenses")
a_in(18, "ovh0", "Overhead, Month 1 (per month)", model.OVERHEAD, USD)
a_in(19, "ovh_g", "Overhead growth over 12 months", model.OVERHEAD_GROWTH, PCT,
     "Case brief allows linear or another approach. Chosen: linear, so it rises by the same amount each month, "
     "from $40,000 in Month 1 to $46,000 in Month 12.")
a_in(20, "ga", "G&A / Legal (per month, fixed)", model.GA, USD)

a_hdr(22, "Cash")
a_in(23, "cash0", "Opening cash balance (start of Month 1)", model.OPENING_CASH, USD,
     "MY ASSUMPTION (not in brief): needed to calculate cash position and runway. Change it freely.")

a_hdr(25, "Loan scenario")
a_in(26, "loan", "Loan amount (EUR)", model.LOAN_EUR, EUR, "Case brief. Assumed received in full in Month 1.")
a_in(27, "rate", "Interest rate (annual)", model.LOAN_RATE, PCT, "Case brief. 0.5% per month.")
a_in(28, "term", "Term (months)", model.LOAN_TERM, "0")
a_in(29, "io", "Interest-only months", model.LOAN_IO, "0",
     "Case brief. Then equal monthly payments over the remaining 18 months.")
a_in(30, "fx", "FX rate (USD per EUR)", model.FX, "0.00")

A["B32"] = "Not modelled because the brief does not mention them:"
A["B32"].font = F_SEC
A["B33"] = ("income tax, depreciation, customer payment delays (receivables), supplier payment terms and "
            "capital expenditure. Revenue is treated as collected, and costs as paid, in the month they occur.")
A["B33"].font = F_SUB

# ---------------------------------------------------------------- Headcount
title(H, "Headcount plan & hiring costs",
      "Enter new senior and junior hires per month (blue cells). The case requires 25 hires within Months 1–6.")
month_header(H, 4)
h = {}
r = 6
section(H, r, "New hires (input)")
for key, label, row_vals in [("new_sr", "New senior hires", [model.HIRES.get(m, (0, 0))[0] for m in range(1, 13)]),
                             ("new_jr", "New junior hires", [model.HIRES.get(m, (0, 0))[1] for m in range(1, 13)])]:
    r += 1
    h[key] = line(H, r, label, lambda c, pc, v=row_vals: v[MC.index(c)], INT, fill=FILL_IN, font=F_IN)
h["new"] = line(H, r + 1, "Total new hires", lambda c, pc: f"={c}{h['new_sr']}+{c}{h['new_jr']}", INT, bold=True,
                border=TOP)
r = h["new"] + 2
section(H, r, "Headcount at end of month")
h["hc_sr"] = line(H, r + 1, "Senior", lambda c, pc: f"=SUM(${FIRST}{h['new_sr']}:{c}{h['new_sr']})", INT, total="last")
h["hc_jr"] = line(H, r + 2, "Junior", lambda c, pc: f"=SUM(${FIRST}{h['new_jr']}:{c}{h['new_jr']})", INT, total="last")
h["hc"] = line(H, r + 3, "Total headcount (new hires)", lambda c, pc: f"={c}{r + 1}+{c}{r + 2}", INT, total="last",
               bold=True, border=TOP)
r = h["hc"] + 2
section(H, r, "People costs")
h["sal"] = line(H, r + 1, "Salaries",
                lambda c, pc: f"={c}{h['hc_sr']}*{ref['sr']}+{c}{h['hc_jr']}*{ref['jr']}")
h["ben"] = line(H, r + 2, "Benefits (15% of salary)", lambda c, pc: f"={c}{r + 1}*{ref['ben']}")
h["hire"] = line(H, r + 3, "Hiring costs ($6,000 per new hire)", lambda c, pc: f"={c}{h['new']}*{ref['hire_cost']}")
h["people"] = line(H, r + 4, "Total people cost", lambda c, pc: f"=SUM({c}{r + 1}:{c}{r + 3})", bold=True,
                   border=TOP)
H[f"A{h['people'] + 2}"] = "Check: total hires in Months 1–6 (must be 25)"
H[f"C{h['people'] + 2}"] = f"=SUM({FIRST}{h['new']}:{MC[5]}{h['new']})"
H[f"D{h['people'] + 2}"] = f'=IF(C{h["people"] + 2}=25,"OK","Not 25 – adjust inputs")'
H[f"C{h['people'] + 2}"].font = H[f"D{h['people'] + 2}"].font = F_B

# ---------------------------------------------------------------- P&L
title(P, "Profit & loss (monthly)", "Interest is shown separately so profit can be read with and without the loan.")
month_header(P, 4)
p = {}
r = 6
section(P, r, "Revenue & gross margin")
p["rev"] = line(P, r + 1, "Revenue", lambda c, pc: f"={ref['rev0']}*(1+{ref['g']})^({c}$4-1)", bold=True)
p["gm"] = line(P, r + 2, "Gross margin %", lambda c, pc: f"={ref['gm1']}+({ref['gm12']}-{ref['gm1']})*({c}$4-1)/11",
               PCT, total=f"={TOT}{r + 4}/{TOT}{r + 1}")
p["cogs"] = line(P, r + 3, "Cost of revenue", lambda c, pc: f"=-{c}{r + 1}*(1-{c}{r + 2})")
p["gp"] = line(P, r + 4, "Gross profit", lambda c, pc: f"={c}{r + 1}+{c}{r + 3}", bold=True, border=TOP)
r = p["gp"] + 2
section(P, r, "Operating expenses")
p["sal"] = line(P, r + 1, "Salaries", lambda c, pc: f"=-Headcount!{c}{h['sal']}")
p["ben"] = line(P, r + 2, "Benefits", lambda c, pc: f"=-Headcount!{c}{h['ben']}")
p["hire"] = line(P, r + 3, "Hiring costs", lambda c, pc: f"=-Headcount!{c}{h['hire']}")
p["ovh"] = line(P, r + 4, "Overhead", lambda c, pc: f"=-{ref['ovh0']}*(1+{ref['ovh_g']}*({c}$4-1)/11)")
p["ga"] = line(P, r + 5, "G&A / Legal", lambda c, pc: f"=-{ref['ga']}")
p["opex"] = line(P, r + 6, "Total operating expenses", lambda c, pc: f"=SUM({c}{r + 1}:{c}{r + 5})", bold=True,
                 border=TOP)
r = p["opex"] + 2
p["op"] = line(P, r, "Operating profit", lambda c, pc: f"={c}{p['gp']}+{c}{p['opex']}", bold=True, border=TOP)
p["op_m"] = line(P, r + 1, "Operating margin %", lambda c, pc: f"={c}{r}/{c}{p['rev']}", PCT,
                 total=f"={TOT}{r}/{TOT}{p['rev']}")
p["np_nl"] = line(P, r + 3, "Net profit – WITHOUT loan", lambda c, pc: f"={c}{r}", bold=True)
p["int"] = line(P, r + 4, "Loan interest", lambda c, pc: f"=-INDEX(Loan!$J$6:$J$29,{c}$4)")
p["np_wl"] = line(P, r + 5, "Net profit – WITH loan", lambda c, pc: f"={c}{r}+{c}{r + 4}", bold=True,
                  border=TOPBOT)

# ---------------------------------------------------------------- Loan
title(LN, "€500k loan – repayment schedule (24 months)",
      "6% a year = 0.5% a month. Months 1–6: interest only. Months 7–24: equal monthly payments. "
      "Converted to USD at the Assumptions FX rate.")
heads = ["Month", "Phase", "", "Opening balance (€)", "Interest (€)", "Principal (€)", "Total payment (€)",
         "Closing balance (€)", "Principal ($)", "Interest ($)", "Total payment ($)", "Closing balance ($)"]
for j, t in enumerate(heads):
    LN.cell(row=5, column=1 + j, value=t)
style_range(LN, "A5:L5", font=F_HDR, fill=FILL_HDR, align=Alignment(horizontal="center", wrap_text=True))
LN.row_dimensions[5].height = 30
for j, w in enumerate([8, 14, 2, 17, 13, 13, 15, 17, 13, 13, 15, 17]):
    LN.column_dimensions[L(1 + j)].width = w
pmt = f"-PMT({ref['rate']}/12,{ref['term']}-{ref['io']},{ref['loan']})"
for m in range(1, 25):
    rr = 5 + m
    LN[f"A{rr}"] = m
    LN[f"B{rr}"] = f'=IF(A{rr}<={ref["io"]},"Interest only","Repaying")'
    LN[f"D{rr}"] = f"={ref['loan']}" if m == 1 else f"=H{rr - 1}"
    LN[f"E{rr}"] = f"=D{rr}*{ref['rate']}/12"
    LN[f"G{rr}"] = f"=IF(A{rr}<={ref['io']},E{rr},{pmt})"
    LN[f"F{rr}"] = f"=G{rr}-E{rr}"
    LN[f"H{rr}"] = f"=D{rr}-F{rr}"
    for col, src in zip("IJKL", "FEGH"):
        LN[f"{col}{rr}"] = f"={src}{rr}*{ref['fx']}"
        LN[f"{col}{rr}"].number_format = USD
    for col in "DEFGH":
        LN[f"{col}{rr}"].number_format = EUR
    if m == 12:
        style_range(LN, f"A{rr}:L{rr}", border=Border(bottom=Side(style="medium", color=TEAL)))
LN["B30"], LN["B31"] = "Total, 24 months", "Total, Months 1–12"
for col in "EFGIJK":
    LN[f"{col}30"] = f"=SUM({col}6:{col}29)"
    LN[f"{col}31"] = f"=SUM({col}6:{col}17)"
    LN[f"{col}30"].number_format = LN[f"{col}31"].number_format = EUR if col in "EFG" else USD
style_range(LN, "A30:L31", font=F_B, fill=FILL_TOT)
LN.freeze_panes = "A6"
LN.sheet_view.showGridLines = False

# ---------------------------------------------------------------- Cash Flow (indirect)
title(C, "Cash flow statement – indirect method",
      "Starts from net profit and adjusts for non-cash items and working capital. The case brief has none of "
      "these, so those lines are zero and cash flow from operations equals net profit.")
month_header(C, 4)
cf = {}


def cf_block(start, label, np_row, with_loan):
    k = {}
    section(C, start, label)
    k["np"] = line(C, start + 1, "Net profit", lambda c, pc: f"='P&L'!{c}{np_row}")
    k["nc"] = line(C, start + 2, "+ Non-cash items (e.g. depreciation) – none in brief", lambda c, pc: "=0")
    k["wc"] = line(C, start + 3, "± Change in working capital – none in brief", lambda c, pc: "=0")
    k["cfo"] = line(C, start + 4, "Cash flow from operations", lambda c, pc: f"=SUM({c}{start + 1}:{c}{start + 3})",
                    bold=True, border=TOP)
    k["cfi"] = line(C, start + 5, "Cash flow from investing – none in brief", lambda c, pc: "=0")
    if with_loan:
        k["draw"] = line(C, start + 6, "Loan received (€500k × FX)",
                         lambda c, pc: f"=IF({c}$4=1,{ref['loan']}*{ref['fx']},0)")
        k["princ"] = line(C, start + 7, "Loan principal repaid", lambda c, pc: f"=-INDEX(Loan!$I$6:$I$29,{c}$4)")
    else:
        k["draw"] = line(C, start + 6, "Loan received", lambda c, pc: "=0")
        k["princ"] = line(C, start + 7, "Loan principal repaid", lambda c, pc: "=0")
    k["cff"] = line(C, start + 8, "Cash flow from financing", lambda c, pc: f"={c}{start + 6}+{c}{start + 7}",
                    bold=True, border=TOP)
    k["net"] = line(C, start + 9, "Net change in cash",
                    lambda c, pc: f"={c}{start + 4}+{c}{start + 5}+{c}{start + 8}", bold=True)
    k["open"] = line(C, start + 10, "Opening cash",
                     lambda c, pc: f"={ref['cash0']}" if pc is None else f"={pc}{start + 11}",
                     total=f"={FIRST}{start + 10}")
    k["close"] = line(C, start + 11, "Closing cash", lambda c, pc: f"={c}{start + 10}+{c}{start + 9}", bold=True,
                      border=TOPBOT, total="last")
    return k


cf["nl"] = cf_block(6, "A. WITHOUT loan", p["np_nl"], False)
cf["wl"] = cf_block(20, "B. WITH €500k loan", p["np_wl"], True)

# ---------------------------------------------------------------- Runway
title(R, "Cash position, burn & runway",
      "Burn = cash going out minus cash coming in. Runway = months until cash runs out at the current burn.")
month_header(R, 4, opening_label="Start")
rw = {}


def rw_block(start, label, k):
    b = {}
    section(R, start, label)
    b["net"] = line(R, start + 1, "Net cash flow in month", lambda c, pc: f"='Cash Flow'!{c}{cf[k]['net']}")
    b["burn"] = line(R, start + 2, "Monthly burn (positive = losing cash)", lambda c, pc: f"=MAX(0,-{c}{start + 1})")
    b["cash"] = line(R, start + 3, "Cash position (end of month)", lambda c, pc: f"='Cash Flow'!{c}{cf[k]['close']}",
                     bold=True, total="last", opening=f"={ref['cash0']}")
    b["cum"] = line(R, start + 4, "Cumulative cash generated since start",
                    lambda c, pc: f"={c}{start + 3}-{ref['cash0']}", total="last")
    b["runway"] = line(R, start + 5, "Runway (months of cash at this month's burn)",
                       lambda c, pc: f'=IF({c}{start + 2}>0,{c}{start + 3}/{c}{start + 2},"No burn")',
                       '0.0', total=None)
    style_range(R, f"C{start + 5}:{LAST}{start + 5}", align=Alignment(horizontal="right"))
    return b


rw["nl"] = rw_block(6, "A. WITHOUT loan", "nl")
rw["wl"] = rw_block(14, "B. WITH €500k loan", "wl")
for k in ("nl", "wl"):
    R.conditional_formatting.add(f"C{rw[k]['cash']}:{LAST}{rw[k]['cash']}",
                                 CellIsRule(operator="lessThan", formula=["0"], font=Font(color="C00000", bold=True)))

kp = 23
for j, t in enumerate(["Comparison", "", "Without loan", "With loan", "Difference"]):
    R.cell(row=kp, column=1 + j, value=t)
style_range(R, f"A{kp}:E{kp}", font=F_HDR, fill=FILL_HDR)
rngc = lambda k, key: f"${FIRST}${rw[k][key]}:${LAST}${rw[k][key]}"
kpis = [
    ("Cash at end of Month 12", f"={LAST}{rw['nl']['cash']}", f"={LAST}{rw['wl']['cash']}", USD),
    ("Lowest cash position in the 12 months", f"=MIN({rngc('nl', 'cash')})", f"=MIN({rngc('wl', 'cash')})", USD),
    ("Months with cash burn (of 12)", f'=COUNTIF({rngc("nl", "burn")},">0")', f'=COUNTIF({rngc("wl", "burn")},">0")',
     "0"),
    ("Loan still owed at end of Month 12", "=0", "=Loan!L17", USD),
    ("Cash minus loan owed, Month 12", f"=C{kp + 1}-C{kp + 4}", f"=D{kp + 1}-D{kp + 4}", USD),
    ("Interest paid in the 12 months", "=0", "=Loan!J31", USD),
]
for i, (lab, fnl, fwl, fmt) in enumerate(kpis, start=1):
    rr = kp + i
    R[f"A{rr}"], R[f"C{rr}"], R[f"D{rr}"], R[f"E{rr}"] = lab, fnl, fwl, f"=D{rr}-C{rr}"
    for col in "CDE":
        R[f"{col}{rr}"].number_format = fmt
        R[f"{col}{rr}"].font = F_B
rr = kp + len(kpis) + 1
R[f"A{rr}"] = "Runway"
R[f"C{rr}"] = f'=IF(C{kp + 3}=0,"No burn",MIN({rngc("nl", "runway")}))'
R[f"D{rr}"] = f'=IF(D{kp + 3}=0,"No burn",MIN({rngc("wl", "runway")}))'
style_range(R, f"A{rr}:E{rr}", font=F_B, fill=FILL_TOT, align=Alignment(horizontal="right"))
R[f"A{rr}"].alignment = Alignment(horizontal="left")

ch = LineChart()
ch.title = "Cash position: with vs. without loan"
ch.height, ch.width = 7.5, 18
ch.y_axis.numFmt = '#,##0'
for k, name in (("nl", "Without loan"), ("wl", "With €500k loan")):
    ch.add_data(Reference(R, min_col=3, max_col=3 + M - 1, min_row=rw[k]["cash"]), from_rows=True)
    ch.series[-1].tx = SeriesLabel(v=name)
ch.set_categories(Reference(R, min_col=3, max_col=3 + M - 1, min_row=4))
R.add_chart(ch, f"G{kp}")

# ---------------------------------------------------------------- Summary
title(SM, "Luca – 12-month financial model", "All data fictional. Inputs are on the Assumptions and Headcount tabs.")
SM.column_dimensions["A"].width = 3
SM.column_dimensions["B"].width = 42
for col in "CDE":
    SM.column_dimensions[col].width = 16
for j, t in enumerate(["Key figures", "Month 1", "Month 12", "12-month total"]):
    SM.cell(row=4, column=2 + j, value=t)
style_range(SM, "B4:E4", font=F_HDR, fill=FILL_HDR)
rows = [
    ("Revenue", f"='P&L'!{FIRST}{p['rev']}", f"='P&L'!{LAST}{p['rev']}", f"='P&L'!{TOT}{p['rev']}", USD),
    ("Gross margin %", f"='P&L'!{FIRST}{p['gm']}", f"='P&L'!{LAST}{p['gm']}", f"='P&L'!{TOT}{p['gm']}", PCT),
    ("Total operating expenses", f"=-'P&L'!{FIRST}{p['opex']}", f"=-'P&L'!{LAST}{p['opex']}",
     f"=-'P&L'!{TOT}{p['opex']}", USD),
    ("Operating profit", f"='P&L'!{FIRST}{p['op']}", f"='P&L'!{LAST}{p['op']}", f"='P&L'!{TOT}{p['op']}", USD),
    ("Headcount (new hires, end of month)", f"=Headcount!{FIRST}{h['hc']}", f"=Headcount!{LAST}{h['hc']}", "", INT),
    ("Cash position – without loan", f"=Runway!{FIRST}{rw['nl']['cash']}", f"=Runway!{LAST}{rw['nl']['cash']}", "",
     USD),
    ("Cash position – with loan", f"=Runway!{FIRST}{rw['wl']['cash']}", f"=Runway!{LAST}{rw['wl']['cash']}", "", USD),
    ("Runway – without loan", "", "", f"=Runway!C{rr}", USD),
    ("Runway – with loan", "", "", f"=Runway!D{rr}", USD),
]
for i, (lab, f1, f12, ft, fmt) in enumerate(rows):
    r_ = 5 + i
    SM[f"B{r_}"] = lab
    for col, f in zip("CDE", (f1, f12, ft)):
        SM[f"{col}{r_}"] = f or None
        SM[f"{col}{r_}"].number_format = fmt
    if i % 2 == 0:
        style_range(SM, f"B{r_}:E{r_}", fill=FILL_TOT)
SM["B16"] = "Tabs"
SM["B16"].font = F_SEC
for i, t in enumerate([
    "Assumptions – every number from the case brief, plus opening cash (my assumption).",
    "Headcount – new senior / junior hires per month → headcount and people costs.",
    "P&L – revenue, gross margin, costs, operating profit; net profit with and without loan interest.",
    "Cash Flow – indirect method, once without the loan and once with it.",
    "Loan – 24-month repayment schedule in EUR and USD.",
    "Runway – monthly and cumulative cash position, burn and runway, with vs. without the loan.",
]):
    SM[f"B{17 + i}"] = t

for ws in wb.worksheets:
    ws.sheet_properties.tabColor = {"Summary": TEAL, "Assumptions": "F59E0B"}.get(ws.title, NAVY)
    ws.page_setup.orientation = "landscape"
    ws.page_setup.paperSize = ws.PAPERSIZE_A4
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    for row in ws.iter_rows():
        for c in row:
            if c.font is None or c.font.name != "Arial":
                c.font = Font(name="Arial", size=c.font.size if c.font else 10, bold=c.font.bold,
                              italic=c.font.italic, color=c.font.color)

OUT.parent.mkdir(parents=True, exist_ok=True)
wb.save(OUT)
print("saved", OUT)
print({"p": p, "cf_nl_close": cf["nl"]["close"], "cf_wl_close": cf["wl"]["close"],
       "rw": {k: v["cash"] for k, v in rw.items()}, "runway_row": rr, "hc": h})
