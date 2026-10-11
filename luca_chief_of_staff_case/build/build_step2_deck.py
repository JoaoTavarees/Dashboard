"""Step 2 – Financial & Operations Summary (5 slides, board-ready), built from the lean model."""
from pathlib import Path

from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import model
from deck_helpers import (AMBER, AMBER_LT, CORAL, CORAL_LT, GREY, GREY_LT, INK, NAVY, TEAL, TEAL_LT, WHITE,
                          base_slide, card, new_deck, panel, pill, rect, style_chart, table, text)

OUT = Path(__file__).resolve().parents[1] / "deliverables" / "Luca_Step2_Financial_Operations_Summary.pptx"
SRC = "Source: Luca_Step1_Financial_Model.xlsx (case assumptions; opening cash $3.0M assumed)"

o = model.run()
loan = model.loan_schedule()
months = [f"M{m}" for m in range(1, 13)]


def m(x, d=1):
    return f"${x / 1e6:,.{d}f}M"


def k(x):
    return f"${x / 1e3:,.0f}k"


rev12 = sum(o["revenue"])
people = [a + b + c for a, b, c in zip(o["salaries"], o["benefits"], o["hiring_costs"])]
tot_cost = sum(o["cogs"]) + sum(o["opex"])
gp_lost = sum(r * (model.GM_START - g) for r, g in zip(o["revenue"], o["gm"]))
int24 = sum(r["interest"] for r in loan) * model.FX
int12 = sum(o["interest"])
bal12 = o["loan_bal"][-1]
fx_hit = sum(r["payment"] for r in loan) * 0.10
breakeven = o["opex"][-1] / o["gm"][-1]
run_rate_people = o["salaries"][-1] + o["benefits"][-1]

prs = new_deck()

# ------------------------------------------------------------------------------------------- 1
s = base_slide(prs, "1 · Who we are & executive summary",
               "Luca can fund its next growth leap from operations; the plan's real risk is execution", 1)
rect(s, 0.5, 1.6, 4.1, 5.3, NAVY, shape=1)
text(s, 0.75, 1.75, 3.7, 5.1, [
    dict(t="LUCA AT A GLANCE", size=10, bold=True, color=TEAL, space=8),
    dict(t="EdTech SaaS for K–12 education in Latin America", size=15, bold=True, color=WHITE, space=10),
    dict(t="One of the fastest-growing startups in Mexico, selling its learning platform and video content "
           "to schools.", size=11, color=WHITE, space=12),
    dict(t="STAGE", size=9, bold=True, color=TEAL, space=2),
    dict(t="Post-Series A · $1.5M monthly revenue ($18M a year run-rate) · growing 6% a month",
         size=11, color=WHITE, space=12),
    dict(t="AMBITION", size=9, bold=True, color=TEAL, space=2),
    dict(t="Reach unicorn status within 7 years and expand globally.", size=11, color=WHITE, space=12),
    dict(t="ROLE OF THE CHIEF OF STAFF", size=9, bold=True, color=TEAL, space=2),
    dict(t="Support that growth with financial insight, scalable systems, strong execution and compliant "
           "international operations.", size=11, color=WHITE),
])
text(s, 4.9, 1.55, 8, 0.4, [dict(t="What the board needs to know", size=14, bold=True, color=NAVY)])
items = [
    ("GROWTH", f"{m(o['revenue'][0])} → {m(o['revenue'][-1], 2)} monthly revenue",
     f"{m(rev12)} over 12 months; annual run-rate ≈ {m(o['revenue'][-1] * 12, 0)} by Month 12.", TEAL_LT, TEAL),
    ("PROFITABILITY", f"{sum(o['op_profit']) / rev12:.0%} operating margin; cash {m(model.OPENING_CASH)} → "
                      f"{m(o['cash_nl'][-1])}",
     f"Every month is cash-positive. Gross margin slips 65% → 60% (≈{k(gp_lost)} of gross profit "
     "forgone).", TEAL_LT, TEAL),
    ("HIRING", "25 hires in M1–M6: 10 senior, 15 junior",
     f"Seniors first, then juniors. People cost settles at {k(run_rate_people)}/month from M6.", TEAL_LT, TEAL),
    ("LOAN", "Recommendation: do not take the €500k loan",
     f"Cash never drops below {m(min(o['cash_nl']))} without it. The loan adds interest cost and EUR exposure "
     "for no runway benefit.", AMBER_LT, AMBER),
    ("WATCH", "Gross margin and cash collection",
     f"Each 1 pt of gross margin = {k(o['revenue'][-1] * 0.01)}/month by M12. The model assumes schools pay in "
     "the month invoiced.", CORAL_LT, CORAL),
]
y = 2.0
for lab, head, sub, fill, acc in items:
    rect(s, 4.9, y, 7.95, 0.92, fill, shape=5)
    rect(s, 4.9, y + 0.1, 0.06, 0.72, acc)
    text(s, 5.1, y + 0.06, 1.55, 0.8, [dict(t=lab, size=10, bold=True, color=acc)], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.6, y + 0.04, 6.15, 0.86, [dict(t=head, size=12.5, bold=True, color=NAVY, space=2),
                                        dict(t=sub, size=10, color=INK, space=0)], anchor=MSO_ANCHOR.MIDDLE)
    y += 1.0

# ------------------------------------------------------------------------------------------- 2
s = base_slide(prs, "2 · Key insights from the 12-month model",
               "Revenue nearly doubles and every month generates cash; cost of revenue, not headcount, drives cost",
               2, SRC)
cards = [
    ("Revenue trajectory", f"{m(o['revenue'][0])} → {m(o['revenue'][-1], 2)}",
     f"+6% a month, compounding · {m(rev12)} over 12 months"),
    ("Gross margin trend", "65% → 60%", f"Equal monthly steps · {k(gp_lost)} gross profit lost vs. flat 65%"),
    ("Monthly burn", "None: cash-positive",
     f"Cash generated: +{k(o['net_cf_nl'][0])} (M1) → +{k(o['net_cf_nl'][-1])} (M12)"),
    ("Runway", "Not a constraint",
     f"Cash {m(model.OPENING_CASH)} → {m(o['cash_nl'][-1])} without loan; {m(o['cash_wl'][-1])} with it"),
]
for i, (lab, val, sub) in enumerate(cards):
    card(s, 0.5 + i * 3.12, 1.6, 2.98, 1.25, lab, val, sub, value_size=19)

cd = CategoryChartData()
cd.categories = months
cd.add_series("Revenue", [round(v / 1e3) for v in o["revenue"]])
cd.add_series("Gross profit", [round(v / 1e3) for v in o["gross_profit"]])
cd.add_series("Operating profit", [round(v / 1e3) for v in o["op_profit"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.45), Inches(3.05), Inches(6.4), Inches(3.95), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Monthly revenue, gross profit & operating profit ($k)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col in zip(ch.series, [NAVY, TEAL, AMBER]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
ch.plots[0].gap_width = 60
ch.plots[0].overlap = -10
va = ch.value_axis
va.major_gridlines.format.line.color.rgb = GREY_LT
va.tick_labels.number_format = '#,##0'
va.tick_labels.number_format_is_linked = False
va.format.line.fill.background()

text(s, 7.1, 3.0, 5.75, 0.35, [dict(t="Main cost drivers and how they evolve", size=12.5, bold=True, color=NAVY)])
rows = [["Cost line", "M1", "M12", "12m total", "Share", "M1→M12"]]
for lab, ser in [("Cost of revenue", o["cogs"]), ("People (salary + benefits + hiring)", people),
                 ("Overhead (+15% over 12m)", o["overhead"]), ("G&A / Legal (fixed)", o["ga"])]:
    rows.append([lab, k(ser[0]), k(ser[-1]), m(sum(ser), 2), f"{sum(ser) / tot_cost:.0%}",
                 f"{ser[-1] / ser[0] - 1:+.0%}"])
table(s, 7.1, 3.38, 5.75, 1.75, rows, col_w=[2.35, 0.62, 0.68, 0.8, 0.6, 0.7], size=9.5, align_right_from=1)
text(s, 7.1, 5.25, 5.75, 1.8, [
    dict(t=f"**Cost of revenue is {sum(o['cogs']) / tot_cost:.0%} of all costs** and more than doubles as revenue "
           "grows and margin compresses. Unit cost of content and delivery is the lever that matters most.",
         bullet=True, size=10.5),
    dict(t=f"**Headcount is a small share**: the 25 hires cost {m(sum(people), 2)} in year 1. People cost peaks "
           f"in M6 ({k(people[5])}, the last month with hiring fees), then stays flat.", bullet=True, size=10.5),
    dict(t=f"**Large safety margin**: costs outside cost of revenue are only ≈{k(o['opex'][-1])}/month, so "
           f"revenue would have to fall below ≈{m(breakeven, 2)}/month before Luca burns cash.",
         bullet=True, size=10.5),
], spacing=5)

# ------------------------------------------------------------------------------------------- 3
s = base_slide(prs, "3 · Recommended hiring plan",
               "25 hires front-loaded to M1–M3: seniors first, then juniors where production is constrained", 3,
               "Source: model Headcount tab · salaries $3k junior / $6k senior + 15% benefits; $6k per hire")
cd = CategoryChartData()
cd.categories = [f"M{i + 1} (Σ{o['hc_total'][i]})" for i in range(6)]
cd.add_series("Senior hires", o["new_sr"][:6])
cd.add_series("Junior hires", o["new_jr"][:6])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_STACKED, Inches(0.45), Inches(1.55), Inches(4.6), Inches(3.3), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "New hires per month (Σ = cumulative headcount)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col in zip(ch.series, [NAVY, TEAL]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
    ser.data_labels.show_value = True
    ser.data_labels.font.color.rgb = WHITE
    ser.data_labels.font.size = Pt(9)
    ser.data_labels.font.bold = True
ch.plots[0].gap_width = 45
ch.value_axis.visible = False
ch.value_axis.has_major_gridlines = False
ch.value_axis.maximum_scale = 6.5

panel(s, 0.5, 5.0, 4.5, 1.95, "Why this shape", [
    dict(t="**Seniors early (8 of 10 by M4)** so they can set up processes and onboard the juniors who follow.",
         bullet=True, size=10),
    dict(t="**About 5 hires a month** at most, so each new person can be onboarded properly; it slows to 3 a "
           "month in M5–M6.", bullet=True, size=10),
    dict(t="**60% junior** keeps the average cost at $4.8k per head and builds a bench for promotion.",
         bullet=True, size=10),
], size=10)

rows = [["Suggested allocation", "Hires (S/J)", "Operational capacity it adds"]]
for fname, sr, jr, why in [
    ("Content production", 2, 8, "Script, design, motion, AI-video and sound capacity toward the 30 videos/week "
                                 "target, plus 2 leads to run the pipeline"),
    ("Sales", 2, 2, "New-school pipeline and closing to sustain 6% monthly growth"),
    ("Customer success", 1, 4, "Onboarding and renewals as the customer base roughly doubles"),
    ("Product & engineering", 3, 0, "AI content tooling to cut cost per video and protect gross margin"),
    ("Finance, legal & ops", 2, 1, "Monthly close, cash forecasting, international compliance"),
]:
    rows.append([dict(t=fname, bold=True), f"{sr + jr}  ({sr}/{jr})", why])
rows.append([dict(t="Total", bold=True), dict(t="25  (10/15)", bold=True),
             dict(t="New-hire headcount 5 → 25 by M6", bold=True)])
table(s, 5.3, 1.6, 7.55, 3.9, rows, col_w=[1.9, 1.05, 4.6], size=9.8)
kp = [("People cost M1", k(people[0])), ("Run-rate from M6", k(run_rate_people) + "/mo"),
      ("One-off hiring cost", k(sum(o["hiring_costs"]))),
      ("Run-rate as % of M12 revenue", f"{run_rate_people / o['revenue'][-1]:.1%}")]
for i, (lab, val) in enumerate(kp):
    card(s, 5.3 + i * 1.9, 5.7, 1.8, 1.25, lab, val, value_size=17)

# ------------------------------------------------------------------------------------------- 4
s = base_slide(prs, "4 · Operational implications",
               "To make the plan realistic we need a hiring engine, one source of truth, and spend & cash controls",
               4)
cols = [
    ("PROCESSES", TEAL, TEAL_LT, [
        ("Before M1", "**Hiring engine**: role scorecards, structured interviews and a weekly pipeline review "
                      "to sustain 5 hires a month."),
        ("Before M1", "**30-60-90 onboarding**: every junior gets a senior buddy, with goals set in week 1."),
        ("By M2", "**Monthly close in 10 working days**, plus a rolling 13-week cash forecast and a monthly "
                  "board KPI pack."),
        ("By M2", "**Invoicing & collections cadence**: the plan assumes schools pay in the month invoiced, so "
                  "late payments need to be chased early."),
    ]),
    ("SYSTEMS", NAVY, GREY_LT, [
        ("Before M1", "**HR & payroll system** able to onboard 25 people in 6 months, plus an applicant "
                      "tracking system for the hiring funnel."),
        ("By M2", "**Accounting system** with Mexican e-invoicing and multi-currency, ready for international "
                  "entities."),
        ("By M3", "**CRM as the single source of truth** for pipeline → contracts → billing, feeding a weekly "
                  "dashboard."),
        ("By M2", "**Content production tracker** with stage deadlines and work-in-progress limits (see "
                  "Execution deck)."),
    ]),
    ("CONTROLS", CORAL, CORAL_LT, [
        ("Before M1", "**Headcount gate**: hire only against the approved 25-role plan; any change needs CEO "
                      "sign-off and a cost note."),
        ("Before M1", "**Spend approval limits** by amount, dual sign-off on payments, and a monthly vendor "
                      "review."),
        ("Monthly", "**Gross margin guardrail**: track cost per customer and per video. If margin is below plan "
                    "for 2 months, run a cost review."),
        ("By M6", "**International compliance**: entity and tax set-up and data protection for minors before "
                  "selling outside Mexico."),
    ]),
]
for i, (head, acc, fill, its) in enumerate(cols):
    x = 0.5 + i * 4.15
    rect(s, x, 1.6, 4.0, 0.45, acc, shape=5)
    text(s, x, 1.6, 4.0, 0.45, [dict(t=head, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE)
    rect(s, x, 2.1, 4.0, 4.1, fill, shape=5)
    y = 2.2
    for when, body in its:
        pill(s, x + 0.12, y + 0.03, 0.85, 0.26, when, fill=acc, size=8)
        text(s, x + 1.02, y - 0.02, 2.9, 1.0, [dict(t=body, size=9.8, space=0)])
        y += 0.99
rect(s, 0.5, 6.32, 12.35, 0.65, NAVY, shape=5)
text(s, 0.7, 6.32, 12.0, 0.65, [dict(
    t="**Owner model:** the Chief of Staff runs the operating cadence (weekly leadership KPIs, monthly business "
      "review, quarterly board pack). Each function head owns their hiring, budget and deadlines.",
    size=10.5, color=WHITE)], anchor=MSO_ANCHOR.MIDDLE)

# ------------------------------------------------------------------------------------------- 5
s = base_slide(prs, "5 · Loan decision & key risks",
               "Don't take the €500k loan: Luca doesn't need the cash, and it adds cost and FX risk", 5, SRC)
rect(s, 0.5, 1.6, 6.3, 0.75, AMBER_LT, shape=5)
rect(s, 0.5, 1.7, 0.06, 0.55, AMBER)
text(s, 0.7, 1.6, 6.0, 0.75, [
    dict(t="RECOMMENDATION", size=9, bold=True, color=AMBER, space=1),
    dict(t="Decline the loan. Revisit only if a real cash need appears, such as expansion spending in euros.",
         size=11.5, bold=True, color=NAVY, space=0)], anchor=MSO_ANCHOR.MIDDLE)

cd = CategoryChartData()
cd.categories = ["M0"] + months
cd.add_series("Without loan", [model.OPENING_CASH / 1e6] + [round(v / 1e6, 2) for v in o["cash_nl"]])
cd.add_series("With €500k loan", [model.OPENING_CASH / 1e6] + [round(v / 1e6, 2) for v in o["cash_wl"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.45), Inches(2.45), Inches(3.6), Inches(2.9), cd)
ch = gf.chart
style_chart(ch, size=8)
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.has_title = True
ch.chart_title.text_frame.text = "Cash position ($M)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col in zip(ch.series, [NAVY, TEAL]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
ch.value_axis.major_gridlines.format.line.color.rgb = GREY_LT
ch.value_axis.tick_labels.number_format = '0'
ch.value_axis.tick_labels.number_format_is_linked = False
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW

rows = [["Month 12", "No loan", "With loan"],
        ["Cash", m(o["cash_nl"][-1], 2), m(o["cash_wl"][-1], 2)],
        ["Loan still owed", "–", k(bal12)],
        ["Cash minus loan", m(o["cash_nl"][-1], 2), m(o["cash_wl"][-1] - bal12, 2)],
        ["Lowest cash (12m)", m(min(o["cash_nl"]), 2), m(min(o["cash_wl"]), 2)]]
table(s, 4.15, 2.55, 2.65, 1.75, rows, col_w=[1.05, 0.8, 0.8], size=8.8, align_right_from=1)
text(s, 4.15, 4.4, 2.65, 1.0, [
    dict(t=f"The loan adds {k(model.LOAN_EUR * model.FX)} of cash, about "
           f"{model.LOAN_EUR * model.FX / o['cash_nl'][-1]:.0%} of Month-12 cash, and leaves Luca "
           f"{k(int12)} worse off after interest.", size=9.5)])
text(s, 0.5, 5.45, 6.3, 1.55, [
    dict(t=f"**Runway:** Luca generates cash every month. Runway is not limited with or without the loan.",
         bullet=True, size=9.8),
    dict(t=f"**Cost:** {k(int24)} of interest over 24 months ({k(int12)} in year 1), with {k(bal12)} still owed "
           "at Month 12.", bullet=True, size=9.8),
    dict(t=f"**Risk:** the debt is in euros, but revenue isn't. If the euro rises by $0.10, repayments cost "
           f"≈{k(fx_hit)} more.", bullet=True, size=9.8),
    dict(t="**Flexibility:** not borrowing keeps the balance sheet clean for a future equity round or a "
           "better-matched facility.", bullet=True, size=9.8),
], spacing=3)

text(s, 7.1, 1.55, 5.75, 0.4, [dict(t="Two risks leadership should monitor monthly", size=13, bold=True,
                                    color=NAVY)])
risks = [
    ("1", "Gross margin falling faster than plan", CORAL, CORAL_LT, [
        f"The plan already assumes 65% → 60%. Each extra point lost costs ≈{k(o['revenue'][-1] * 0.01)}/month "
        "by M12.",
        "**Watch:** monthly gross margin %, cost per customer, cost per finished video.",
        "**Act if:** margin is below plan 2 months in a row → cost review (suppliers, AI tooling for content, "
        "pricing).",
    ]),
    ("2", "Growth and cash collection", AMBER, AMBER_LT, [
        "6% monthly growth compounds; small slips add up. The model also assumes schools pay in the month "
        f"invoiced: a 30-day delay would tie up about one month of revenue ({m(o['revenue'][0])}–"
        f"{m(o['revenue'][-1])}).",
        "**Watch:** monthly growth, sales pipeline, customer churn, days to collect payment.",
        "**Act if:** growth is below 4% for 2 months → pause the M5–M6 hires and review the plan.",
    ]),
]
y = 2.0
for num, head, acc, fill, bullets in risks:
    rect(s, 7.1, y, 5.75, 2.3, fill, shape=5)
    pill(s, 7.25, y + 0.15, 0.42, 0.42, num, fill=acc, size=14)
    text(s, 7.8, y + 0.12, 4.95, 0.5, [dict(t=head, size=12, bold=True, color=NAVY)], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.3, y + 0.68, 5.45, 1.6, [dict(t=b, size=10.2, bullet=True) for b in bullets], spacing=5)
    y += 2.45
text(s, 7.1, 6.88, 5.75, 0.25, [dict(
    t="Note: the case cost base covers only the 25 new hires, overhead and G&A.", size=8.5, italic=True,
    color=GREY)])

prs.save(OUT)
print("saved", OUT)
