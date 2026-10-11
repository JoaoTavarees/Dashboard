"""Step 2 – Financial & Operations Summary (5 slides, board-ready)."""
from pathlib import Path

from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import model
from deck_helpers import (AMBER, AMBER_LT, CORAL, CORAL_LT, GREY, GREY_LT, INK, NAVY, TEAL, TEAL_LT, WHITE,
                          base_slide, card, new_deck, panel, pill, rect, style_chart, table, text)

OUT = Path(__file__).resolve().parents[1] / "deliverables" / "Luca_Step2_Financial_Operations_Summary.pptx"
SRC = "Source: Luca_Step1_Financial_Model.xlsx (Base case unless stated)"

base = model.run(model.SCENARIOS["Base"])
stress = model.run(model.SCENARIOS["Stress"])
down = model.run(model.SCENARIOS["Downside"])
loan = model.loan_schedule(model.Inputs())
fx = model.Inputs().fx
months = [f"M{m}" for m in range(1, 13)]


def m(x, d=1):
    return f"${x / 1e6:,.{d}f}M"


def k(x):
    return f"${x / 1e3:,.0f}k"


rev12 = sum(base["revenue"])
people12 = sum(base["salaries"]) + sum(base["benefits"]) + sum(base["hiring_costs"])
cogs12 = sum(base["cogs"])
ovh12, ga12 = sum(base["overhead"]), sum(base["ga"])
tot_cost = cogs12 + people12 + ovh12 + ga12 + sum(base["dep"])
gp_lost = sum(r * (0.65 - g) for r, g in zip(base["revenue"], base["gm"]))
ar_build = base["ar"][-1] - model.Inputs().dso_open / 30 * model.Inputs().start_revenue
rw_s_nl = model.runway(stress["cash_nl"], stress["net_cf_nl"])
rw_s_wl = model.runway(stress["cash_wl"], stress["net_cf_wl"])
head_nl = model.headroom(model.SCENARIOS["Base"])
int24 = sum(r["interest"] for r in loan) * fx
int12 = sum(r["interest"] for r in loan[:12]) * fx
bal12 = loan[11]["closing"] * fx
netcash_diff = (base["cash_wl"][-1] - bal12) - base["cash_nl"][-1]
fx_hit = sum(r["payment"] for r in loan) * 0.10

prs = new_deck()

# ------------------------------------------------------------------------------------------- 1
s = base_slide(prs, "1 · Who we are & executive summary",
               "Luca can fund its next growth leap from operations; the plan's real risk is execution", 1)
rect(s, 0.5, 1.6, 4.1, 5.3, NAVY, shape=1)
text(s, 0.75, 1.75, 3.7, 5.1, [
    dict(t="LUCA AT A GLANCE", size=10, bold=True, color=TEAL, space=8),
    dict(t="AI-enabled K–12 learning platform for Latin America", size=15, bold=True, color=WHITE, space=10),
    dict(t="Curriculum-aligned short video lessons, practice and teacher analytics, sold to schools "
           "as SaaS. One of Mexico's fastest-growing EdTechs.", size=11, color=WHITE, space=12),
    dict(t="STAGE", size=9, bold=True, color=TEAL, space=2),
    dict(t="Post-Series A · $1.5M monthly revenue ($18M ARR run-rate) · growing 6% MoM (≈2× a year)",
         size=11, color=WHITE, space=12),
    dict(t="AMBITION", size=9, bold=True, color=TEAL, space=2),
    dict(t="Unicorn within 7 years; expand beyond Mexico to the rest of LatAm and then globally.",
         size=11, color=WHITE, space=12),
    dict(t="MY ROLE AS CHIEF OF STAFF", size=9, bold=True, color=TEAL, space=2),
    dict(t="Turn the plan into an operating system: financial visibility, scalable processes, "
           "disciplined hiring and compliant international operations.", size=11, color=WHITE),
])
text(s, 4.9, 1.55, 8, 0.4, [dict(t="What the board needs to know", size=14, bold=True, color=NAVY)])
items = [
    ("GROWTH", f"{m(base['revenue'][0])} → {m(base['revenue'][-1], 2)} monthly revenue",
     f"{m(rev12)} revenue over 12 months; exit ARR run-rate ≈ {m(base['revenue'][-1] * 12, 0)}.", TEAL_LT, TEAL),
    ("PROFITABILITY", f"54% EBITDA margin, {m(base['cash_nl'][-1])} cash by M12",
     f"Every month is cash-positive on the case cost base. Gross margin slips 65% → 60% "
     f"(≈{k(gp_lost)} of gross profit forgone).", TEAL_LT, TEAL),
    ("HIRING", "25 hires in M1–M6: 10 senior, 15 junior",
     "Leaders first, then juniors. 10 of the 25 go to content production; people cost runs at "
     f"{k(base['salaries'][-1] + base['benefits'][-1])}/month from M6.", TEAL_LT, TEAL),
    ("LOAN", "Recommendation: do not draw the €500k term loan",
     f"Runway is not the constraint: cash never falls below {m(min(base['cash_nl']))}. Negotiate an undrawn "
     "USD/MXN standby line for collection shocks instead.", AMBER_LT, AMBER),
    ("WATCH", "Gross margin and cash collections",
     f"Each 1 pt of GM = {k(base['revenue'][-1] * 0.01)}/month by M12. Receivables absorb "
     f"{m(ar_build)} of cash as schools pay at 45 days.", CORAL_LT, CORAL),
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
               "Revenue nearly doubles and every month generates cash; COGS, not headcount, drives cost", 2, SRC)
cards = [
    ("Revenue trajectory", f"{m(base['revenue'][0])} → {m(base['revenue'][-1], 2)}",
     f"+6% MoM compound · {m(rev12)} over 12 months"),
    ("Gross margin trend", "65% → 60%", f"Linear compression · {k(gp_lost)} gross profit lost vs. flat 65%"),
    ("Monthly burn", "None: cash-positive",
     f"Free cash flow +{k(base['net_cf_nl'][0])} (M1) → +{k(base['net_cf_nl'][-1])} (M12)"),
    ("Runway", "Self-funding",
     f"Cash {m(model.Inputs().opening_cash)} → {m(base['cash_nl'][-1])} without loan; "
     f"{m(base['cash_wl'][-1])} with it"),
]
for i, (lab, val, sub) in enumerate(cards):
    card(s, 0.5 + i * 3.12, 1.6, 2.98, 1.25, lab, val, sub, value_size=19)

cd = CategoryChartData()
cd.categories = months
cd.add_series("Revenue", [round(v / 1e3) for v in base["revenue"]])
cd.add_series("Gross profit", [round(v / 1e3) for v in base["gross_profit"]])
cd.add_series("EBITDA", [round(v / 1e3) for v in base["ebitda"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.45), Inches(3.05), Inches(6.4), Inches(3.95), cd)
ch = gf.chart
style_chart(ch)
ch.has_title = True
ch.chart_title.text_frame.text = "Monthly revenue, gross profit & EBITDA ($k)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col in zip(ch.series, [NAVY, TEAL, AMBER]):
    ser.format.fill.solid()
    ser.format.fill.fore_color.rgb = col
ch.plots[0].gap_width = 60
ch.plots[0].overlap = -10
va = ch.value_axis
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = GREY_LT
va.tick_labels.number_format = '#,##0'
va.tick_labels.number_format_is_linked = False
va.format.line.fill.background()

text(s, 7.1, 3.0, 5.75, 0.35, [dict(t="Main cost drivers and how they evolve", size=12.5, bold=True, color=NAVY)])
rows = [["Cost line", "M1", "M12", "12m total", "Share", "M1→M12"]]
lines = [
    ("COGS (delivery, hosting, content)", base["cogs"]),
    ("People (salary + 15% benefits + hiring)",
     [a + b + c for a, b, c in zip(base["salaries"], base["benefits"], base["hiring_costs"])]),
    ("Overhead (+15% linear)", base["overhead"]),
    ("G&A / Legal (fixed)", base["ga"]),
]
for lab, ser in lines:
    rows.append([lab, k(ser[0]), k(ser[-1]), m(sum(ser), 2), f"{sum(ser) / tot_cost:.0%}",
                 f"{ser[-1] / ser[0] - 1:+.0%}"])
table(s, 7.1, 3.38, 5.75, 1.75, rows, col_w=[2.35, 0.62, 0.68, 0.8, 0.6, 0.7], size=9.5, align_right_from=1)
text(s, 7.1, 5.25, 5.75, 1.8, [
    dict(t=f"**COGS is 82% of all costs** and more than doubles (+117%) as revenue grows and margin compresses. "
           "Unit cost of content and delivery is the lever that matters most.", bullet=True, size=10.5),
    dict(t=f"**Headcount is a small share of the P&L**: the 25 hires cost {m(people12, 2)} in year 1. They are "
           f"spread over M1–M6, so people cost peaks in M6 ({k(base['salaries'][5] + base['benefits'][5] + base['hiring_costs'][5])}) "
           "and then stays flat.", bullet=True, size=10.5),
    dict(t=f"**Cash conversion ≈ 61% of EBITDA**: {m(sum(base['tax_nl']))} corporate tax (30%) and "
           f"{m(ar_build)} of receivables build-up are the biggest uses of cash.", bullet=True, size=10.5),
], spacing=5)

# ------------------------------------------------------------------------------------------- 3
s = base_slide(prs, "3 · Recommended hiring plan",
               "25 hires front-loaded to M1–M3: leaders first, then juniors where production is constrained", 3,
               "Source: model Headcount tab · salaries $3k junior / $6k senior + 15% benefits; $6k per hire")
cd = CategoryChartData()
cd.categories = [f"M{i + 1} (Σ{base['hc_total'][i]})" for i in range(6)]
cd.add_series("Senior hires", [sum(1 for h in model.HIRING_PLAN if h[4] == mm and h[3] == "Senior") for mm in range(1, 7)])
cd.add_series("Junior hires", [sum(1 for h in model.HIRING_PLAN if h[4] == mm and h[3] == "Junior") for mm in range(1, 7)])
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
    dict(t="**Seniors early (8 of 10 by M4)** to set up processes and onboard the juniors who follow.",
         bullet=True, size=10),
    dict(t="**About 5 hires a month** is the most we can onboard well; it slows to 3 a month once M4 leaders are in.",
         bullet=True, size=10),
    dict(t="**60% junior** keeps the average cost at $4.8k per head and creates a pipeline for promotions.",
         bullet=True, size=10),
], size=10)

rows = [["Function", "Hires (S/J)", "Timing", "Operational capacity it unlocks"]]
fn_rows = [
    ("Content & Curriculum", "M1–M6", "Head of Content Ops + Curriculum Lead + 8 creators (script, motion, "
     "design, Synthesia, sound, QA) sized to the 30 videos/week target"),
    ("Sales & Partnerships", "M1–M4", "2 AEs (Mexico + first expansion market) + 2 SDRs to keep up 6% MoM "
     "new-logo growth"),
    ("Customer Success", "M2–M6", "Head of CS + 4 associates / implementation: onboarding and renewals "
     "keep pace as the school base roughly doubles"),
    ("Product & Engineering", "M1–M5", "AI/backend + full-stack engineers + PM: AI content tooling to cut cost per "
     "video and protect gross margin"),
    ("Finance, Legal & Ops", "M2–M6", "FP&A, BI analyst, international legal & compliance: close, cash forecasting, "
     "entity set-up"),
]
for fname, timing, why in fn_rows:
    hs = [h for h in model.HIRING_PLAN if h[2] == fname]
    sr = sum(1 for h in hs if h[3] == "Senior")
    rows.append([dict(t=fname, bold=True), f"{len(hs)}  ({sr}/{len(hs) - sr})", timing, why])
rows.append([dict(t="Total", bold=True), dict(t="25  (10/15)", bold=True), dict(t="M1–M6", bold=True),
             dict(t="New-plan headcount 5 → 25 by M6", bold=True)])
table(s, 5.3, 1.6, 7.55, 3.9, rows, col_w=[1.75, 0.95, 0.75, 4.1], size=9.5)
kp = [("People cost M1", k(base["salaries"][0] + base["benefits"][0] + base["hiring_costs"][0])),
      ("Run-rate from M6", k(base["salaries"][-1] + base["benefits"][-1]) + "/mo"),
      ("One-off hiring cost", k(sum(base["hiring_costs"]))),
      ("Run-rate as % of M12 revenue", f"{(base['salaries'][-1] + base['benefits'][-1]) / base['revenue'][-1]:.1%}")]
for i, (lab, val) in enumerate(kp):
    card(s, 5.3 + i * 1.9, 5.7, 1.8, 1.25, lab, val, value_size=17)

# ------------------------------------------------------------------------------------------- 4
s = base_slide(prs, "4 · Operational implications",
               "To make the plan realistic we need a hiring engine, one source of truth, and spend & cash controls",
               4)
cols = [
    ("PROCESSES", TEAL, TEAL_LT, [
        ("Before M1", "**Hiring engine**: role scorecards, structured interviews, weekly pipeline review. "
                      "Target ≤45 days to hire; 5 roles open at any time."),
        ("Before M1", "**30-60-90 onboarding**: every junior gets a senior buddy, and goals are set in week 1."),
        ("By M2", "**Monthly close in 10 working days**, plus a rolling 13-week cash forecast and a "
                  "monthly board KPI pack."),
        ("By M2", "**Collections cadence**: invoice on day 1, reminders at 30/45/60 days, escalation at 75 days. "
                  "Receivables are the biggest use of cash."),
    ]),
    ("SYSTEMS", NAVY, GREY_LT, [
        ("Before M1", "**HRIS + payroll** for Mexico; an Employer of Record (EOR) for first hires abroad; "
                      "ATS to track hiring funnels."),
        ("By M2", "**Finance stack**: accounting/ERP with Mexican e-invoicing (CFDI/SAT) and multi-currency, "
                  "ready for new entities."),
        ("By M3", "**CRM as the revenue source of truth** (pipeline → bookings → billing) feeding a weekly "
                  "BI dashboard."),
        ("By M2", "**Content production tracker** with stage SLAs and WIP limits, plus AI tooling (see "
                  "Execution deck)."),
    ]),
    ("CONTROLS", CORAL, CORAL_LT, [
        ("Before M1", "**Headcount gate**: hire only against the approved 25-role plan. Any change needs "
                      "CEO + CoS sign-off and a P&L impact note."),
        ("Before M1", "**Delegation-of-authority matrix**: spend approvals by threshold; dual sign-off on "
                      "payments; monthly vendor review."),
        ("Monthly", "**Gross margin guardrail**: track COGS per student and cost per video. If GM is below plan "
                    "for 2 months, run a cost review."),
        ("By M6", "**International compliance**: entity and tax set-up, transfer pricing, data protection "
                  "for minors (Mexican LFPDPPP, Brazilian LGPD, GDPR)."),
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
    t="**Owner model:** CoS runs the operating cadence (weekly leadership KPIs, monthly business review, "
      "quarterly board pack). Each function head owns their hiring, budget and SLAs. FP&A (hire #6, M2) "
      "owns the model and the 13-week cash view.", size=10.5, color=WHITE)], anchor=MSO_ANCHOR.MIDDLE)

# ------------------------------------------------------------------------------------------- 5
s = base_slide(prs, "5 · Loan decision & key risks",
               "Don't draw the €500k loan: it barely moves runway and adds FX risk. Watch margin and collections",
               5, SRC + "; Stress = flat revenue, GM→55%, DSO→90 days + $1.0M/mo existing cost base (illustrative)")
rect(s, 0.5, 1.6, 6.3, 0.75, AMBER_LT, shape=5)
rect(s, 0.5, 1.7, 0.06, 0.55, AMBER)
text(s, 0.7, 1.6, 6.0, 0.75, [
    dict(t="RECOMMENDATION", size=9, bold=True, color=AMBER, space=1),
    dict(t="Decline the €500k term loan as structured; negotiate an undrawn USD/MXN standby facility instead",
         size=11.5, bold=True, color=NAVY, space=0)], anchor=MSO_ANCHOR.MIDDLE)

cd = CategoryChartData()
cd.categories = ["M0"] + months
cd.add_series("Without loan", [3.0] + [round(v / 1e6, 2) for v in base["cash_nl"]])
cd.add_series("With €500k loan", [3.0] + [round(v / 1e6, 2) for v in base["cash_wl"]])
cd.add_series("Stress – without loan", [3.0] + [round(v / 1e6, 2) for v in stress["cash_nl"]])
cd.add_series("Stress – with loan", [3.0] + [round(v / 1e6, 2) for v in stress["cash_wl"]])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(0.45), Inches(2.45), Inches(3.6), Inches(2.9), cd)
ch = gf.chart
style_chart(ch, size=8)
ch.has_title = True
ch.chart_title.text_frame.text = "Month-end cash ($M)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col, dash in zip(ch.series, [NAVY, TEAL, CORAL, AMBER], [False, False, True, True]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
    if dash:
        ser.format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
ch.value_axis.major_gridlines.format.line.color.rgb = GREY_LT
ch.value_axis.tick_labels.number_format = '0'
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.value_axis.tick_labels.number_format_is_linked = False

rows = [["Runway", "No loan", "With loan"],
        ["Base", "Self-funding", "Self-funding"],
        ["Downside", "Self-funding", "Self-funding"],
        ["Stress", f"{rw_s_nl:.1f} mo", f"{rw_s_wl:.1f} mo"]]
table(s, 4.1, 2.55, 2.7, 1.45, rows, col_w=[0.8, 0.95, 0.95], size=9, align_right_from=1)
text(s, 4.1, 4.05, 2.7, 1.3, [
    dict(t=f"Even under Stress, the loan buys only **+{rw_s_wl - rw_s_nl:.1f} months**. It does not fix "
           "a structural burn.", size=9.5, space=4),
    dict(t=f"Base plan can absorb **{m(head_nl, 2)}/month** of extra fixed cost before cash runs out.",
         size=9.5)])
text(s, 0.5, 5.35, 6.3, 1.65, [
    dict(t=f"**Runway:** cash never drops below {m(min(base['cash_nl']))} without the loan; free cash flow is "
           "positive in all 12 months.", bullet=True, size=9.8),
    dict(t=f"**Cost:** ${int24 / 1e3:,.0f}k interest over 24 months. Net cash at M12 is {k(-netcash_diff)} lower "
           f"with the loan, and {k(bal12)} of debt is still owed at M12.", bullet=True, size=9.8),
    dict(t=f"**Risk:** a EUR liability against MXN/USD revenue. A +0.10 move in EUR/USD adds ≈{k(fx_hit)} to "
           "repayments. Lenders may also impose covenants and reporting.", bullet=True, size=9.8),
    dict(t="**Flexibility:** a committed but undrawn line covers the real risk (collection delays) without "
           "paying interest. Revisit if we open EUR-cost markets, where EUR debt would be a natural hedge.",
         bullet=True, size=9.8),
], spacing=3)

text(s, 7.1, 1.55, 5.75, 0.4, [dict(t="Two risks leadership should monitor monthly", size=13, bold=True,
                                    color=NAVY)])
risks = [
    ("1", "Gross margin compression outpacing plan", CORAL, CORAL_LT, [
        f"Plan already assumes 65% → 60%. Each extra 1 pt lost costs ≈{k(base['revenue'][-1] * 0.01)}/month by M12.",
        "**KPIs:** monthly GM %, COGS per active student, cost per finished video, hosting per user.",
        "**Trigger:** GM below plan for 2 months running → cost review: vendor renegotiation, AI tooling "
        "for content, pricing.",
    ]),
    ("2", "Collections and quality of growth", AMBER, AMBER_LT, [
        f"Receivables grow by {m(ar_build)} even at 45 days. If schools slip to 90 days, "
        f"receivables double: ≈{m(base['revenue'][-1] * 45 / 30)} more cash tied up at M12 revenue.",
        "**KPIs:** DSO, % of AR over 60 days, MoM growth, pipeline coverage (3× next quarter), "
        "logo churn.",
        "**Trigger:** DSO above 60 days, or growth below 4% for 2 months → pause M5–M6 hires and activate "
        "the standby line.",
    ]),
]
y = 2.0
for num, head, acc, fill, bullets in risks:
    rect(s, 7.1, y, 5.75, 2.3, fill, shape=5)
    pill(s, 7.25, y + 0.15, 0.42, 0.42, num, fill=acc, size=14)
    text(s, 7.8, y + 0.12, 4.95, 0.5, [dict(t=head, size=12, bold=True, color=NAVY)], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 7.3, y + 0.68, 5.45, 1.6, [dict(t=b, size=10.5, bullet=True) for b in bullets], spacing=5)
    y += 2.45
text(s, 7.1, 6.88, 5.75, 0.25, [dict(
    t="Also validate before the board: the case cost base excludes Luca's existing team (see Stress).",
    size=8.5, italic=True, color=GREY)])

prs.save(OUT)
print("saved", OUT)
