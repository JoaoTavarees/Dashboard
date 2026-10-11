"""Reference implementation of the Luca 12-month model.

Mirrors the formulas written into the spreadsheet so the workbook can be
cross-checked after LibreOffice recalculation, and feeds the slide charts.
"""
from dataclasses import dataclass, field

MONTHS = 12

# (hire_id, role, function, level, start_month)
HIRING_PLAN = [
    (1, "Head of Content Operations", "Content & Curriculum", "Senior", 1),
    (2, "Senior Account Executive (Mexico)", "Sales & Partnerships", "Senior", 1),
    (3, "Senior AI / Backend Engineer", "Product & Engineering", "Senior", 1),
    (4, "Motion Designer", "Content & Curriculum", "Junior", 1),
    (5, "Scriptwriter", "Content & Curriculum", "Junior", 1),
    (6, "FP&A / Finance Manager", "Finance, Legal & Ops", "Senior", 2),
    (7, "Senior Product Manager (AI Content Tools)", "Product & Engineering", "Senior", 2),
    (8, "Motion Designer", "Content & Curriculum", "Junior", 2),
    (9, "Customer Success Associate", "Customer Success", "Junior", 2),
    (10, "SDR / Business Development Rep", "Sales & Partnerships", "Junior", 2),
    (11, "Senior Account Executive (Expansion market)", "Sales & Partnerships", "Senior", 3),
    (12, "Senior Curriculum Lead (Secondary)", "Content & Curriculum", "Senior", 3),
    (13, "Scriptwriter", "Content & Curriculum", "Junior", 3),
    (14, "Synthesia / AI Video Specialist", "Content & Curriculum", "Junior", 3),
    (15, "Customer Success Associate", "Customer Success", "Junior", 3),
    (16, "Head of Customer Success", "Customer Success", "Senior", 4),
    (17, "Graphic Designer", "Content & Curriculum", "Junior", 4),
    (18, "SDR / Business Development Rep", "Sales & Partnerships", "Junior", 4),
    (19, "Data Analyst (RevOps / BI)", "Finance, Legal & Ops", "Junior", 4),
    (20, "Senior Full-stack Engineer", "Product & Engineering", "Senior", 5),
    (21, "Implementation Specialist", "Customer Success", "Junior", 5),
    (22, "Sound Editor", "Content & Curriculum", "Junior", 5),
    (23, "Legal & Compliance Manager (International)", "Finance, Legal & Ops", "Senior", 6),
    (24, "Customer Success Associate", "Customer Success", "Junior", 6),
    (25, "QA / Content Reviewer", "Content & Curriculum", "Junior", 6),
]


@dataclass
class Scenario:
    name: str
    mom_growth: float
    gm_start: float
    gm_end: float
    dso_end: float
    extra_opex: float = 0.0


SCENARIOS = {
    "Base": Scenario("Base", 0.06, 0.65, 0.60, 45),
    "Downside": Scenario("Downside", 0.00, 0.65, 0.55, 90),
    "Stress": Scenario("Stress", 0.00, 0.65, 0.55, 90, 1_000_000),
}


@dataclass
class Inputs:
    start_revenue: float = 1_500_000
    junior_salary: float = 3_000
    senior_salary: float = 6_000
    hiring_cost: float = 6_000
    benefits_pct: float = 0.15
    overhead_start: float = 40_000
    overhead_growth_12m: float = 0.15
    ga_legal: float = 20_000
    opening_cash: float = 3_000_000
    dso_open: float = 45
    ap_days: float = 30
    tax_rate: float = 0.30
    capex_per_hire: float = 1_500
    dep_months: int = 36
    loan_eur: float = 500_000
    loan_rate: float = 0.06
    loan_term: int = 24
    loan_io_months: int = 6
    fx: float = 1.10
    hiring_plan: list = field(default_factory=lambda: HIRING_PLAN)


def loan_schedule(inp: Inputs):
    """24-month waterfall in EUR: opening, interest, principal, payment, closing."""
    r = inp.loan_rate / 12
    n_amort = inp.loan_term - inp.loan_io_months
    pmt = inp.loan_eur * r / (1 - (1 + r) ** -n_amort)
    rows, bal = [], inp.loan_eur
    for m in range(1, inp.loan_term + 1):
        interest = bal * r
        payment = interest if m <= inp.loan_io_months else pmt
        principal = payment - interest
        rows.append(dict(month=m, opening=bal, interest=interest, principal=principal,
                         payment=payment, closing=bal - principal))
        bal -= principal
    return rows


def run(sc: Scenario, inp: Inputs = Inputs()):
    out = {k: [] for k in [
        "revenue", "gm", "cogs", "gross_profit", "hc_junior", "hc_senior", "hc_total", "new_hires",
        "salaries", "benefits", "hiring_costs", "overhead", "ga", "opex", "ebitda", "dep", "ebit",
        "interest", "tax_nl", "tax_wl", "ni_nl", "ni_wl", "ar", "ap", "capex",
        "cfo_nl", "cfo_wl", "net_cf_nl", "net_cf_wl", "cash_nl", "cash_wl", "loan_bal_usd",
        "principal_usd", "drawdown_usd", "dso", "extra",
    ]}
    loan = loan_schedule(inp)
    ar_prev = inp.dso_open / 30 * inp.start_revenue
    ap_prev = None
    tax_nl_prev = tax_wl_prev = 0.0
    cash_nl = cash_wl = inp.opening_cash
    capex_hist = []
    for m in range(1, MONTHS + 1):
        rev = inp.start_revenue * (1 + sc.mom_growth) ** (m - 1)
        gm = sc.gm_start + (sc.gm_end - sc.gm_start) * (m - 1) / (MONTHS - 1)
        cogs = rev * (1 - gm)
        hj = sum(1 for h in inp.hiring_plan if h[3] == "Junior" and h[4] <= m)
        hs = sum(1 for h in inp.hiring_plan if h[3] == "Senior" and h[4] <= m)
        new = sum(1 for h in inp.hiring_plan if h[4] == m)
        sal = hj * inp.junior_salary + hs * inp.senior_salary
        ben = sal * inp.benefits_pct
        hire_cost = new * inp.hiring_cost
        ovh = inp.overhead_start * (1 + inp.overhead_growth_12m * (m - 1) / (MONTHS - 1))
        opex = sal + ben + hire_cost + ovh + inp.ga_legal + sc.extra_opex
        ebitda = rev - cogs - opex
        capex = new * inp.capex_per_hire
        capex_hist.append(capex)
        dep = sum(capex_hist) / inp.dep_months
        ebit = ebitda - dep
        interest = loan[m - 1]["interest"] * inp.fx
        tax_nl = max(0.0, ebit) * inp.tax_rate
        tax_wl = max(0.0, ebit - interest) * inp.tax_rate
        ni_nl = ebit - tax_nl
        ni_wl = ebit - interest - tax_wl
        dso = inp.dso_open + (sc.dso_end - inp.dso_open) * (m - 1) / (MONTHS - 1)
        ar = rev * dso / 30
        ap = (cogs + ovh) * inp.ap_days / 30
        if ap_prev is None:
            # opening AP and taxes payable are set at the Month-1 run-rate
            ap_prev = ap
            tax_nl_prev, tax_wl_prev = tax_nl, tax_wl
        d_ar, d_ap = ar - ar_prev, ap - ap_prev
        cfo_nl = ni_nl + dep - d_ar + d_ap + (tax_nl - tax_nl_prev)
        cfo_wl = ni_wl + dep - d_ar + d_ap + (tax_wl - tax_wl_prev)
        draw = inp.loan_eur * inp.fx if m == 1 else 0.0
        princ = loan[m - 1]["principal"] * inp.fx
        net_nl = cfo_nl - capex
        net_wl = cfo_wl - capex + draw - princ
        cash_nl += net_nl
        cash_wl += net_wl
        for k, v in dict(revenue=rev, gm=gm, cogs=cogs, gross_profit=rev - cogs, hc_junior=hj,
                         hc_senior=hs, hc_total=hj + hs, new_hires=new, salaries=sal, benefits=ben,
                         hiring_costs=hire_cost, overhead=ovh, ga=inp.ga_legal, opex=opex,
                         ebitda=ebitda, dep=dep, ebit=ebit, interest=interest, tax_nl=tax_nl,
                         tax_wl=tax_wl, ni_nl=ni_nl, ni_wl=ni_wl, ar=ar, ap=ap, capex=capex,
                         cfo_nl=cfo_nl, cfo_wl=cfo_wl, net_cf_nl=net_nl, net_cf_wl=net_wl,
                         cash_nl=cash_nl, cash_wl=cash_wl,
                         loan_bal_usd=loan[m - 1]["closing"] * inp.fx, principal_usd=princ,
                         drawdown_usd=draw, dso=dso, extra=sc.extra_opex).items():
            out[k].append(v)
        ar_prev, ap_prev = ar, ap
        tax_nl_prev, tax_wl_prev = tax_nl, tax_wl
    return out


def runway(cash, net_cf):
    """Months from today until cash < 0; beyond M12 extrapolates on the avg burn of M10-M12."""
    for i, c in enumerate(cash):
        if c < 0:
            prev = cash[i - 1] if i else None
            return i + (prev / (prev - c) if prev is not None else 0)
    burn = -sum(net_cf[-3:]) / 3
    return None if burn <= 0 else MONTHS + cash[-1] / burn


def headroom(sc: Scenario, key="cash_nl", inp: Inputs = Inputs()):
    """Largest extra monthly fixed cost that keeps cash >= 0 for all 12 months."""
    lo, hi = 0.0, 5_000_000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        o = run(Scenario(sc.name, sc.mom_growth, sc.gm_start, sc.gm_end, sc.dso_end,
                         sc.extra_opex + mid), inp)
        lo, hi = (mid, hi) if min(o[key]) >= 0 else (lo, mid)
    return lo


if __name__ == "__main__":
    for name, sc in SCENARIOS.items():
        o = run(sc)
        print(f"== {name}")
        print("  revenue 12m total %.0f  M12 %.0f" % (sum(o["revenue"]), o["revenue"][-1]))
        print("  EBITDA 12m %.0f  NI(no loan) %.0f  NI(loan) %.0f" % (
            sum(o["ebitda"]), sum(o["ni_nl"]), sum(o["ni_wl"])))
        print("  cash M12 no loan %.0f with loan %.0f" % (o["cash_nl"][-1], o["cash_wl"][-1]))
        print("  min cash no loan %.0f with loan %.0f" % (min(o["cash_nl"]), min(o["cash_wl"])))
        print("  net cf nl", [round(x / 1e3) for x in o["net_cf_nl"]])
        print("  cash nl", [round(x / 1e3) for x in o["cash_nl"]])
        print("  runway nl", runway(o["cash_nl"], o["net_cf_nl"]), "wl", runway(o["cash_wl"], o["net_cf_wl"]))
        print("  headroom nl %.0f wl %.0f" % (headroom(sc), headroom(sc, "cash_wl")))
    ls = loan_schedule(Inputs())
    print("loan pmt", ls[6]["payment"], "total interest", sum(r["interest"] for r in ls))
