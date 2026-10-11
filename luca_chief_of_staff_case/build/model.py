"""Reference implementation of the lean Luca 12-month model.

Uses only the case-brief assumptions plus two required choices: opening cash and the
junior/senior hiring schedule. Mirrors the spreadsheet formulas so the workbook can be
cross-checked after LibreOffice recalculation, and feeds the slide charts.
"""
MONTHS = 12

# New hires starting each month (Months 1-6): (senior, junior)
HIRES = {1: (3, 2), 2: (2, 3), 3: (2, 3), 4: (1, 3), 5: (1, 2), 6: (1, 2)}

START_REVENUE = 1_500_000
GROWTH = 0.06
GM_START, GM_END = 0.65, 0.60
JUNIOR, SENIOR = 3_000, 6_000
HIRING_COST = 6_000
BENEFITS = 0.15
OVERHEAD, OVERHEAD_GROWTH = 40_000, 0.15
GA = 20_000
OPENING_CASH = 3_000_000
LOAN_EUR, LOAN_RATE, LOAN_TERM, LOAN_IO, FX = 500_000, 0.06, 24, 6, 1.10


def loan_schedule():
    """24-month waterfall in EUR: opening, interest, principal, payment, closing."""
    r = LOAN_RATE / 12
    pmt = LOAN_EUR * r / (1 - (1 + r) ** -(LOAN_TERM - LOAN_IO))
    rows, bal = [], LOAN_EUR
    for m in range(1, LOAN_TERM + 1):
        interest = bal * r
        payment = interest if m <= LOAN_IO else pmt
        principal = payment - interest
        rows.append(dict(month=m, opening=bal, interest=interest, principal=principal,
                         payment=payment, closing=bal - principal))
        bal -= principal
    return rows


def run():
    loan = loan_schedule()
    o = {k: [] for k in ["revenue", "gm", "cogs", "gross_profit", "new_sr", "new_jr", "hc_sr", "hc_jr",
                         "hc_total", "salaries", "benefits", "hiring_costs", "overhead", "ga", "opex",
                         "op_profit", "interest", "np_nl", "np_wl", "principal", "drawdown",
                         "net_cf_nl", "net_cf_wl", "cash_nl", "cash_wl", "loan_bal"]}
    cash_nl = cash_wl = OPENING_CASH
    hc_sr = hc_jr = 0
    for m in range(1, MONTHS + 1):
        rev = START_REVENUE * (1 + GROWTH) ** (m - 1)
        gm = GM_START + (GM_END - GM_START) * (m - 1) / (MONTHS - 1)
        cogs = rev * (1 - gm)
        new_sr, new_jr = HIRES.get(m, (0, 0))
        hc_sr += new_sr
        hc_jr += new_jr
        sal = hc_sr * SENIOR + hc_jr * JUNIOR
        ben = sal * BENEFITS
        hire = (new_sr + new_jr) * HIRING_COST
        ovh = OVERHEAD * (1 + OVERHEAD_GROWTH * (m - 1) / (MONTHS - 1))
        opex = sal + ben + hire + ovh + GA
        op = rev - cogs - opex
        interest = loan[m - 1]["interest"] * FX
        principal = loan[m - 1]["principal"] * FX
        draw = LOAN_EUR * FX if m == 1 else 0.0
        net_nl = op
        net_wl = op - interest + draw - principal
        cash_nl += net_nl
        cash_wl += net_wl
        vals = dict(revenue=rev, gm=gm, cogs=cogs, gross_profit=rev - cogs, new_sr=new_sr, new_jr=new_jr,
                    hc_sr=hc_sr, hc_jr=hc_jr, hc_total=hc_sr + hc_jr, salaries=sal, benefits=ben,
                    hiring_costs=hire, overhead=ovh, ga=GA, opex=opex, op_profit=op, interest=interest,
                    np_nl=op, np_wl=op - interest, principal=principal, drawdown=draw,
                    net_cf_nl=net_nl, net_cf_wl=net_wl, cash_nl=cash_nl, cash_wl=cash_wl,
                    loan_bal=loan[m - 1]["closing"] * FX)
        for k, v in vals.items():
            o[k].append(v)
    return o


if __name__ == "__main__":
    o = run()
    print("revenue 12m %.0f  M12 %.0f" % (sum(o["revenue"]), o["revenue"][-1]))
    print("op profit 12m %.0f" % sum(o["op_profit"]))
    print("cash M12 no loan %.0f with loan %.0f; loan bal %.0f" % (o["cash_nl"][-1], o["cash_wl"][-1], o["loan_bal"][-1]))
    print("min cash nl %.0f wl %.0f" % (min(o["cash_nl"]), min(o["cash_wl"])))
    print("net cf nl", [round(x / 1e3) for x in o["net_cf_nl"]])
    print("opex", [round(x / 1e3) for x in o["opex"]])
    print("interest 12m %.0f, 24m %.0f" % (sum(o["interest"]), sum(r["interest"] for r in loan_schedule()) * FX))
