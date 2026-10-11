"""Step 3 – Execution & Content Operations (4 slides)."""
from pathlib import Path

from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from deck_helpers import (AMBER, AMBER_LT, CORAL, CORAL_LT, GREY, GREY_LT, INK, LINE, NAVY, TEAL, TEAL_LT, WHITE,
                          base_slide, card, new_deck, panel, pill, rect, style_chart, table, text)

OUT = Path(__file__).resolve().parents[1] / "deliverables" / "Luca_Step3_Execution_Content_Operations.pptx"
SRC = "Source: outgoing PMO Week 1 tracking"

STAGES = [  # name, owner, started, finished, h/unit, blockers, blocker text
    ("1. Brief & content definition", "Content Authors", 35, 35, 1.0, 0, "—"),
    ("2. Script creation & review", "Scriptwriters", 35, 20, 4.0, 15, "Misalignment with authors; heavy rework"),
    ("3. Visual asset design", "Graphic Designers", 20, 15, 3.0, 5, "Waiting for final script"),
    ("4. Motion graphics & animation", "Motion Designers", 15, 8, 5.0, 7, "Frequent changes from earlier stages"),
    ("5. Final video (Synthesia + sound)", "Synthesia / Sound", 8, 5, 6.0, 3, "A/V sync; last-minute visual tweaks"),
]
BOTTLENECK = {1: "#1", 3: "#2"}
TARGET, WEEKS, TOTAL = 30, 10, 300
done_w1 = STAGES[-1][3]
need = (TOTAL - done_w1) / (WEEKS - 1)
wip = sum(s[2] - s[3] for s in STAGES)
hours_per_video = sum(s[4] for s in STAGES)

prs = new_deck()

# ------------------------------------------------------------------------------------------- 1
s = base_slide(prs, "1 · Quick diagnosis of Week 1",
               "We finished 5 of 30 videos: rework in scripting and change churn in motion are choking the line",
               1, SRC)
text(s, 0.5, 1.58, 7.3, 0.3, [dict(t="Units started vs. finished by stage (bar = 35 units)", size=11, bold=True,
                                   color=NAVY)])
y0, bar_x, bar_w = 1.98, 3.15, 2.45
for i, (name, owner, st, fin, h, blk, why) in enumerate(STAGES):
    y = y0 + i * 0.78
    hot = i in BOTTLENECK
    if hot:
        rect(s, 0.45, y - 0.06, 7.45, 0.74, CORAL_LT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 0.55, y, 2.55, 0.62, [dict(t=name, size=10.5, bold=True, color=NAVY, space=1),
                                  dict(t=owner, size=9, color=GREY, space=0)], anchor=MSO_ANCHOR.MIDDLE)
    rect(s, bar_x, y + 0.12, bar_w * st / 35, 0.38, LINE)
    rect(s, bar_x, y + 0.12, bar_w * fin / 35, 0.38, CORAL if hot else TEAL)
    text(s, bar_x + 0.05, y + 0.12, 1.4, 0.38, [dict(t=f"{fin}/{st}", size=10, bold=True, color=WHITE)],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 5.65, y, 0.75, 0.62, [dict(t=f"{fin / st:.0%}", size=13, bold=True, color=CORAL if hot else NAVY,
                                       align=PP_ALIGN.CENTER, space=0),
                                  dict(t="finished", size=8, color=GREY, align=PP_ALIGN.CENTER)],
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, 6.4, y, 1.45, 0.62, [dict(t=f"{h:.0f} h/unit · {blk} blocked", size=9.5, bold=True, space=1),
                                 dict(t=why, size=8.5, color=GREY, space=0)], anchor=MSO_ANCHOR.MIDDLE)
    if hot:
        pill(s, 0.12, y + 0.17, 0.42, 0.3, BOTTLENECK[i], fill=CORAL, size=9)
for i, (lab, val, sub) in enumerate([
    ("Week 1 output", f"{done_w1} / {TARGET}", "17% of weekly target"),
    ("Needed W2–W10", f"{need:.1f} / week", f"to still deliver {TOTAL} by Week {WEEKS}"),
    ("Stuck in progress", f"{wip} units", f"{hours_per_video:.0f} touch-hours per finished video"),
]):
    card(s, 0.5 + i * 2.5, 5.95, 2.38, 1.05, lab, val, sub, value_size=18, fill=GREY_LT, accent=NAVY)

bn = [
    ("#1  Script creation & review (Stage 2)", [
        "**Evidence:** 15 of 35 units (43%) stuck, the largest loss in the line. It also starves Stage 3, "
        "where 5 units are waiting for a final script.",
        "**Root cause:** no shared 'definition of ready' between Content Authors and Scriptwriters. "
        "Reviews happen one after another, so each misunderstanding means a full rewrite (4 h/unit).",
        "**So what:** at most 20 scripts a week can enter production, so 30 a week is impossible upstream.",
    ]),
    ("#2  Motion graphics & animation (Stage 4)", [
        "**Evidence:** lowest completion rate (53%, 8 of 15) and the most effort of any creative stage "
        "(5 h/unit). 7 units are blocked by changes from earlier stages.",
        "**Root cause:** no freeze point. Script and visual edits keep arriving after animation has started, "
        "and every video is animated from scratch.",
        "**So what:** only 8 units reach final assembly. At target pace motion needs ~150 h/week; it "
        "lost a big part of Week 1 to redoing work.",
    ]),
]
y = 1.58
for head, bullets in bn:
    rect(s, 8.1, y, 4.75, 2.4, CORAL_LT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, 8.25, y + 0.08, 4.5, 0.35, [dict(t=head, size=11.5, bold=True, color=CORAL)])
    text(s, 8.25, y + 0.45, 4.5, 2.1, [dict(t=b, size=9.6, bullet=True) for b in bullets], spacing=3)
    y += 2.5
text(s, 8.1, 6.55, 4.75, 0.45, [dict(t="Common thread: changes flow downstream unchecked, and Stage 1 keeps "
                                       "starting work the line can't absorb.", size=8.5, italic=True, color=GREY)])

# ------------------------------------------------------------------------------------------- 2
s = base_slide(prs, "1 · 2-week action plan",
               "Five rules for the next 2 weeks: align upstream, freeze changes, limit WIP, add motion capacity",
               2, "Targets are commitments to review in the weekly pipeline meeting")
rows = [["#", "Action (decision / rule we put in place)", "Owner", "When", "Success metric (end of Week 3)"]]
acts = [
    ("**One brief, co-signed.** The Content Author and Scriptwriter fill in a 1-page 'script-ready brief' in a "
     "20-min pairing before any script starts: learning objective, key concepts, scene outline, 600–850 words. "
     "Author reviews within 24 h, max 2 rounds; a 3rd round goes to the Curriculum Lead to decide.",
     "Head of Content Ops", "Day 1–2", "≥80% scripts approved first pass; ≤1.2 rework loops / unit"),
    ("**Lock points + change control.** The script locks when it leaves Stage 2 and visuals lock when they leave "
     "Stage 4. After a lock, only factual or pedagogical errors may change (Content Lead approves within 24 h). "
     "Everything else goes to a v2 backlog.", "Curriculum Lead", "Day 1",
     "0 change-driven blockers in Stages 4–5"),
    ("**Stop starting, start finishing.** Cap the queue in front of each stage at 10 units. Stage 1 pauses new "
     "briefs while the script queue is over 10. Authors spend 50% of their time in pair-fix sessions to clear "
     "the 15 blocked scripts.", "PMO / Chief of Staff", "Day 1–5",
     "Script backlog 15 → <5; WIP ≤ 40 units"),
    ("**Add capacity where it's scarce.** 3 complexity tiers (A bespoke 20% · B template 50% · C avatar + "
     "static 30%). A library of 15–20 reusable motion templates. 2–3 freelance motion designers for W2–W6. "
     "Designers start from the approved scene outline, in parallel with the script.",
     "Motion Lead + Head of Content Ops", "Week 1–2", "Motion 5.0 → ≤3.0 h/unit; ≥36 units/week out of Stage 4"),
    ("**Right-first-time final assembly.** One technical spec (audio −16 LUFS, 48 kHz, timings from the "
     "locked script), a Synthesia template per subject, sound + Synthesia working as one cell on the same "
     "batch, and a 10-point QC checklist before export.", "Synthesia Lead", "Week 2",
     "0 A/V sync rejects; Stage 5 6 → 4 h/unit"),
]
for i, (a, o, w, kpi) in enumerate(acts, start=1):
    rows.append([dict(t=str(i), bold=True, color=TEAL), a, o, w, kpi])
table(s, 0.5, 1.6, 8.6, 5.35, rows, col_w=[0.35, 4.85, 1.2, 0.75, 1.45], size=9.2)

W = list(range(1, 11))
plan = [TARGET * w for w in W]
rec_weekly = [5, 15, 25] + [36, 36, 37, 36, 37, 36, 37]
rec = [sum(rec_weekly[:w]) for w in W]
cd = CategoryChartData()
cd.categories = [f"W{w}" for w in W]
cd.add_series("Original plan (30/wk)", plan)
cd.add_series("Recovery plan", rec)
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(9.25), Inches(1.55), Inches(3.65), Inches(3.0), cd)
ch = gf.chart
style_chart(ch, size=8)
ch.legend.position = XL_LEGEND_POSITION.BOTTOM
ch.has_title = True
ch.chart_title.text_frame.text = "Cumulative finished videos"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(10)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
for ser, col in zip(ch.series, [GREY, TEAL]):
    ser.format.line.color.rgb = col
    ser.format.line.width = Pt(2.25)
    ser.smooth = False
    ser.marker.format.fill.solid()
    ser.marker.format.fill.fore_color.rgb = col
    ser.marker.format.line.color.rgb = col
ch.series[0].format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
ch.value_axis.major_gridlines.format.line.color.rgb = GREY_LT
ch.value_axis.maximum_scale = 300
panel(s, 9.25, 4.65, 3.6, 2.3, "Weekly finish targets", [
    dict(t="**W2: 15 · W3: 25 · W4–W10: ≥36/week** → 300 by Week 10.", size=9.8),
    dict(t="W2 mostly clears the 30 units already in progress; from W4 the tiers and templates carry "
           "the higher pace.", size=9.5),
    dict(t="**Escalation rule:** if W3 is under 20, the steering group moves more videos to Tier C and orders "
           "subjects by the school-term calendar.", size=9.5),
], fill=TEAL_LT, accent=TEAL, size=9.5, head_size=11)

# ------------------------------------------------------------------------------------------- 3
s = base_slide(prs, "1 · AI video production tool & architecture",
               "One AI-orchestrated pipeline from curriculum to finished video, with humans approving at key gates",
               3, "Hour savings are planning estimates to validate in the MVP")
rect(s, 0.5, 1.58, 12.35, 0.62, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 0.65, 1.58, 12.1, 0.62, [dict(
    t="**KNOWLEDGE LAYER (RAG)**   SEP curriculum & learning objectives · Luca lesson library & pedagogy style "
      "guide · brand/asset & motion-template library · performance data (watch time, quiz results)",
    size=10, color=WHITE)], anchor=MSO_ANCHOR.MIDDLE)
steps = [
    ("Brief agent", "LLM + RAG drafts objective, concepts, misconceptions, scene outline", "Author approves",
     "1.0 → 0.3 h"),
    ("Script agent + QA", "Writes in Luca template; auto-checks length, reading level, accuracy, safety",
     "Scriptwriter edits & locks", "4.0 → 1.5 h"),
    ("Scene spec (JSON)", "Per scene: narration, on-screen text, visual prompt, template ID, duration",
     "Single source of truth", "auto"),
    ("Visual assets", "Reuse library first; generate brand-styled images; diagrams & math as code",
     "Designer curates", "3.0 → 1.5 h"),
    ("Motion renderer", "Fills motion templates from scene JSON; bespoke only for Tier A",
     "Motion designer polishes", "5.0 → 2.0 h"),
    ("Avatar, voice & finish", "Synthesia API render, voice-over, music, loudness, captions, sync check",
     "Final QA sign-off", "6.0 → 2.0 h"),
]
x, w = 0.5, 2.0
for i, (head, body, gate, save) in enumerate(steps):
    xx = x + i * 2.07
    shp = rect(s, xx, 2.32, w + 0.12, 0.55, TEAL, shape=MSO_SHAPE.CHEVRON if i else MSO_SHAPE.PENTAGON)
    text(s, xx + 0.18, 2.32, w - 0.2, 0.55, [dict(t=f"{i + 1}. {head}", size=9.8, bold=True, color=WHITE,
                                                  align=PP_ALIGN.CENTER)], anchor=MSO_ANCHOR.MIDDLE)
    rect(s, xx + 0.04, 2.95, w - 0.04, 1.35, GREY_LT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    text(s, xx + 0.1, 2.98, w - 0.16, 0.95, [dict(t=body, size=8.8)])
    pill(s, xx + 0.12, 3.92, w - 0.2, 0.28, gate, fill=AMBER_LT, color=INK, size=8)
    text(s, xx + 0.04, 4.33, w - 0.04, 0.25, [dict(t=f"Effort/unit: {save}", size=8.5, bold=True, color=TEAL,
                                                   align=PP_ALIGN.CENTER)])
rows = [["Layer", "Engine & sourcing (buy best-of-breed APIs, build only the thin orchestration layer)"],
        ["LLMs: brief, script, QA", "Claude for long-form pedagogy in Spanish & Portuguese; GPT / Gemini through a model "
         "router for structured JSON and second-opinion QA. pgvector RAG over the curriculum; 50 gold scripts as eval set."],
        ["Visual assets", "Adobe Firefly / Ideogram / Flux (commercially safe image generation) with brand style "
         "presets; SVG + Manim for diagrams and math; Figma asset library."],
        ["Motion", "Remotion (programmatic React video) or After Effects templates rendered via Nexrender / "
         "Plainly from the scene JSON; Runway / Veo for short B-roll clips."],
        ["Avatar, voice, sound", "Synthesia API (current tool, Luca avatars); ElevenLabs for voice-over and LatAm "
         "dubbing; licensed music API; ffmpeg loudness (EBU R128); Whisper captions + script-vs-audio sync check."],
        ["Orchestration & tracking", "n8n for the MVP → Temporal at scale; status written to the master tracker "
         "via API; review UI in Lovable / Retool with comments per scene; S3 storage with versioning."]]
table(s, 0.5, 4.68, 8.45, 2.3, rows, col_w=[1.75, 6.7], size=8.6, bold_first_col=True)
panel(s, 9.1, 4.68, 3.75, 2.3, "Impact, guardrails & prototype", [
    dict(t=f"**≈{hours_per_video:.0f} h → ~7 h of human effort per video** (−60%), so the same team can "
           "deliver 2–3× the output.", size=9, bullet=True),
    dict(t="**Guardrails:** human sign-off on brief and final; child-safety, bias and accuracy checks; "
           "an IP/licence log per asset; cost per video tracked.", size=9, bullet=True),
    dict(t="**Prototype:** no link in this version. I propose a 2-week MVP (Lovable front end + Claude API): "
           "topic → brief → script → scene JSON → Synthesia draft, piloted on Primaria Math.",
         size=9, bullet=True),
], fill=TEAL_LT, accent=TEAL, size=9, head_size=11)

# ------------------------------------------------------------------------------------------- 4
s = base_slide(prs, "2 · Tracking & operating rhythm",
               "Every Friday we know whether we're on track: one tracker, five fields, three meetings", 4,
               "Template: Luca_Content_Pipeline_Master_Tracker.xlsx (pre-loaded with the Week 1 snapshot)")
text(s, 0.5, 1.58, 6.0, 0.3, [dict(t="Master tracker: 5 key fields beyond Unit ID & Stage", size=12, bold=True,
                                   color=NAVY)])
rows = [["Field", "What it captures", "How it spots delays / drives accountability"],
        ["1. Accountable owner", "One named person for the unit's current stage",
         "No shared ownership: every late unit has a name attached, and the stand-up goes owner by owner."],
        ["2. Stage due date (SLA)", "Stage start + SLA (e.g. Script 2 days, Motion 2 days)",
         "Auto-calculates 'days late'; red the moment a unit misses its SLA, not at week end."],
        ["3. Days in stage", "Today − stage start date (aging)",
         "Shows silent stalls before anyone flags a blocker; >2 days turns amber."],
        ["4. Status + blocker reason", "On track / At risk / Blocked + a standard reason list (e.g. content "
         "misalignment, change after lock, A/V sync)",
         "Blockers by reason × stage show the system's failure modes; blocked >24 h is escalated."],
        ["5. Rework loops", "Number of times the unit was sent back to an earlier stage",
         "Measures quality at hand-offs; ≥2 loops triggers a root-cause look in the weekly review."]]
table(s, 0.5, 1.95, 6.1, 4.25, rows, col_w=[1.45, 2.0, 2.65], size=8.8, bold_first_col=True)
rect(s, 0.5, 6.3, 6.1, 0.68, GREY_LT, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
text(s, 0.62, 6.3, 5.9, 0.68, [dict(
    t="**Weekly scoreboard (auto, in the tracker):** finished vs. 30 target, cumulative vs. plan, run-rate needed "
      "to hit 300, WIP and blockers per stage, average days in stage.", size=9.2)], anchor=MSO_ANCHOR.MIDDLE)

text(s, 6.85, 1.58, 6.0, 0.3, [dict(t="Weekly operating rhythm (3 recurring meetings)", size=12, bold=True,
                                    color=NAVY)])
mtgs = [
    ("Daily production stand-up", "Mon–Fri · 15 min", TEAL, TEAL_LT,
     "Stage leads (content, script, design, motion, Synthesia/sound) + PMO",
     "Walk the board right-to-left (closest to done first): units blocked >24 h, queues over the WIP limit, "
     "late units.",
     "Who unblocks what today; capacity swaps between stages; escalations to the Curriculum Lead."),
    ("Weekly pipeline review", "Monday · 45 min", NAVY, GREY_LT,
     "Chief of Staff (chair), Head of Content Ops, Curriculum Lead, all stage leads",
     "Throughput vs. plan, cycle time per stage, rework rate, top blocker reasons, change requests, "
     "next week's capacity.",
     "Weekly finish commitment per stage; WIP limits; tier mix; approve/deny post-lock changes; "
     "freelance hours."),
    ("Bi-weekly steering", "Every other Thursday · 30 min", CORAL, CORAL_LT,
     "CEO / COO, Head of Product / Content, Chief of Staff",
     "Forecast to 300 vs. term start, budget vs. actual, AI tool MVP progress, risks.",
     "Scope trade-offs (subject priority, Tier C share), budget for capacity, AI tool go/no-go."),
]
y = 1.95
for name, when, acc, fill, who, review, decide in mtgs:
    rect(s, 6.85, y, 6.0, 1.52, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    rect(s, 6.85, y + 0.12, 0.07, 1.28, acc)
    text(s, 7.02, y + 0.06, 3.6, 0.32, [dict(t=name, size=11.5, bold=True, color=NAVY)])
    pill(s, 10.55, y + 0.1, 2.2, 0.27, when, fill=acc, size=8.5)
    text(s, 7.02, y + 0.4, 5.75, 1.1, [
        dict(t=f"**Who:** {who}", size=9.2, space=2),
        dict(t=f"**Reviews:** {review}", size=9.2, space=2),
        dict(t=f"**Decides:** {decide}", size=9.2, space=0)])
    y += 1.6
text(s, 6.85, 6.75, 6.0, 0.25, [dict(t="Plus a Friday 1-page status note (not a meeting) to "
                                                  "leadership: on track / at risk, top 3 risks, asks.", size=8.5,
                                                  italic=True, color=GREY)])

prs.save(OUT)
print("saved", OUT)
