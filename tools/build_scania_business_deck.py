from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
OUTPUT_DIR = ROOT / "slides" / "image" / "scania-aps-business-deck"
OUTPUT = OUTPUT_DIR / "scania-aps-business-deck.pptx"

W, H = 13.333, 7.5
BG = "090A0F"
PANEL = "14131C"
PANEL2 = "1B1826"
WHITE = "F7F5FA"
MUTED = "B7B1C2"
PURPLE = "7752B8"
PURPLE2 = "A784E5"
CYAN = "7DD3FC"
GREEN = "75D69A"
AMBER = "F5C76D"
RED = "F18B9A"
LINE = "332B49"


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


def add_shape(slide, shape_type, x, y, w, h, fill=BG, line=LINE, width=1):
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(width)
    return shape


def add_text(slide, text, x, y, w, h, size=14, color=WHITE, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE, italic=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.04)
    frame.margin_right = Inches(0.04)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    frame.vertical_anchor = valign
    p = frame.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    font = run.font
    font.name = "Aptos"
    font.size = Pt(size)
    font.bold = bold
    font.italic = italic
    font.color.rgb = rgb(color)
    return box


def add_bullets(slide, items, x, y, w, h, size=13, color=WHITE, gap=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.08)
    frame.margin_right = Inches(0.05)
    frame.margin_top = Inches(0.03)
    frame.margin_bottom = Inches(0.03)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, item in enumerate(items):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = item
        p.level = 0
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(gap)
        p.bullet = True
    return box


def add_title(slide, title, kicker):
    add_text(slide, kicker.upper(), 0.62, 0.34, 6.6, 0.18, 7.5, PURPLE2, True)
    add_text(slide, title, 0.60, 0.62, 11.9, 0.48, 26, WHITE, True)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.62), Inches(1.25), Inches(1.45), Inches(0.04))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb(PURPLE)
    line.line.fill.background()


def add_footer(slide, number):
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(7.05), Inches(12.2), Inches(0.01))
    line.fill.solid()
    line.fill.fore_color.rgb = rgb("29233A")
    line.line.fill.background()
    add_text(slide, "APS FAILURE AT SCANIA TRUCKS", 0.58, 7.12, 4.5, 0.15, 6.5, "8D879A", True)
    add_text(slide, str(number), 12.33, 7.10, 0.35, 0.15, 7, "8D879A", False, PP_ALIGN.RIGHT)


def add_base_slide(prs, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    background = slide.background.fill
    background.solid()
    background.fore_color.rgb = rgb(BG)
    add_footer(slide, number)
    return slide


def add_card(slide, x, y, w, h, title, value, note, accent=PURPLE):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, PANEL, LINE)
    accent_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.065), Inches(h))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = rgb(accent)
    accent_bar.line.fill.background()
    add_text(slide, title.upper(), x + 0.22, y + 0.17, w - 0.32, 0.18, 7, MUTED, True)
    add_text(slide, value, x + 0.22, y + 0.43, w - 0.32, 0.42, 22, WHITE, True)
    add_text(slide, note, x + 0.22, y + h - 0.40, w - 0.32, 0.22, 8.5, MUTED)


def add_callout(slide, text, x, y, w, h, accent=PURPLE, size=12):
    add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h, PANEL2, accent, 1.2)
    add_text(slide, text, x + 0.22, y + 0.12, w - 0.42, h - 0.22, size, WHITE, False)


def add_table(slide, rows, x, y, w, h, widths, font_size=8.3, header=True, row_height=0.38):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    for i, col_width in enumerate(widths):
        table.columns[i].width = Inches(col_width)
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(row_height)
        for c, value in enumerate(row):
            cell = table.cell(r, c)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb("1B1826" if r % 2 else PANEL)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.04)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.border_left = None
            for paragraph in cell.text_frame.paragraphs:
                paragraph.alignment = PP_ALIGN.LEFT
                for run in paragraph.runs:
                    run.font.name = "Aptos"
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = rgb(PURPLE2 if header and r == 0 else WHITE)
                    run.font.bold = bool(header and r == 0)
    return table_shape


def add_image(slide, path, x, y, w=None, h=None):
    kwargs = {"left": Inches(x), "top": Inches(y)}
    if w is not None:
        kwargs["width"] = Inches(w)
    if h is not None:
        kwargs["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), **kwargs)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "APS Failure at Scania Trucks"
    prs.core_properties.subject = "Cost-sensitive APS component failure classification"
    prs.core_properties.author = "Klopi710"
    prs.core_properties.company = "Klopi710"

    # 1. Conclusion
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(BG)
    add_shape(slide, MSO_SHAPE.RECTANGLE, 8.70, 0, 4.63, 7.5, "171225", "171225")
    add_text(slide, "APS FAILURE\nAT SCANIA TRUCKS", 0.70, 0.70, 7.0, 1.20, 29, WHITE, True)
    add_text(slide, "A cost-sensitive inspection-prioritisation recommendation", 0.72, 2.23, 6.5, 0.30, 15, MUTED)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.72), Inches(2.92), Inches(1.5), Inches(0.04))
    line.fill.solid(); line.fill.fore_color.rgb = rgb(PURPLE2); line.line.fill.background()
    add_text(slide, "Use Random Forest to rank trucks by estimated APS-failure risk and send the highest-risk cases to mechanic inspection.", 0.72, 3.34, 7.0, 0.88, 18, WHITE, True)
    add_callout(slide, "Official frozen test: 97.6% recall at the development-selected cost-aware threshold.\nThe model does not automatically declare a truck safe or unsafe.", 0.72, 4.62, 7.0, 1.15, PURPLE)
    add_text(slide, "Cost avoided versus inspecting nobody: 176,250 challenge cost units", 0.72, 6.20, 7.2, 0.28, 10, AMBER, True)
    add_footer(slide, 1)

    # 2. Operational case
    slide = add_base_slide(prs, 2)
    add_title(slide, "The operational case", "01 / decision summary")
    add_card(slide, 0.65, 1.65, 2.85, 1.55, "Official test cases", "16,000", "375 APS failures", CYAN)
    add_card(slide, 3.72, 1.65, 2.85, 1.55, "Inspections", "1,041", "6.5% of test trucks", PURPLE)
    add_card(slide, 6.79, 1.65, 2.85, 1.55, "Precision", "35.2%", "366 of 1,041 inspections", GREEN)
    add_card(slide, 9.86, 1.65, 2.85, 1.55, "Recall", "97.6%", "366 of 375 failures", AMBER)
    add_text(slide, "The asymmetric cost matters", 0.68, 3.75, 4.4, 0.30, 18, WHITE, True)
    add_text(slide, "A missed APS failure costs 500 challenge units. An unnecessary mechanic inspection costs 10. Missing a failure is therefore 50 times more expensive.", 0.70, 4.25, 5.70, 0.95, 15, MUTED)
    add_callout(slide, "Recommended official-test cost: 11,250 units\nCost avoided versus the all-negative policy: 176,250 units", 7.0, 3.72, 5.3, 1.15, CYAN, 15)
    add_text(slide, "These are challenge-unit calculations, not measured realised savings or proof that an inspection caused a breakdown to be avoided.", 7.04, 5.35, 5.15, 0.65, 11, AMBER)

    # 3. Model comparison
    slide = add_base_slide(prs, 3)
    add_title(slide, "Why Random Forest?", "02 / model selection")
    add_text(slide, "Candidates were compared on the same development data using out-of-fold predictions and the asymmetric Scania cost criterion.", 0.72, 1.45, 11.7, 0.30, 12, MUTED)
    add_table(slide, [
        ["Model", "ROC-AUC", "Average precision", "Best dev. cost", "Cost avoided"],
        ["Random Forest", "0.983", "0.705", "46,220", "453,780"],
        ["Logistic Regression", "0.968", "0.668", "67,900", "432,100"],
        ["Decision Tree", "0.932", "0.567", "82,280", "417,720"],
        ["Majority baseline", "0.500", "0.017", "500,000", "0"],
    ], 0.72, 2.05, 11.85, 2.15, [3.05, 1.65, 2.25, 2.30, 2.60], 10, True, 0.42)
    add_callout(slide, "Random Forest wins on the development selection criterion. The official test is used once for frozen confirmation, not for choosing the model or threshold.", 0.72, 4.72, 11.85, 0.82, PURPLE, 14)
    add_text(slide, "Selection is based on ranking and operational cost, not accuracy alone.", 0.75, 5.95, 9.0, 0.28, 15, CYAN, True)

    # 4. Dataset and target
    slide = add_base_slide(prs, 4)
    add_title(slide, "What the database contains", "03 / data and target")
    add_card(slide, 0.72, 1.60, 2.65, 1.50, "Training rows", "60,000", "1,000 positive", CYAN)
    add_card(slide, 3.58, 1.60, 2.65, 1.50, "Official test", "16,000", "375 positive", PURPLE)
    add_card(slide, 6.44, 1.60, 2.65, 1.50, "Predictors", "170", "171 attributes incl. class", GREEN)
    add_card(slide, 9.30, 1.60, 2.65, 1.50, "Source", "UCI", "Scania APS dataset", AMBER)
    add_text(slide, "Target definition", 0.75, 3.72, 3.0, 0.30, 18, WHITE, True)
    add_text(slide, "pos = failure of the APS component in focus.\nneg = a failure of another component, unrelated to the APS component.", 0.75, 4.15, 5.55, 0.78, 15, MUTED)
    add_text(slide, "Important limitation", 7.05, 3.72, 3.2, 0.30, 18, WHITE, True)
    add_text(slide, "The class is a benchmark label, not an observed future time-stamped failure outcome. The feature names are anonymised, so the public release does not identify the physical meaning of each aa_000-style column.", 7.05, 4.15, 5.1, 1.15, 14, MUTED)
    add_text(slide, "One row = one selected truck operating example; NaN = missing measurement.", 0.75, 5.95, 11.4, 0.28, 12, AMBER, True)

    # 5. Preparation
    slide = add_base_slide(prs, 5)
    add_title(slide, "How the data is prepared", "04 / leakage control")
    steps = [
        ("Split", "60,000 labelled training rows for development; 16,000 official test rows remain frozen."),
        ("Missing values", "Median imputation plus missingness indicators inside the modelling pipeline."),
        ("Outliers", "IQR for skewed/non-normal features; z-score +/-3 for approximately symmetric features."),
        ("No deletion", "Extreme values are capped inside each training fold because they may carry operational signal."),
    ]
    for i, (title, body) in enumerate(steps):
        y = 1.58 + i * 1.07
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.78, y, 11.75, 0.78, PANEL if i % 2 == 0 else PANEL2, LINE)
        add_text(slide, str(i + 1), 1.03, y + 0.19, 0.35, 0.22, 14, PURPLE2, True, PP_ALIGN.CENTER)
        add_text(slide, title, 1.60, y + 0.13, 2.6, 0.25, 14, WHITE, True)
        add_text(slide, body, 4.18, y + 0.12, 7.85, 0.38, 12, MUTED)
    add_callout(slide, "All preprocessing bounds are learned only from the relevant training fold. This prevents information from the frozen test leaking into model development.", 0.78, 6.03, 11.75, 0.53, CYAN, 11)

    # 6. Policy mechanics
    slide = add_base_slide(prs, 6)
    add_title(slide, "How the inspection policy works", "05 / operating rule")
    xs = [0.85, 3.55, 6.25, 8.95]
    titles = ["Score", "Rank", "Prioritise", "Inspect"]
    bodies = [
        "Estimate APS-failure risk from the operational features.",
        "Sort trucks from highest to lowest estimated risk.",
        "Use the development-selected cost-aware threshold.",
        "Send flagged cases to a mechanic; keep final authority human.",
    ]
    for i, x in enumerate(xs):
        add_shape(slide, MSO_SHAPE.OVAL, x, 1.72, 0.55, 0.55, PURPLE if i == 3 else PANEL2, PURPLE2 if i == 3 else LINE, 1.5)
        add_text(slide, str(i + 1), x, 1.84, 0.55, 0.18, 12, WHITE, True, PP_ALIGN.CENTER)
        if i < 3:
            connector = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 0.72), Inches(1.86), Inches(1.65), Inches(0.22))
            connector.fill.solid(); connector.fill.fore_color.rgb = rgb(LINE); connector.line.fill.background()
        add_text(slide, titles[i], x - 0.10, 2.55, 2.0, 0.25, 15, WHITE, True)
        add_text(slide, bodies[i], x - 0.10, 2.90, 2.1, 1.0, 11, MUTED, False, PP_ALIGN.LEFT, MSO_ANCHOR.TOP)
    add_callout(slide, "The model does not automatically declare a truck safe or unsafe. It prioritises scarce mechanic attention.", 0.85, 4.52, 11.7, 0.92, CYAN, 14)
    add_text(slide, "Recommended threshold: score >= 0.146073 on the official frozen test.", 0.85, 5.88, 8.6, 0.28, 14, AMBER, True)

    # 7. Metrics
    slide = add_base_slide(prs, 7)
    add_title(slide, "What the official test results mean", "06 / interpretation")
    add_card(slide, 0.75, 1.58, 3.45, 1.78, "Precision", "35.2%", "366 of 1,041 flagged trucks were APS failures", GREEN)
    add_card(slide, 4.95, 1.58, 3.45, 1.78, "Recall", "97.6%", "366 of 375 labelled APS failures were captured", AMBER)
    add_card(slide, 9.15, 1.58, 3.45, 1.78, "Missed failures", "9", "9 false negatives remain", RED)
    add_text(slide, "The trade-off", 0.78, 4.02, 2.6, 0.28, 18, WHITE, True)
    add_text(slide, "The policy catches nearly every labelled APS failure, but 675 of the 1,041 inspections are unnecessary under the historical labels. That is acceptable only because a missed failure costs 50 times more than an unnecessary check in this challenge.", 0.78, 4.45, 6.0, 1.05, 14, MUTED)
    add_image(slide, FIGURES / "recommended-confusion-matrix.png", 8.0, 3.90, 4.3, 2.75)

    # 8. Sensitivity
    slide = add_base_slide(prs, 8)
    add_title(slide, "Does the recommendation survive different costs?", "07 / sensitivity")
    add_text(slide, "Sensitivity changes the cost assumptions, not the model predictions. The selected policy remains cost-saving in each scenario below.", 0.72, 1.45, 11.6, 0.35, 12, MUTED)
    add_table(slide, [
        ["Scenario", "Inspection cost", "Missed-failure cost", "Total cost", "Cost avoided"],
        ["Official base", "10", "500", "11,250", "176,250"],
        ["Lower missed-failure cost", "10", "350", "9,900", "121,350"],
        ["Severe missed-failure cost", "10", "750", "13,500", "267,750"],
        ["Expensive inspection", "50", "500", "38,250", "149,250"],
    ], 0.72, 2.05, 11.85, 2.20, [3.25, 2.0, 2.4, 1.8, 2.4], 10, True, 0.42)
    add_callout(slide, "The result is conditional challenge-unit arithmetic. It does not establish real-world profitability or causal failure prevention.", 0.72, 4.72, 11.85, 0.78, AMBER, 14)
    add_text(slide, "Management question: is the policy directionally robust if the assumed costs are wrong? In this analysis, yes.", 0.75, 5.93, 11.4, 0.30, 13, CYAN, True)

    # 9. Governance
    slide = add_base_slide(prs, 9)
    add_title(slide, "Controls before any rollout", "08 / governance")
    controls = [
        ("Validate the outcome", "Replace the benchmark label with observed prospective failure and inspection outcomes where possible."),
        ("Test prospectively", "Check whether historical rates represent future truck populations and operating conditions."),
        ("Monitor the queue", "Track false negatives, inspection workload, missingness, calibration and drift."),
        ("Keep human authority", "No automatic safe/unsafe decision; define escalation and mechanic review procedures."),
    ]
    for i, (title, body) in enumerate(controls):
        y = 1.55 + i * 1.08
        add_shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, 0.78, y, 11.75, 0.82, PANEL if i % 2 == 0 else PANEL2, LINE)
        add_text(slide, str(i + 1), 1.03, y + 0.20, 0.35, 0.22, 14, PURPLE2, True, PP_ALIGN.CENTER)
        add_text(slide, title, 1.60, y + 0.15, 2.75, 0.25, 14, WHITE, True)
        add_text(slide, body, 4.42, y + 0.13, 7.55, 0.38, 12, MUTED)
    add_callout(slide, "The subgroup audit is diagnostic only. The dataset has no protected-group attributes; missingness burden is not a fairness certification.", 0.78, 6.03, 11.75, 0.53, AMBER, 11)

    # 10. Recommendation
    slide = add_base_slide(prs, 10)
    add_title(slide, "Recommendation", "09 / decision")
    add_text(slide, "Use Random Forest as an inspection-prioritisation aid alongside the existing maintenance workflow.", 0.72, 1.60, 9.8, 0.58, 24, WHITE, True)
    add_callout(slide, "At the official frozen-test cost-aware point:\n366 true APS failures captured\n9 labelled APS failures missed\n1,041 mechanic inspections", 0.75, 2.72, 5.60, 1.72, PURPLE, 16)
    add_bullets(slide, [
        "Approve only a controlled pilot, not autonomous maintenance decisions.",
        "Record the actual inspection result and subsequent failure outcome.",
        "Re-estimate costs using real mechanic time and real breakdown exposure.",
        "Revalidate before expanding beyond the tested operating population.",
    ], 7.0, 2.72, 5.25, 2.2, 13, WHITE, 8)
    add_text(slide, "The model is promising for prioritising scarce attention, but the dataset alone does not prove causal breakdown prevention.", 0.75, 5.55, 11.0, 0.40, 17, AMBER, True)
    add_text(slide, "Thank you", 0.75, 6.18, 11.4, 0.42, 25, CYAN, True)

    # Appendix A1. Frozen-test evidence
    slide = add_base_slide(prs, 11)
    add_title(slide, "Frozen-test evidence", "Appendix A1 / technical detail")
    add_text(slide, "The model and policy were selected on development data. The separate official test was then scored once for confirmation.", 0.72, 1.45, 11.7, 0.30, 12, MUTED)
    add_image(slide, FIGURES / "recommended-confusion-matrix.png", 0.78, 1.90, 5.25, 4.45)
    add_table(slide, [
        ["Metric", "Official test"],
        ["TN", "14,950"], ["FP", "675"], ["FN", "9"], ["TP", "366"],
        ["Precision", "35.2%"], ["Recall", "97.6%"], ["F1", "51.7%"],
        ["Total challenge cost", "11,250"], ["Cost avoided", "176,250"],
    ], 6.45, 1.92, 5.95, 4.18, [3.65, 2.30], 10, True, 0.38)

    # Appendix A2. Curves
    slide = add_base_slide(prs, 12)
    add_title(slide, "Ranking quality across cutoffs", "Appendix A2 / technical detail")
    add_text(slide, "ROC-AUC and average precision evaluate score ordering across many possible cutoffs; the policy table evaluates concrete inspection rules.", 0.72, 1.45, 11.7, 0.30, 12, MUTED)
    add_image(slide, FIGURES / "roc-pr-curves.png", 0.72, 1.92, 7.45, 4.65)
    add_image(slide, FIGURES / "cost-by-threshold.png", 8.45, 2.03, 4.15, 3.75)
    add_callout(slide, "Development Random Forest: ROC-AUC 0.983; average precision 0.705.\nThe model is judged by ranking quality plus the cost of the chosen operating point.", 8.45, 5.98, 4.15, 0.55, CYAN, 10)

    # Appendix A3. Operating point table
    slide = add_base_slide(prs, 13)
    add_title(slide, "Required operating-point table", "Appendix A3 / threshold analysis")
    add_text(slide, "The threshold table makes the trade-off explicit. The high-recall and mid-range rows exceed the illustrative 10% inspection capacity; the cost-aware row is the recommendation.", 0.72, 1.42, 11.7, 0.36, 11, MUTED)
    add_table(slide, [
        ["Operating point", "Rule", "Inspections", "TP / FP / FN / TN", "Precision", "Recall", "Total cost"],
        ["Default", "score >= 0.50", "523", "326 / 197 / 49 / 15,428", "62.3%", "86.9%", "26,470"],
        ["High recall", "score >= 0.310875", "744", "353 / 391 / 22 / 15,234", "47.4%", "94.1%", "14,910"],
        ["Mid-range", "score >= 0.20", "925", "365 / 560 / 10 / 15,065", "39.5%", "97.3%", "10,600"],
        ["High precision", "score >= 0.810244", "222", "194 / 28 / 181 / 15,597", "87.4%", "51.7%", "90,780"],
        ["Cost-aware", "score >= 0.146073", "1,041", "366 / 675 / 9 / 14,950", "35.2%", "97.6%", "11,250"],
        ["Exact top-10%", "rank highest 10%", "1,600", "370 / 1,230 / 5 / 14,395", "23.1%", "98.7%", "14,800"],
    ], 0.52, 2.02, 12.3, 3.78, [2.02, 1.48, 1.05, 3.28, 1.10, 0.92, 1.35], 7.3, True, 0.39)
    add_callout(slide, "The threshold is a reference boundary. If capacity is fixed, production should use an exact ranking rule instead of assuming a fixed number of inspections from a probability cutoff.", 0.72, 6.10, 11.85, 0.45, PURPLE, 10)

    # Appendix A4. Reproducibility, outliers and preview
    slide = add_base_slide(prs, 14)
    add_title(slide, "Reproducibility and data reading guide", "Appendix A4 / implementation detail")
    add_text(slide, "How to read one raw row", 0.72, 1.45, 3.8, 0.28, 16, WHITE, True)
    add_table(slide, [
        ["Raw field", "Reader-friendly meaning"],
        ["class", "Target: pos = APS failure; neg = unrelated failure"],
        ["aa_000 / ab_000", "Anonymised operational value; physical meaning not disclosed"],
        ["ag_000 / ag_001", "Anonymised operational value or histogram-bin value"],
        ["NaN", "Missing measurement; imputed inside the modelling pipeline"],
    ], 0.72, 1.87, 5.85, 2.18, [1.70, 4.15], 9, True, 0.42)
    add_text(slide, "Reproducibility controls", 7.05, 1.45, 3.8, 0.28, 16, WHITE, True)
    add_bullets(slide, [
        "Seed: 77.",
        "Development: 60,000 labelled training rows with out-of-fold selection.",
        "Confirmation: 16,000 official test rows scored once after freezing.",
        "Outliers: 118 IQR rules, 6 z-score rules and 46 insufficient-variation features.",
        "Source notebook: notebooks/scania_aps_failure.ipynb.",
    ], 7.05, 1.87, 5.45, 2.30, 11, WHITE, 5)
    add_image(slide, FIGURES / "outlier-flags.png", 0.72, 4.42, 5.85, 1.85)
    add_callout(slide, "The renamed preview and column guide are saved under data/processed/. The public dataset intentionally anonymises the physical feature names.", 7.05, 4.52, 5.45, 1.00, CYAN, 13)

    prs.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
