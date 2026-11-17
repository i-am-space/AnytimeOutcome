"""Add the Evaluation 2 evidence update to the preserved October report.

Only reads saved measurements. This builder never starts a GPU worker and does
not modify campaign inputs, source fingerprints, or historical artifacts.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
SNAP = OUT / "evidence_snapshot"
for directory in (FIG, SNAP):
    directory.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "B1": "results/next_stage/shared_uniform_v1_2026-11-17/baselines/summary.json",
    "B2": "results/next_stage/shared_uniform_v1_2026-11-17/baselines/selection.json",
    "B3": "results/next_stage/shared_uniform_v1_2026-11-17/baselines/manifest.json",
    "U1": "results/next_stage/overnight_extensions_v2_2026-11-17/heldout/summary.json",
    "U2": "results/next_stage/overnight_extensions_v2_2026-11-17/depth_tune/summary.json",
    "O1": "results/next_stage/prefix_overlap_v1_2026-11-17/correctness/summary.json",
    "O2": "results/next_stage/prefix_overlap_v1_2026-11-17/stress/summary.json",
    "A1": "results/next_stage/adaptive_publication_v1_2026-11-17/correctness/summary.json",
    "L1": "results/next_stage/publication_performance_v1_2026-11-17/status.json",
    "D1": "docs/prefix_publication.md",
    "D2": "docs/publication_performance_campaign.md",
    "D3": "docs/shared_fast_backend.md",
    "D4": "docs/adaptive_publication.md",
    "T1": "results/next_stage/publication_performance_v1_2026-11-17/cpu_tests.log",
    "T2": "results/next_stage/prefix_overlap_v1_2026-11-17/gpu_tests.log",
    "T3": "results/next_stage/publication_performance_v1_2026-11-17/campaign_tests.log",
    "H1": "reports/evaluation1_2026-10-22_update/Anytime_SSD_Performance_Report.pdf",
}

# Snapshot once. Rebuilding the PDF does not silently change its evidence date.
manifest_path = OUT / "evidence_manifest.json"
if not manifest_path.exists():
    entries = {}
    for key, source in SOURCES.items():
        data = (ROOT / source).read_bytes()
        target = SNAP / (key + Path(source).suffix)
        target.write_bytes(data)
        entries[key] = {
            "source": source,
            "snapshot": str(target.relative_to(OUT)),
            "sha256": sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest_path.write_text(json.dumps({
        "captured_at_ist": datetime.now(ZoneInfo("Asia/Calcutta")).isoformat(),
        "sources": entries,
        "rubric_source": "User-supplied Evaluation 2 requirements in this conversation",
    }, indent=2) + "\n")
manifest = json.loads(manifest_path.read_text())
for entry in manifest["sources"].values():
    assert sha256((OUT / entry["snapshot"]).read_bytes()).hexdigest() == entry["sha256"]


def data(key):
    return json.loads((OUT / manifest["sources"][key]["snapshot"]).read_text())


baseline, selection, workload = data("B1"), data("B2"), data("B3")
uniform, depths, normal, stress, adaptive, live = (data(k) for k in ("U1", "U2", "O1", "O2", "A1", "L1"))
assert baseline["status"] == "complete"
assert (baseline["tuning_requests"], baseline["heldout_requests"]) == (480, 2250)
assert uniform["performance_evidence"] and uniform["exact_matches"] == 1500
assert not normal["performance_evidence"] and not stress["performance_evidence"]
assert normal["exact_matches"] + stress["exact_matches"] == 102
assert normal["requests"] + stress["requests"] == 120
for row in baseline["results"]:
    assert row["correctness"]["requests_matching_paired_ar"] == row["statistics"]["requests"]

NAVY, TEAL, BLUE, ORANGE = "#152E49", "#008C88", "#4778AE", "#D98225"
INK, GRAY, LIGHT, RED = "#26394B", "#697B8E", "#EFF4F8", "#C64D56"
for name, filename in (("Body", "DejaVuSans.ttf"), ("Bold", "DejaVuSans-Bold.ttf")):
    pdfmetrics.registerFont(TTFont(name, "/usr/share/fonts/truetype/dejavu/" + filename))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Bold", italic="Body", boldItalic="Bold")
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelcolor": INK, "text.color": INK, "axes.edgecolor": "#B5C0CA",
    "xtick.color": INK, "ytick.color": INK, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
})


def base(protocol, mode, family="all"):
    return next(r for r in baseline["results"] if (r["protocol"], r["mode"], r["family"]) == (protocol, mode, family))


def uni(protocol, label):
    return next(r for r in uniform["statistics"] if r["protocol"] == protocol and r["config"]["label"] == label)


def save(fig, name):
    path = FIG / (name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def baseline_plot(protocol):
    fig, ax = plt.subplots(figsize=(7.0, 3.25))
    rows = [base(protocol, m) for m in ("ar", "sd", "ssd")]
    values = [r["statistics"]["mean_latency_s"] for r in rows]
    bars = ax.bar(["Shared AR\n1 GPU", "SD K=6\n1 GPU", "SSD K=7/F=3\n2 GPUs"], values, color=[GRAY, TEAL, BLUE], width=.57)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value+.12, f"{value:.3f} s", ha="center", weight="bold")
    ax.set(ylabel="Mean full-request latency (s)", ylim=(0, max(values)*1.2))
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return save(fig, "baseline_" + protocol)


fixed_plot, natural_plot = baseline_plot("fixed"), baseline_plot("natural")
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.5), gridspec_kw={"width_ratios": [1, 1.65]})
for ax, mode in zip(axes, ["sd", "ssd"]):
    rows = [r for r in selection["scores"] if r["config"]["mode"] == mode]
    values = [r["mean_latency_s"] for r in rows]
    labels = [f"K{r['config']['k']}" + (f"\nF{r['config']['f']}" if mode == "ssd" else "") for r in rows]
    winner = min(values)
    ax.bar(labels, values, color=[TEAL if v == winner else "#B8C7D6" for v in values])
    for i, value in enumerate(values):
        ax.text(i, value+.045, f"{value:.2f}", ha="center", fontsize=9)
    ax.set(title=("SD: K sweep" if mode == "sd" else "SSD: K/F grid"), ylabel="Mean request latency (s)", ylim=(0, max(values)*1.2))
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.tight_layout(w_pad=2)
tuning_plot = save(fig, "baseline_tuning")

fig, ax = plt.subplots(figsize=(7.2, 3.4))
x = np.arange(3)
for offset, mode, color in [(-.25, "ar", GRAY), (0, "sd", TEAL), (.25, "ssd", BLUE)]:
    vals = [base("fixed", mode, f)["statistics"]["mean_latency_s"] for f in ["chat", "code", "doc"]]
    ax.bar(x+offset, vals, width=.23, color=color, label=mode.upper() + (" (2 GPUs)" if mode == "ssd" else " (1 GPU)"))
ax.set(xticks=x, xticklabels=["Chat", "Code", "Document"], ylabel="Mean request latency (s)", ylim=(0, 7.5))
ax.legend(frameon=False, ncol=3, fontsize=9)
ax.grid(axis="y", alpha=.15)
ax.set_axisbelow(True)
fig.tight_layout()
family_plot = save(fig, "family_latency")

fig, ax = plt.subplots(figsize=(7.2, 3.35))
keys = ["fixed/all", "natural/all"]
intervals = [uniform["paired_intervals"][k] for k in keys]
values = [r["latency_reduction"]*100 for r in intervals]
lower = [(r["latency_reduction"]-r["latency_reduction_ci95"][0])*100 for r in intervals]
upper = [(r["latency_reduction_ci95"][1]-r["latency_reduction"])*100 for r in intervals]
ax.errorbar(values, [1, 0], xerr=[lower, upper], fmt="o", color=ORANGE, capsize=5, markersize=8)
ax.axvline(0, color=GRAY, ls="--", lw=1)
ax.set(yticks=[1, 0], yticklabels=["Fixed 256", "Natural EOS"], xlabel="Latency reduction from uniform publication (%)", xlim=(-1.8, .6), ylim=(-.6, 1.6))
ax.text(-1.7, 1.42, "Negative values mean slower than complete SSD", fontsize=9, color=RED)
ax.grid(axis="x", alpha=.15)
fig.tight_layout()
publication_plot = save(fig, "uniform_publication_effect")

normal_counts = [int(normal["overlap_coverage"][f"{p}_r0_overlap_k10"]["responses_before_tail_done"]) for p in ["fixed", "natural"]]
assert normal_counts == [323, 310]
fig, ax = plt.subplots(figsize=(6.8, 3.1))
bars = ax.bar(["Fixed 256", "Natural EOS"], normal_counts, color=[TEAL, BLUE], width=.5)
for bar, value in zip(bars, normal_counts):
    ax.text(bar.get_x()+bar.get_width()/2, value+8, str(value), ha="center", weight="bold")
ax.set(ylabel="Responses delivered before tail finished", ylim=(0, 395))
ax.grid(axis="y", alpha=.15)
ax.set_axisbelow(True)
fig.tight_layout()
overlap_plot = save(fig, "actual_overlap")


class PDF:
    w, h = 842, 595

    def __init__(self, path):
        self.c = canvas.Canvas(str(path), pagesize=(self.w, self.h))
        self.c.setTitle("Anytime SSD | Performance evidence and Evaluation 2 update | 17 November 2026")
        self.c.setAuthor("Ishaan Romil; Vishal Rao; Eshaan Sharma")
        self.pages = []
        self.boxes = []

    def para(self, text, x, y, w, size=10.3, color=INK, bold=False, limit=530):
        style = ParagraphStyle("p", fontName="Bold" if bold else "Body", fontSize=size, leading=size*1.37, textColor=HexColor(color))
        p = Paragraph(text, style)
        _, height = p.wrap(w, 2000)
        if y+height > limit:
            raise RuntimeError(f"Page {len(self.pages)} text overflow: {y+height:.1f}>{limit}: {text[:70]}")
        p.drawOn(self.c, x, self.h-y-height)
        self.boxes.append([len(self.pages), x, y, w, height])
        return y+height

    def start(self, title, subtitle, tag="COMPLETED EVIDENCE / SHARED AR REFERENCE"):
        if self.pages:
            self.c.showPage()
        self.pages.append(title)
        self.c.setFillColor(HexColor(NAVY))
        self.c.rect(0, self.h-10, self.w, 10, fill=1, stroke=0)
        self.para("ANYTIME OUTCOME-CACHE CONSTRUCTION", 32, 24, 778, 8.3, TEAL, True)
        self.para(title, 32, 46, 778, 23, NAVY, True, 83)
        self.para(subtitle, 32, 85, 778, 9.4, GRAY, limit=108)
        self.c.setFillColor(HexColor(LIGHT))
        self.c.roundRect(32, self.h-134, 778, 25, 5, fill=1, stroke=0)
        self.para(tag, 42, 116, 758, 8.0, TEAL, True, limit=132)
        self.c.setStrokeColor(HexColor("#D6E0E8"))
        self.c.line(32, 34, 810, 34)
        self.para("EVALUATION 2 UPDATE / 17 NOVEMBER 2026 / SHARED-FIXED16-V1", 32, 570, 730, 7.1, GRAY, limit=589)
        self.para(f"{len(self.pages):02d}", 783, 569, 27, 8, GRAY, True, limit=589)

    def blocks(self, items, x=32, y=153, w=778, size=10.5, gap=13):
        for heading, text in items:
            y = self.para(heading, x, y, w, size, TEAL, True)+4
            y = self.para(text, x, y, w, size)+gap
        return y

    def source(self, text):
        self.para("Evidence: " + text, 32, 539, 778, 7.0, GRAY, limit=560)

    def table(self, headers, rows, x=32, y=153, w=778, widths=None, size=9, padding=7):
        style = ParagraphStyle("cell", fontName="Body", fontSize=size, leading=size*1.3, textColor=HexColor(INK))
        header = ParagraphStyle("head", parent=style, fontName="Bold", textColor=white)
        cells = [[Paragraph(str(c), header) for c in headers]] + [[Paragraph(str(c), style) for c in row] for row in rows]
        table = Table(cells, colWidths=[v*w for v in widths] if widths else [w/len(headers)]*len(headers))
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor(NAVY)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor(LIGHT)]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), padding),
            ("RIGHTPADDING", (0, 0), (-1, -1), padding),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        _, height = table.wrap(w, 2000)
        if y+height > 530:
            raise RuntimeError(f"Page {len(self.pages)} table overflow {y+height:.1f}")
        table.drawOn(self.c, x, self.h-y-height)
        return y+height

    def image(self, path, x, y, w, h):
        self.c.drawImage(str(path), x, self.h-y-h, w, h, preserveAspectRatio=True, anchor="c", mask="auto")

    def chart(self, title, subtitle, path, items, sources, tag="COMPLETED PERFORMANCE MEASUREMENTS / SHARED AR REFERENCE"):
        self.start(title, subtitle, tag)
        self.image(path, 32, 153, 485, 302)
        self.blocks(items, x=540, y=155, w=270, size=10.1)
        self.source(sources)

    def finish(self):
        self.c.save()


pdf = PDF(OUT / "Evaluation2_Progress_Update.pdf")
pdf.start("Performance evidence and Evaluation 2", "Implementation, initial results, lessons, and remaining experiments | Update to the 22 October report", "STATUS: CORE HANDOFF IMPLEMENTED / LATENCY BENEFIT STILL TO BE ESTABLISHED")
pdf.blocks([
    ("What we have built", "A shared fast numerical reference, a completed reproducible baseline campaign, uniform and adaptive publication prototypes, and a stable-prefix handoff that can deliver a response while deeper queued drafting finishes."),
    ("What the completed results show", "SD is 1.98× and SSD is 2.84× faster than shared AR on the fixed-output baseline. SSD uses two GPUs. These are baseline algorithm results. Our first tuned uniform-publication variant was about 1% slower than complete SSD."),
    ("What is established for the new overlap path", "102/102 speculative outputs matched fresh shared AR in its correctness gate. Normal K=10 runs recorded 633 actual responses before tail completion. This demonstrates the mechanism on tested inputs; dedicated performance validation remains pending."),
], x=32, y=153, w=478, size=10.7)
captured = datetime.fromisoformat(live["at"]).astimezone(ZoneInfo("Asia/Calcutta")).strftime("17 Nov, %H:%M IST")
pdf.blocks([
    ("Evaluation 2", "Nov 9–21<br/>10-minute presentation + 5-minute questions<br/>30 marks: 8 + 8 + 8 + 6"),
    ("Campaign snapshot", f"As of {captured}, the new performance campaign is waiting for dedicated GPUs. A contaminated attempt was discarded. No new winner or held-out result exists."),
    ("Reading this report", "Pages 1–16 contain the current update. Pages 17–56 preserve the 22 October report as a labelled historical appendix. Its old pending-status statements describe that earlier date."),
], x=544, y=153, w=266, size=10.2)
pdf.source("B1; U1; O1–O2; L1. Snapshot is dated, not a live dashboard. Full source hashes accompany this PDF.")

pdf.start("How our work maps to Evaluation 2", "The rubric asks what has been built, what has been learned, and what remains to be established.", "ASSESSMENT MAP / COVERAGE OF THE CRITERIA, NOT A PREDICTED GRADE")
pdf.table(["Criterion", "Marks", "Evidence we can present", "Remaining evidence"], [
    ["Implementation &amp; Progress", "8", "Working uniform, bounded/adaptive, and overlap paths; completion events, valid lengths, independent snapshots, safe KV reuse and draining; reproducible campaign.", "Unequal continuation-depth allocation remains to implement."],
    ["Initial Results &amp; Analysis", "8", "480 tuning + 2,250 baseline test requests; 1,500 publication comparisons; exactness and actual overlap counters; useful negative result.", "A dedicated latency comparison for the latest overlap version."],
    ["Comparisons / Trade-offs / Ablations", "8", "K/F tuning; complete vs uniform; fixed vs natural EOS; task families; bounded/adaptive diagnostics.", "Tune K/P/queue bound; compare frozen winners; isolate drain, copy, verification and memory costs."],
    ["Presentation &amp; Technical Clarity", "6", "A simple execution diagram, graphs with GPU counts, explicit numerical reference, and separation of measured results from plans.", "Keep claims scoped; use the 10-minute outline on page 15."],
], widths=[.19, .065, .405, .34], size=9.4)
pdf.para("<b>Total: 30 marks.</b> The evidence supports an implementation-and-initial-results presentation. A successful final acceleration claim is still open.", 32, 478, 778, 10.5)
pdf.source("Rubric supplied by the user; B1; U1; O1–O2; D1–D2. No score estimate is implied.")

pdf.start("Progress since the mid-submission", "Comparison with the saved Evaluation 1 report dated 22 October 2026.", "ENGINEERING AND EXPERIMENTAL PROGRESS / RESEARCH GAIN IS STILL OPEN")
pdf.table(["Area", "22 October snapshot", "17 November update"], [
    ["Profiling", "Machine roofline and Nsight mechanism analysis already available.", "Used those observations to implement and instrument publication policies; old profiles remain historical."],
    ["Baseline measurements", "480 tuning requests; 150 dedicated held-out requests; five repetitions incomplete.", "Fresh shared-backend campaign complete: 480 tuning + 2,250 held-out requests, five repetitions, two protocols."],
    ["Output agreement", "Fast SD/SSD differed from original AR on all three saved prompts.", "Shared numerical reference introduced; 1,500/1,500 speculative baseline outputs match that shared AR."],
    ["Core project feature", "Early publication and unequal-depth scheduling were future work.", "Uniform, bounded/adaptive, and stable-prefix overlap are implemented and tested."],
    ["Project result", "No measured anytime-publication benefit.", "Uniform variant about 1% slower; actual overlap demonstrated; latest latency benefit unmeasured."],
    ["Main work still open", "Build and evaluate the scheduler.", "Finish overlap performance study, implement unequal depths, run final ablations and update conclusions."],
], widths=[.17, .395, .435], size=9.2, padding=6)
pdf.source("H1 pages 1–3; B1; U1; O1–O2. Historical and shared-reference timings must not be pooled.")

pdf.start("What stable-prefix publication changes", "The target can receive a completed prefix while already-queued deeper work finishes.", "IMPLEMENTED / SAFE SNAPSHOT + SEPARATE HANDOFF STREAM")
pdf.table(["Path", "When the target can receive a continuation", "What happens to queued tail work"], [
    ["Complete-cache SSD", "After the complete continuation is available.", "Finish before the complete handoff."],
    ["First uniform / bounded prototypes", "Choose a stopping point, then finish already-submitted work before exposure.", "Drain before handoff; submitted work may erase the intended earlier return."],
    ["Stable-prefix overlap", "Copy the completed eligible prefix into independent snapshot buffers and deliver it on a handoff stream.", "May finish concurrently with transfer / target verification; drain before graph or KV reuse."],
], widths=[.20, .43, .37], size=10)
for i, label in enumerate([
    "Prefix event<br/>complete",
    "Copy valid prefix<br/>to snapshot",
    "Send snapshot<br/>on handoff stream",
    "Target verifies<br/>the valid prefix",
    "Join tail before<br/>buffer / KV reuse",
]):
    x = 32 + i*158
    pdf.c.setFillColor(HexColor(LIGHT))
    pdf.c.roundRect(x, pdf.h-350, 146, 43, 5, fill=1, stroke=0)
    pdf.para(label, x+8, 313, 130, 8.7, NAVY, bold=True, limit=350)
    if i < 4:
        pdf.para("→", x+148, 319, 12, 10, TEAL, bold=True, limit=350)
pdf.blocks([
    ("Example: K=10, minimum prefix P=4", "Four or more tokens are complete → a completion event authorizes a snapshot → send only its declared valid tokens → the target verifies while the submitted tail finishes. This is an execution example, not a measured timeline."),
    ("Safety and limits", "Epoch/context checks reject stale state. Rejection, cache misses, next-tree reuse and request-end barriers remain explicit. Full-request latency includes final draining. There is no kernel preemption, and verification still computes a padded K+1 graph."),
], y=366, size=10.2)
pdf.source("D1; O1–O2. A separate stream enables overlap opportunities; actual event counters establish whether overlap occurred.")

pdf.start("Correctness uses an explicit reference", "Identical model weights do not guarantee identical greedy outputs under different finite-precision execution paths.", "EXACTNESS IS AGAINST SHARED-FIXED16-V1 AR / ORIGINAL AR REMAINS DISTINCT")
pdf.blocks([
    ("Why the original outputs differed", "Single-query and batched execution used different numerical paths. BF16 rounding and reduction-order differences can change nearly tied logits, flip the next greedy token, and then change the rest of the response. The original implementation also exhibited discrepancies."),
    ("What we changed", "AR, SD and SSD use shared fixed-shape target arithmetic and batch-independent attention. The shared AR reference is versioned explicitly. It can generate different words from original AR; matching it does not repair historical-original-AR equivalence."),
], x=32, y=153, w=475, size=10.5)
pdf.table(["Completed check", "Exact matches"], [
    ["SD/SSD baseline held-out requests", "1,500 / 1,500"],
    ["Uniform-publication comparison", "1,500 / 1,500"],
    ["Bounded/adaptive diagnostic", "84 / 84"],
    ["Overlap normal gate", "96 / 96"],
    ["Overlap artificial-delay gate", "6 / 6"],
], x=540, y=153, w=270, widths=[.69, .31], size=9, padding=6)
pdf.para("<b>Testing:</b> latest CPU suite: 201 passed, 16 CUDA-specific skips; standalone paged-attention GPU test excluded from that CPU invocation. Separate targeted GPU validation: 16 passed. Campaign-focused tests: 18 passed. These are separate invocations, not an additive test total.", 32, 419, 778, 10.2)
pdf.para("All exactness statements are limited to the tested workload. The overlap diagnostics permit recorded unrelated GPU activity and cannot establish performance rankings.", 32, 486, 778, 9.7, GRAY)
pdf.source("B1; U1; A1; O1–O2; D3; T1–T3. Historical original-AR agreement is not claimed.")

pdf.start("The completed baseline experiment", "A declared tuning split, frozen winners, and repeated comparisons on matching test inputs.", "BATCH ONE / BF16 / NO PROFILER / DEDICATED GPU BLOCKS")
pdf.table(["Component", "Declared setup and meaning"], [
    ["Models and hardware", "Qwen3-8B target / Qwen3-0.6B drafter. Two 48 GB L40S GPUs over PCIe across NUMA sockets; no NVLink. AR and SD use GPU0 only; SSD uses both GPUs."],
    ["Balanced workload", "24 tuning prompts: 8 chat, 8 code, 8 document. Test split: 75 prompts, 25 per family. Three separate warmup prompts. Exclude the three historically investigated test prompts."],
    ["Frozen inputs", "Seed 0; exact saved input token IDs and chat template; complete inputs ≤2,048 tokens; model capacity 4,096. Models, tokenizer and execution fingerprints recorded."],
    ["Output protocols", "Primary: exactly 256 output tokens, ignoring EOS. Secondary: stop at natural EOS, cap 256. Secondary results reuse fixed-output-selected settings."],
    ["Execution and timing", "CUDA graphs and JIT miss fallback on; target and draft KV fractions 0.70. Three unscored warmups per fresh worker. Prefix caching off. Timer includes prefills and final drain."],
    ["Statistics", "Five rounds per method/protocol. Paired 95% confidence intervals: 10,000 seeded bootstrap samples over rounds and prompt IDs within families."],
], widths=[.21, .79], size=9.1, padding=6)
pdf.source("B1–B3. Batch size one means one user request at a time; K is draft depth, F is outcome-cache fan-out.")

pdf.chart("Tuning selected SD K=6 and SSD K=7/F=3", "Each setting: 24 balanced tuning prompts × two repetitions, fixed 256-token output.", tuning_plot, [
    ("Selection rule", "Lowest mean full-request tuning latency. Exact ties favor smaller K, then F. Winners were frozen before test measurements."),
    ("What the grid shows", "Larger K was not always better. K=10 increased SSD latency in this baseline grid; F=3 helped at K=4 and K=7."),
    ("Meaning for the project", "Our publication variants must beat a tuned strong SSD baseline. Improving a slow K=10 control alone is insufficient."),
], "B2. Green marks the winner within each method. AR contributes 48 additional tuning requests; total tuning requests = 480.")


def performance_page(protocol, plot):
    title = "Baseline results: fixed 256-token output" if protocol == "fixed" else "Baseline results: natural EOS, cap 256"
    subtitle = "Primary evaluation: identical output length; 75 prompts × five rounds per method." if protocol == "fixed" else "Secondary evaluation: the same fixed-output-selected configurations; no EOS retuning."
    pdf.start(title, subtitle)
    pdf.image(plot, 32, 148, 475, 233)
    ar, sd, ssd = [base(protocol, m) for m in ["ar", "sd", "ssd"]]
    pdf.blocks([
        ("Paired speedup vs shared AR", f"SD: {sd['comparison_to_ar']['speedup']:.2f}× (95% CI {sd['comparison_to_ar']['speedup_ci95'][0]:.2f}–{sd['comparison_to_ar']['speedup_ci95'][1]:.2f}).<br/>SSD: {ssd['comparison_to_ar']['speedup']:.2f}× ({ssd['comparison_to_ar']['speedup_ci95'][0]:.2f}–{ssd['comparison_to_ar']['speedup_ci95'][1]:.2f})."),
        ("Interpretation", "These are baseline-algorithm results against shared AR. SSD receives an extra GPU. They do not establish the contribution of our publication feature."),
    ], x=540, y=155, w=270, size=10.2)
    rows = []
    for label, row in zip(["Shared AR", "SD K6", "SSD K7/F3"], [ar, sd, ssd]):
        st = row["statistics"]
        rows.append([label, row["gpu_count"], f"{st['mean_latency_s']:.3f} / {st['p50_latency_s']:.3f} / {st['p95_latency_s']:.3f}", f"{st['committed_tok_per_s']:.2f}", f"{st['mean_first_delivery_s']*1000:.1f}", f"{st['mean_output_tokens']:.1f}"])
    pdf.table(["Method", "GPUs", "Mean / p50 / p95 (s)", "Tokens/s", "First delivery (ms)", "Output tokens"], rows, y=392, widths=[.18,.07,.28,.13,.17,.17], size=8.8, padding=6)
    pdf.para("p50 is the median request; p95 is the latency at or below which 95% of requests finish. First delivery is later with speculation even when the whole request finishes sooner.", 32, 504, 778, 8.6, GRAY)
    pdf.source("B1. Each row has 375 requests. Speculative outputs match shared AR; every method is repeatable on all 75 tested prompts.")


performance_page("fixed", fixed_plot)
performance_page("natural", natural_plot)

pdf.chart("The workload changes the size of the gain", "Fixed-output results, 25 prompts per family × five rounds for every method.", family_plot, [
    ("Measured SSD speedup", "Chat: 2.62×.<br/>Code: 4.36×.<br/>Document: 2.25×.<br/>All use two GPUs versus AR’s one."),
    ("What accompanies the difference", "SSD draft acceptance: code 62.9%, chat 36.3%, document 31.6%. Code also has more tokens committed per verification."),
    ("Meaning for evaluation", "Report every family. A balanced overall average can still hide a policy that helps predictable code but harms other tasks."),
], "B1. No family has a baseline mean-latency regression versus shared AR. Acceptance is descriptive, not a controlled causal ablation.")

pdf.chart("Uniform publication did not reduce latency", "A separate matched comparison: 75 prompts × five rounds × two SSD arms × two protocols = 1,500 requests.", publication_plot, [
    ("Fixed-output result", "Complete: 2.2006 s.<br/>Uniform: 2.2230 s.<br/>Uniform is 1.02% slower; paired 95% CI for slowdown: 0.71–1.32%."),
    ("Natural-EOS result", "Complete: 1.9045 s.<br/>Uniform: 1.9203 s.<br/>Uniform is 0.83% slower."),
    ("What this establishes", "At K=7, tuning chose P=7 among positive thresholds. No partial handoffs occurred. The experiment measured management overhead without an earlier prefix benefit."),
], "U1–U2. All 1,500 outputs match shared AR. CI uses paired rounds/prompts within families; all final draining is included.")
pdf.para("P is the minimum number of completed continuation tokens eligible for publication. The positive-threshold tuning grid was P=1,2,4,7; complete SSD (P=0) was faster than every positive setting.", 32, 477, 485, 9.0, GRAY)

pdf.chart("The latest handoff really overlaps queued work", "Normal correctness gate: overlap SSD at K=10, F=3, P=4, queue bound two.", overlap_plot, [
    ("Observed mechanism", "323 fixed-output + 310 natural-EOS responses completed before the submitted tail event: 633 actual handoffs."),
    ("Coverage", "Normal and stress phases: 22 valid blocks, 120 requests, 102/102 speculative outputs match shared AR. Rejection, 2,048-token inputs, EOS and final draining are exercised."),
    ("Limit", "K=7 produced no response-before-tail events. These 0.10-KV diagnostic runs are not dedicated repeated performance evidence."),
], "O1–O2; D1. Counts are internal handoffs across requests, not 633 independent prompts. Artificial-delay stress adds 100, shown separately in source data.", "CORRECTNESS AND MECHANISM EVIDENCE / NO SPEEDUP CLAIM")

pdf.start("What profiling and experiments taught us", "The historical traces motivated the implementation; current measurements determine its benefit.", "OBSERVATIONS, INTERPRETATION, AND DESIGN CONSEQUENCES")
pdf.table(["Observation", "Interpretation", "Consequence for the project"], [
    ["Original target decode and small verify phases lie left of the measured roofline ridge.", "At the measured widths, weight/data movement limits useful throughput; prefill can be compute-oriented.", "Do not describe all verification as compute-bound. Separate phases and profile the changed backend again."],
    ["Historical target wait: 9.2 ms on hits, 30.0 ms on misses; often five of seven depths were already ready.", "A ready prefix could reduce waiting. The old trace does not separate every wait cause.", "Measure transfer, completion, acceptance and draining separately; old values are not current-backend timings."],
    ["The selected K7/P7 uniform policy produced no partial handoffs and a small slowdown.", "Checkpoint and publication overhead can remain even when there is nothing useful to publish earlier.", "Keep complete-cache SSD as the strong control and tune publication parameters independently."],
    ["Bounded/adaptive diagnostics often exposed more queued depth after draining.", "Extra submitted work can postpone the handoff; adaptive decisions were not broadly exercised.", "Independent snapshots and handoff streams remove one dependency; broader policy coverage remains necessary."],
    ["Actual overlap occurs at K10; no latest dedicated latency result exists.", "A mechanism can work while total request time still increases.", "Include extra verification rounds, snapshots, memory and final drain in the comparison."],
], widths=[.35,.31,.34], size=9.0, padding=6)
pdf.source("Historical profiling: H1. Current publication: U1; A1; O1–O2; D4. No old timing is substituted into the new campaign.")

pdf.start("Remaining comparisons and ablations", "First establish whether the latest publication path beats the strongest matched complete SSD.", "DECLARED PLAN / NO FUTURE MEASUREMENTS ARE PLOTTED")
pdf.table(["Next comparison", "Purpose and trade-off", "Status"], [
    ["Complete vs uniform vs overlap", "Tune K=4/7/10 for complete; K=7/10 and P=2/4/7 for publication; overlap queue bound 1/2. Hold F=3.", "Prepared: 22 configurations, 1,056 tuning requests."],
    ["Frozen winners, five repeats", "AR + three SSD winners; 75 matching test prompts, fixed and natural protocols. Add complete K7 anchor if needed.", "Pending: 3,000–3,750 test requests."],
    ["Equal K/P, queue 1 vs 2", "Separate benefit from different draft depth; quantify excess queued work, target wait, and draining.", "Declared tuning cells support analysis; add matched repeats before a causal claim."],
    ["Mechanism and resource costs", "Measure acceptance, tokens/verification, cache hits, actual overlap, memory, snapshots and final drain, alongside request latency.", "Instrumentation exists; dedicated measurements pending."],
    ["Unequal depth / scheduling", "Spend deeper effort on useful likely outcomes, keep others shallow; compare against tuned uniform allocation.", "Not yet implemented. Main remaining proposal feature."],
], widths=[.23,.52,.25], size=9.1, padding=6)
pdf.para("<b>Execution:</b> one worker at a time; both GPUs dedicated; 0.70 KV fractions; warm full-prefill requests; no profiler. Contamination invalidates the whole block and scheduling waits. Never stop unrelated jobs.", 32, 465, 778, 9.7)
pdf.para("The 75 test prompts remain separate from configuration tuning, but were used in previous evaluations. They are not a newly unseen workload. New workloads and variable-length verification are further studies.", 32, 501, 778, 8.8, GRAY)
pdf.source("D2; L1. Expected dedicated execution time: roughly 4–6 hours, plus unpredictable GPU waiting; not a completion promise.")

pdf.start("A clear 10-minute Evaluation 2 story", "Presentation scope: implementation, initial results, analysis, and remaining plan; followed by five minutes of questions.", "TECHNICAL CLARITY / EXPLAIN THE RESULT AND ITS LIMITS")
pdf.table(["Time", "What to show", "Message"], [
    ["1 minute", "Problem and mid-submission comparison", "Can safely using ready draft work reduce full-request latency?"],
    ["2 minutes", "Implementation diagram and safety", "We built completed-prefix snapshots and overlapped handoff with queued work, while preserving safe reuse."],
    ["2 minutes", "Tuning and baseline graphs", "The comparison is reproducible; SD and SSD are strong baselines with explicit GPU allocations."],
    ["2 minutes", "Uniform result + actual overlap counts", "The first policy was slightly slower; the new mechanism overlaps, and its latency benefit remains open."],
    ["2 minutes", "Profiling lessons and remaining experiments", "Earlier handoff must outweigh extra rounds and coordination; unequal-depth scheduling is still to build."],
    ["1 minute", "Three precise claims", "Built: safe overlapping handoff. Learned: early availability alone is insufficient. Open: reliable latency improvement."],
], widths=[.12,.35,.53], size=9.4, padding=6)
pdf.para("<b>Do not claim:</b> that SSD’s 2.84× speedup is our publication contribution; that shared AR reproduces original AR; that the overlap correctness run establishes a speedup; or that the proposed ≥10% gain is already achieved.", 32, 479, 778, 10.2)
pdf.source("User-supplied Evaluation 2 rubric; B1; U1; O1–O2. Current evidence supports implementation and initial analysis.")

pdf.start("Evidence register and historical appendix", "The PDF is a dated evidence snapshot. Raw source paths and SHA-256 fingerprints accompany it.", "CURRENT UPDATE: PAGES 1–16 / HISTORICAL OCTOBER REPORT: PAGES 17–56")
pdf.table(["IDs", "Saved evidence", "Supports"], [
    ["B1–B3", "shared_uniform_v1_2026-11-17/baselines/{summary,selection,manifest}.json", "Completed baseline counts, tuning, statistics, exactness and workload."],
    ["U1–U2", "overnight_extensions_v2_2026-11-17/{heldout,depth_tune}/summary.json", "Matched uniform-publication result and positive-threshold tuning."],
    ["A1", "adaptive_publication_v1_2026-11-17/correctness/summary.json", "Bounded/adaptive diagnostic coverage; no headline speedup."],
    ["O1–O2", "prefix_overlap_v1_2026-11-17/{correctness,stress}/summary.json", "Exactness, actual overlap counts and performance-evidence flags."],
    ["D1–D4", "docs/{prefix_publication,publication_performance_campaign,shared_fast_backend,adaptive_publication}.md", "Implementation ownership, reference contract and remaining campaign."],
    ["T1–T3 / L1", "Saved CPU/GPU test logs and publication-performance status.json", "Test evidence and dated pending status; status is not a live promise."],
    ["H1", "22 October Anytime_SSD_Performance_Report.pdf", "Original mid-submission evidence, preserved as the following appendix."],
], widths=[.11,.57,.32], size=8.7, padding=5)
pdf.para("<b>Source-path prefix:</b> experiment folders above are under results/next_stage/. The October appendix retains its original measurements, captions and numbering. Every appendix page is labelled historical; its unfinished-run statements are superseded where this update supplies newer evidence.", 32, 479, 778, 9.5)
pdf.source("evidence_manifest.json records exact original paths, frozen snapshots, capture time and SHA-256 hashes.")
pdf.finish()
assert len(pdf.pages) == 16

# Add historical pages without rewriting their source PDF. Stamp the top margin
# and bottom margin so stale statements cannot be mistaken for current status.
update = fitz.open(OUT / "Evaluation2_Progress_Update.pdf")
combined = fitz.open()
combined.insert_pdf(update)
historical = fitz.open(OUT / manifest["sources"]["H1"]["snapshot"])
assert len(historical) == 40
combined.insert_pdf(historical)
for i in range(16, len(combined)):
    page = combined[i]
    page.draw_rect(fitz.Rect(0, 0, page.rect.width, 17), color=None, fill=(.94, .95, .97), overlay=True)
    page.insert_text((32, 11.8), "HISTORICAL APPENDIX | 22 OCT 2026 SNAPSHOT | CURRENT STATUS AND CLAIMS: PAGES 1-16", fontsize=7.1, color=(.61,.21,.25), overlay=True)
    page.insert_text((32, 589), f"Updated report page {i+1} / {len(combined)} | Original October page {i-15} / 40", fontsize=6, color=(.42,.46,.52), overlay=True)
combined.set_toc([[1, title, i+1] for i, title in enumerate(pdf.pages)] + [[1, "Historical appendix — 22 October 2026", 17]])
combined.set_metadata({
    "title": "Anytime SSD Performance Report — Evaluation 2 update, 17 November 2026",
    "author": "Ishaan Romil; Vishal Rao; Eshaan Sharma",
    "subject": "Completed results, implementation, Evaluation 2 mapping, and remaining studies",
})
combined.save(OUT / "Anytime_SSD_Performance_Report.pdf", garbage=4, deflate=True)
(OUT / "Report_Text.txt").write_text("\n\n".join(f"PAGE {i+1}\n{page.get_text()}" for i, page in enumerate(combined)))
(OUT / "Evaluation2_Update_Text.txt").write_text("\n\n".join(f"PAGE {i+1}\n{page.get_text()}" for i, page in enumerate(update)))

# Validate actual page text and drawing bounds, not only successful export.
expected = {
    2: ["Implementation", "30 marks"],
    3: ["22 October", "2,250"],
    5: ["1,500 / 1,500", "201 passed"],
    8: ["6.271", "3.161", "2.208"],
    9: ["5.419", "2.737", "1.902"],
    11: ["2.2006", "2.2230", "1.02%"],
    12: ["633", "102/102"],
    14: ["1,056", "3,000", "Not yet implemented"],
    15: ["10-minute", "≥10%"],
}
for number, needles in expected.items():
    text = update[number-1].get_text()
    for needle in needles:
        assert needle in text, (number, needle)
for i, page in enumerate(combined):
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                assert x0 >= -.5 and y0 >= -.5 and x1 <= page.rect.width+.5 and y1 <= page.rect.height+.5, (i+1, span)
    if i >= 16:
        assert "HISTORICAL APPENDIX" in page.get_text()

# Contact sheet and selected full-size previews for visual inspection.
from PIL import Image, ImageDraw
thumb_w, thumb_h = 421, 298
sheet = Image.new("RGB", (4*thumb_w, 4*(thumb_h+22)), "#DEE5EC")
draw = ImageDraw.Draw(sheet)
for i, page in enumerate(update):
    pix = page.get_pixmap(matrix=fitz.Matrix(.5, .5), alpha=False)
    im = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    x, y = (i%4)*thumb_w, (i//4)*(thumb_h+22)
    sheet.paste(im, (x, y))
    draw.text((x+12, y+thumb_h+4), f"Page {i+1}: {pdf.pages[i][:42]}", fill="#152E49")
sheet.save(OUT / "preview_update.jpg", quality=92)
for number in (2, 4, 8, 11, 12, 14):
    pix = update[number-1].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    pix.save(OUT / f"preview_page_{number:02d}.png")
validation = {
    "update_pages": len(update), "historical_pages": len(historical), "full_report_pages": len(combined),
    "page_text_checks": "passed", "page_bounds": "passed", "historical_labels": "passed",
    "source_hashes": "passed", "graphs": [p.name for p in sorted(FIG.glob("*.png"))],
    "evidence_captured_at_ist": manifest["captured_at_ist"],
    "notes": ["Original October files unchanged", "No experimental campaign or execution source modified", "Overlap diagnostics are not performance rankings", "Exactness is scoped to shared AR"],
}
(OUT / "artifact_validation.json").write_text(json.dumps(validation, indent=2)+"\n")
print(json.dumps(validation, indent=2))
