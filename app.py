import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# API KEY — reads from Railway env var or
# Streamlit secrets (works on both platforms)
# ─────────────────────────────────────────────
def get_api_key():
    # 1. Railway / any host: set ANTHROPIC_API_KEY in environment variables
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key
    # 2. Streamlit Cloud: set it in the Secrets manager
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return ""

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sorcerer System | Creativity Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .big-font {font-size: 50px !important; font-weight: 800; color: #4da6ff;}
    .win-arrow {color: lime; font-weight: bold;}
    .lose-arrow {color: red; font-weight: bold;}
    .comp-table { width: 100%; text-align: left; border-collapse: collapse; margin-top: 20px;}
    .comp-table th { padding: 12px; border-bottom: 2px solid #4da6ff; font-size: 18px; background-color: #111111; position: sticky; top: 0;}
    .comp-table td { padding: 12px; border-bottom: 1px solid #333; font-size: 16px; }
    .comp-table tr:hover { background-color: #1a1a1a; }
    .score-excellent { background: linear-gradient(90deg, #1a3a1a, #0d0d0d); border-left: 5px solid #00e676; padding: 14px 20px; border-radius: 8px; display: inline-block; min-width: 200px;}
    .score-good      { background: linear-gradient(90deg, #1a2e3a, #0d0d0d); border-left: 5px solid #4da6ff; padding: 14px 20px; border-radius: 8px; display: inline-block; min-width: 200px;}
    .score-average   { background: linear-gradient(90deg, #2e2a0d, #0d0d0d); border-left: 5px solid #ffc107; padding: 14px 20px; border-radius: 8px; display: inline-block; min-width: 200px;}
    .score-poor      { background: linear-gradient(90deg, #3a1a1a, #0d0d0d); border-left: 5px solid #ff5252; padding: 14px 20px; border-radius: 8px; display: inline-block; min-width: 200px;}
    .score-label { font-size: 13px; color: #aaa; margin-bottom: 4px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase;}
    .score-value { font-size: 32px; font-weight: 900; }
    .score-tag   { font-size: 12px; margin-top: 4px; font-weight: 700; letter-spacing: 1px; }
    .fact-box { background: #111827; border: 1px solid #1f2937; border-radius: 10px; padding: 16px 20px; margin-bottom: 10px; }
    .fact-icon { font-size: 20px; margin-right: 8px; }
    .fact-text { font-size: 15px; color: #d1d5db; line-height: 1.5; }
    .fact-badge { display: inline-block; background: #1e3a5f; color: #60a5fa; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-bottom: 6px; letter-spacing: 0.5px; text-transform: uppercase; }
    .summary-box { background: #0d1117; border: 1.5px solid #4da6ff; border-radius: 14px; padding: 24px 28px; margin-top: 20px; }
    .summary-box h3 { color: #4da6ff; font-size: 18px; margin-bottom: 16px; }
    .summary-box p { color: #c9d1d9; font-size: 15px; line-height: 1.8; margin-bottom: 12px; }
    .summary-verdict { background: #161b22; border-left: 4px solid #00e676; border-radius: 6px; padding: 14px 18px; margin-top: 16px; color: #e6edf3; font-size: 15px; line-height: 1.7; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

FALLBACK_IMAGE = "https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_1280.png"

# ─────────────────────────────────────────────
# TIER LOGIC
# ─────────────────────────────────────────────
def score_tier(score):
    if score >= 2.75:
        return "Excellent", "#00e676", "score-excellent"
    elif score >= 2.00:
        return "Good", "#4da6ff", "score-good"
    elif score >= 1.21:
        return "Average", "#ffc107", "score-average"
    else:
        return "Poor", "#ff5252", "score-poor"

# ─────────────────────────────────────────────
# AI COMPARISON SUMMARY
# ─────────────────────────────────────────────
def generate_comparison_summary_ai(p1_data, p2_data, all_stats):
    """Call Claude API — only used when ANTHROPIC_API_KEY is available."""
    import requests

    api_key = get_api_key()
    if not api_key:
        return None

    stat_lines = []
    for stat, v1, v2 in all_stats:
        w = p1_data["Player_Name"] if v1 > v2 else (p2_data["Player_Name"] if v2 > v1 else "Draw")
        stat_lines.append(f"  {stat}: {p1_data['Player_Name']}={v1:.2f} vs {p2_data['Player_Name']}={v2:.2f} → {w} wins")

    prompt = f"""You are an elite football tactical analyst. Write a detailed tactical summary of this head-to-head comparison.

Players:
- {p1_data['Player_Name']} ({p1_data['Team']}, {p1_data['Position']}, Role: {p1_data['Role_Tag']}, Sorcerer Score: {p1_data['Sorcerer_Score']})
- {p2_data['Player_Name']} ({p2_data['Team']}, {p2_data['Position']}, Role: {p2_data['Role_Tag']}, Sorcerer Score: {p2_data['Sorcerer_Score']})

Full stat comparison (all per 90 unless stated):
{chr(10).join(stat_lines)}

Write your analysis in exactly this structure:
1. A paragraph titled "Attacking Output" analysing creative and chance-creation differences.
2. A paragraph titled "Passing Profile" covering passing styles, efficiency, progressive passing, final third entries.
3. A paragraph titled "Ball Carrying" comparing carries and how each player advances play.
4. A paragraph titled "Defensive Contribution" assessing tackles, interceptions, blocks.
5. A final "Verdict" paragraph — direct, opinionated, names a winner and explains why.

Be direct, use specific numbers, write like a top analyst. Do not hedge. Each paragraph 3-5 sentences."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1200,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30,
        )
        data = response.json()
        if "content" in data and data["content"]:
            return data["content"][0]["text"]
        return None
    except Exception:
        return None


def _cmp(v1, v2, n1, n2, metric, high_label, low_label, unit=""):
    diff = abs(v1 - v2)
    if diff < 0.05:
        return f"Both players are virtually identical in {metric} ({v1:.2f}{unit} vs {v2:.2f}{unit}), making this a neutral battleground."
    winner, loser, wv, lv = (n1, n2, v1, v2) if v1 > v2 else (n2, n1, v2, v1)
    gap = "marginally" if diff < 0.3 else ("clearly" if diff < 1.0 else "significantly")
    return f"{winner} {gap} leads in {metric} ({wv:.2f}{unit} vs {lv:.2f}{unit}), confirming {winner} as the {high_label} and {loser} as the {low_label} in this duel."


def generate_comparison_summary_local(p1_data, p2_data, all_stats):
    """Rule-based fallback — works with zero network access."""
    n1 = p1_data["Player_Name"]
    n2 = p2_data["Player_Name"]

    def g(key):
        for stat, v1, v2 in all_stats:
            if key.lower() in stat.lower():
                return v1, v2
        return 0.0, 0.0

    def w(v1, v2):
        if v1 > v2 + 0.05: return n1
        if v2 > v1 + 0.05: return n2
        return "neither"

    sca1,  sca2  = g("SCA")
    ppa1,  ppa2  = g("PPA")
    xa1,   xa2   = g("xA")
    tb1,   tb2   = g("Through Ball")
    pp1,   pp2   = g("Prog Pass")
    ft1,   ft2   = g("Final Third Pass")
    pc1,   pc2   = g("Pass Cmp")
    er1,   er2   = g("Efficiency")
    lpc1,  lpc2  = g("Long Pass Cmp")
    pc1_,  pc2_  = g("Prog Carr")
    cpb1,  cpb2  = g("Carries Pen")
    tkl1,  tkl2  = g("Tackle")
    int1,  int2  = g("Intercept")
    ss1 = float(p1_data.get("Sorcerer_Score", 0))
    ss2 = float(p2_data.get("Sorcerer_Score", 0))

    s1 = [
        _cmp(sca1, sca2, n1, n2, "Shot-Creating Actions", "more prolific creator", "quieter threat"),
        _cmp(ppa1, ppa2, n1, n2, "passes into the penalty area", "more dangerous final-ball player", "safer operator"),
        _cmp(xa1,  xa2,  n1, n2, "expected assists", "higher-quality chance creator", "lower-threat passer"),
    ]
    if abs(tb1 - tb2) > 0.05:
        tbw = n1 if tb1 > tb2 else n2
        s1.append(f"{tbw} is the more willing line-breaker with {max(tb1,tb2):.2f} through balls per 90 vs {min(tb1,tb2):.2f}.")

    s2 = [
        _cmp(pp1, pp2, n1, n2, "progressive passes per 90", "more forward-thinking distributor", "more conservative passer"),
        _cmp(ft1, ft2, n1, n2, "final-third passes", "more penetrative in the last third", "less incisive in transition zones"),
    ]
    if abs(pc1 - pc2) > 2:
        safer = n1 if pc1 > pc2 else n2
        riskier = n2 if pc1 > pc2 else n1
        s2.append(f"{safer} plays it safer with a {max(pc1,pc2):.1f}% pass completion vs {riskier}'s {min(pc1,pc2):.1f}% — reflecting different risk appetites.")
    if abs(er1 - er2) > 0.01:
        s2.append(_cmp(er1, er2, n1, n2, "passing efficiency ratio", "more economical line-breaker", "higher-volume but less efficient distributor"))

    s3 = [
        _cmp(pc1_, pc2_, n1, n2, "progressive carries", "more willing to drive at defences", "more pass-first in nature"),
        _cmp(cpb1, cpb2, n1, n2, "carries into the penalty area", "more box-penetrating", "less direct in their runs"),
    ]
    carry_total1 = pc1_ + cpb1
    carry_total2 = pc2_ + cpb2
    if abs(carry_total1 - carry_total2) > 0.5:
        carrier = n1 if carry_total1 > carry_total2 else n2
        passer  = n2 if carry_total1 > carry_total2 else n1
        s3.append(f"Overall {carrier} is the more carry-dependent player — advancing play physically rather than purely through distribution, which is {passer}'s preference.")

    s4 = [
        _cmp(tkl1, tkl2, n1, n2, "tackles per 90", "more active presser", "more passive out of possession"),
        _cmp(int1, int2, n1, n2, "interceptions", "sharper reader of the game defensively", "less active in cutting lanes"),
    ]
    def_total1 = tkl1 + int1
    def_total2 = tkl2 + int2
    if abs(def_total1 - def_total2) > 0.3:
        harder = n1 if def_total1 > def_total2 else n2
        softer = n2 if def_total1 > def_total2 else n1
        s4.append(f"Combined, {harder} contributes {max(def_total1,def_total2):.2f} defensive actions vs {softer}'s {min(def_total1,def_total2):.2f} — a meaningful gap in defensive engagement.")

    edges = {n1: 0, n2: 0}
    for e in [w(sca1+ppa1+xa1, sca2+ppa2+xa2), w(pc1_+cpb1, pc2_+cpb2), w(tkl1+int1, tkl2+int2), w(pp1+ft1, pp2+ft2)]:
        if e in edges:
            edges[e] += 1

    dominant = n1 if edges[n1] > edges[n2] else (n2 if edges[n2] > edges[n1] else None)
    if dominant:
        other = n2 if dominant == n1 else n1
        dom_ss = ss1 if dominant == n1 else ss2
        oth_ss = ss2 if dominant == n1 else ss1
        verdict = (
            f"{dominant} wins this head-to-head across {edges[dominant]} of 4 dimensions, backed by a Sorcerer Score of {dom_ss} vs {other}'s {oth_ss}. "
            f"The data paints {dominant} as the more complete contributor in this matchup. "
            f"{other} is not without merit — their profile suits a specific tactical context — but the numbers consistently point in one direction. "
            f"If choosing one player for a system demanding creative output and progressive impact, {dominant} is the clear pick."
        )
    else:
        verdict = (
            f"This is a genuinely balanced head-to-head — {n1} and {n2} split the four dimensions evenly with Sorcerer Scores of {ss1} and {ss2} respectively. "
            f"Neither player dominates comprehensively, and context becomes the deciding factor. "
            f"The right choice between them is entirely system-dependent — the data refuses to hand either a clear verdict."
        )

    return [
        ("Attacking Output",       " ".join(s1)),
        ("Passing Profile",        " ".join(s2)),
        ("Ball Carrying",          " ".join(s3)),
        ("Defensive Contribution", " ".join(s4)),
        ("Verdict",                verdict),
    ]


def generate_comparison_summary(p1_data, p2_data, all_stats):
    """Try AI first; fall back to local rule-based generator if no API key or call fails."""
    ai_result = generate_comparison_summary_ai(p1_data, p2_data, all_stats)
    if ai_result:
        return ("ai", ai_result)
    return ("local", generate_comparison_summary_local(p1_data, p2_data, all_stats))


def parse_summary_sections(text):
    """Parse the AI summary into labelled sections."""
    if not text:
        return []

    sections = []
    current_title = None
    current_body = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Detect section headers — numbered or bold-style
        for heading in ["Attacking Output", "Passing Profile", "Ball Carrying", "Defensive Contribution", "Verdict"]:
            if heading.lower() in line.lower() and len(line) < 80:
                if current_title and current_body:
                    sections.append((current_title, " ".join(current_body)))
                current_title = heading
                # Strip the heading from the line if rest of content follows it
                remainder = line.lower().replace(heading.lower(), "").strip(" :-–.1234567890*#")
                current_body = [remainder] if remainder else []
                break
        else:
            if current_title:
                current_body.append(line)

    if current_title and current_body:
        sections.append((current_title, " ".join(current_body)))

    return sections


# ─────────────────────────────────────────────
# PLAYER INSIGHTS (hover tooltip)
# ─────────────────────────────────────────────
PLAYER_INSIGHTS = {
    "Trent Alexander-Arnold": "Elite progressive passer from deep zones. #1 in prog passes & crosses among all 25 players.",
    "João Cancelo":           "Two-way fullback with top-3 tackles & interceptions. Passing efficiency lags behind his reputation.",
    "Andrew Robertson":       "Most minutes of any fullback. Consistent workhorse — leads nothing, drops nothing.",
    "Achraf Hakimi":          "More defensively active than his reputation suggests. Carries are his weapon, not passes.",
    "Kyle Walker":            "Lowest Sorcerer Score in the database. Elite athlete, minimal creator.",
    "Reece James":            "Elite numbers when fit. Injury keeps him off this leaderboard — quality over quantity.",
    "Theo Hernandez":         "Carries forward rather than threading passes. PPA ranks 23rd of 25.",
    "Alphonso Davies":        "#2 in progressive carries. Speed merchant — physical creativity, not technical.",
    "Kieran Trippier":        "#1 passing efficiency ratio. Most economical creator — zero carry game.",
    "Dani Carvajal":          "Top-3 in tackles & interceptions. Defensively elite, creatively conservative.",
    "Ben White":              "Inverted fullback by design. 24th in SCA — positional influence over creative output.",
    "Alejandro Grimaldo":     "Best left back Sorcerer Score outside Dimarco. Short, sharp passing game.",
    "Federico Dimarco":       "Best left back in the database by Sorcerer Score. Crossing specialist in Inzaghi's system.",
    "Jeremie Frimpong":       "#2 carries into box. Pure wingback — runs not passes define his profile.",
    "Jordi Alba":             "Most minutes in the dataset. Low efficiency ratio — volume over line-breaking.",
    "Oleksandr Zinchenko":    "#2 final third passes. Operates as a central 8 in Arteta's inverted system.",
    "Luke Shaw":              "Below-peak numbers — injuries have disrupted what could have been an elite profile.",
    "Bruno Fernandes":        "#2 overall Sorcerer Score among midfielders. Elite across every creativity metric.",
    "Florian Wirtz":          "Elite numbers on limited minutes. Top-3 SCA, PPA, xA — huge ceiling ahead.",
    "Jude Bellingham":        "Top-3 long pass completion. Goals define his impact more than passing.",
    "Jamal Musiala":          "#1 prog carries AND carries into box. Most dangerous carrier in the entire dataset.",
    "Pedri":                  "Highest Sorcerer Score of any deep midfielder. Pass master — quality over volume.",
    "Martin Ødegaard":        "#3 overall Sorcerer Score. Most complete advanced playmaker profile in the database.",
    "Kevin De Bruyne":        "Highest Sorcerer Score (3.99). Leads SCA, PPA, through balls, and xA. The benchmark.",
    "Vitinha":                "#1 pass completion % (91%) AND long pass completion % (82%). Safe by design — low Sorcerer Score.",
}

# ─────────────────────────────────────────────
# PLAYER FACTS
# ─────────────────────────────────────────────
PLAYER_FACTS = {
    "Trent Alexander-Arnold": [
        ("🎯", "ELITE DISTRIBUTOR", "Leads all 25 players in progressive passes per 90 (8.83) — no one in this dataset moves the ball forward more."),
        ("🌐", "CROSSING KING", "#1 in crosses attempted per 90 (7.5), more than double the average fullback in this database."),
        ("⚠️", "RISK TAKER", "Ranks 24th of 25 in pass completion % (76.5%) — his willingness to attempt difficult line-breaking passes is exactly why the Sorcerer Score rewards him."),
        ("📈", "FULLBACK OUTLIER", "His Sorcerer Score of 3.14 is higher than every other fullback by a margin of 0.67. The next best fullback (Trippier) scores 2.47."),
    ],
    "João Cancelo": [
        ("🛡️", "TWO-WAY THREAT", "Ranks #2 of 25 in both tackles and interceptions per 90 — the most defensively active attacking fullback in the dataset."),
        ("⚠️", "EFFICIENCY GAP", "Despite high defensive output, his passing efficiency ratio (0.086) ranks 22nd — his passes rarely break lines."),
        ("📉", "POST-PEAK NUMBERS", "Still competitive but his Sorcerer Score (1.56) reflects the drop from his peak Guardiola era at City."),
    ],
    "Andrew Robertson": [
        ("💪", "WORKHORSE", "Highest minutes played (26,000) of any fullback in the dataset — pure reliability over flash."),
        ("📉", "LONG BALL WEAKNESS", "Ranks 19th of 25 in long pass completion % (58%) — his directness comes at the cost of long-range accuracy."),
        ("⚖️", "CONSISTENT NOT EXPLOSIVE", "Solid across all tabs but leads nothing — the definition of reliable rather than elite."),
    ],
    "Achraf Hakimi": [
        ("🏎️", "DEFENSIVE SURPRISE", "Ranks 3rd in tackles per 90 (2.1) — more defensively engaged than his attacking reputation suggests."),
        ("📉", "CREATIVE CEILING", "Key passes per 90 (1.4) ranks 19th — his impact comes from runs and positioning, not passing creativity."),
    ],
    "Kyle Walker": [
        ("🏆", "PASS SAFE", "Ranks #2 of 25 in pass completion % (89%) — almost nothing he plays goes astray."),
        ("📉", "CREATIVITY FLOOR", "Ranks dead last (25th) in SCA per 90 (1.8). The data confirms he's a positional fullback, not a creative one."),
        ("📊", "LOWEST SCORE", "Sorcerer Score of 0.50 is the lowest in the entire database — the formula correctly identifies him as a destroyer, not a creator."),
    ],
    "Reece James": [
        ("🩹", "INJURY SHADOW", "Only 11,000 minutes played — the lowest among fullbacks. His 1.72 Sorcerer Score is built on a small sample."),
        ("🎯", "QUALITY WHEN FIT", "SCA (3.4) and PPA (1.7) are elite for a fullback when he plays — injury is the only thing keeping him off this leaderboard."),
    ],
    "Theo Hernandez": [
        ("🏃", "CARRIER NOT PASSER", "Progressive carries (5.2) rank high, but PPA per 90 (1.2) ranks 23rd — he drives forward rather than threading passes."),
    ],
    "Alphonso Davies": [
        ("⚡", "SPEED MERCHANT", "Ranks #2 in progressive carries per 90 (6.5) — beats defenders with pace and dribbling, not passing."),
        ("📉", "PASSING LIMITATION", "Ranks 23rd in progressive passes per 90 (4.6) — his creativity is physical, not technical."),
    ],
    "Kieran Trippier": [
        ("📐", "SET PIECE MASTER", "Leads all 25 players in passing efficiency ratio (0.133) — every pass counts. Also tops the charts in tackles (2.4)."),
        ("⚠️", "NO CARRY GAME", "Ranks dead last (25th) in progressive carries (1.8) — plays it safe and quick rather than driving forward."),
        ("🏆", "MOST ECONOMICAL CREATOR", "The highest efficiency ratio means he wastes virtually no passes — the most economical creator in the group."),
    ],
    "Dani Carvajal": [
        ("🛡️", "DEFENSIVE ANCHOR", "Ranks 3rd in both tackles and interceptions — the most defensively balanced fullback alongside Cancelo."),
        ("📉", "CREATIVE DECLINE", "SCA of 2.4 ranks 23rd — conservative late-career role at Real Madrid has reduced his attacking output."),
    ],
    "Ben White": [
        ("🤫", "QUIET OPERATOR", "Ranks 24th in SCA (1.9) — influence is entirely positional and defensive in Arteta's inverted fullback role."),
    ],
    "Alejandro Grimaldo": [
        ("🎯", "LEFT BACK LEADER", "1.79 Sorcerer Score is the highest of any left back in the dataset outside Dimarco — key to Xabi Alonso's system."),
        ("📉", "SHORT GAME ONLY", "Long pass completion % (59%) ranks 18th — his game is short and sharp, not expansive."),
    ],
    "Federico Dimarco": [
        ("🌐", "CROSSING SPECIALIST", "Ranks 3rd in crosses attempted per 90 (6.5) — a natural wide deliverer in Inzaghi's system."),
        ("🏆", "BEST LEFT BACK SCORE", "Sorcerer Score of 2.24 is the highest among all left backs in the database."),
        ("⚠️", "COMPLETION COST", "Ranks 22nd in pass completion % (80%) — ambitious delivery means more misses."),
    ],
    "Jeremie Frimpong": [
        ("🏎️", "BOX ARRIVAL", "Ranks #2 in carries into the penalty area per 90 (2.1) — arrives late and dangerously in the box."),
        ("📉", "PASS VOLUME FLOOR", "Ranks 25th in progressive passes per 90 (3.8) — contribution is entirely through carrying, not distribution."),
    ],
    "Jordi Alba": [
        ("📜", "VETERAN VOLUME", "35,000 minutes — the most in the dataset. Still producing in MLS at an advanced age."),
        ("📉", "EFFICIENCY LOWEST", "Passing efficiency ratio (0.083) is the 2nd lowest — high total passes but very few are line-breaking."),
    ],
    "Oleksandr Zinchenko": [
        ("🧠", "INVERTED INTELLIGENCE", "Ranks #2 in final third passes per 90 (6.8) — operates as a de facto 8 inside the pitch in Arteta's system."),
        ("📉", "FINAL PRODUCT GAP", "Despite high ball involvement, key passes (1.2) rank 22nd — positioning matters more than final product."),
    ],
    "Luke Shaw": [
        ("📉", "BELOW PEAK", "SCA ranks 21st — a player whose best numbers came before persistent injuries disrupted his United career."),
    ],
    "Bruno Fernandes": [
        ("🎯", "ASSIST MACHINE", "Ranks #2 in SCA (6.12), #2 in PPA (3.1), and #2 in through balls (0.8) — consistently elite across every creativity metric."),
        ("📉", "LONG BALL GAMBLER", "Ranks 25th in long pass completion % (52%) — attempts the most ambitious passes and misses plenty."),
        ("🏆", "MIDFIELDER BENCHMARK", "Sorcerer Score of 3.91 is the 2nd highest overall — only De Bruyne scores higher."),
    ],
    "Florian Wirtz": [
        ("⚡", "YOUNG GUN", "Ranks 3rd in SCA (5.6), 3rd in PPA (2.9), and 3rd in xA (0.35) — elite numbers on just 11,000 minutes."),
        ("📈", "CEILING ALERT", "His numbers on limited minutes suggest his Sorcerer Score could rise significantly with more game time."),
        ("📉", "ZERO DEFENSIVE WORK", "Ranks 25th in tackles (1.1) — a pure creator with no defensive responsibility."),
    ],
    "Jude Bellingham": [
        ("🎯", "ELITE LONG PASSER", "Ranks 3rd in long pass completion % (70%) — technically excellent at switching play."),
        ("🌟", "BOX-TO-BOX REALITY", "Sorcerer Score (2.10) confirms he's a premium midfielder, but his goals define his impact more than his passing."),
    ],
    "Jamal Musiala": [
        ("🏎️", "DRIBBLE KING", "Leads all 25 players in both progressive carries (6.8) AND carries into the penalty area (2.4) — the most dangerous carrier in the dataset."),
        ("🎯", "GCA EXCELLENCE", "Ranks 3rd in goal creating actions (0.6) — his carries directly manufacture chances."),
        ("📉", "WIDE PLAY VOID", "Ranks 25th in crosses (1.0) — entirely a central, interior attacker."),
    ],
    "Pedri": [
        ("🧠", "PASS MASTER", "Ranks 3rd in progressive passes (7.5), 2nd in final third passes (6.8), and 2nd in long pass completion % (75%)."),
        ("🌟", "DEEP ROLE OUTLIER", "Sorcerer Score of 2.52 is the highest of any deep/central midfielder — proof the formula rewards quality over volume."),
        ("📉", "WIDE PLAY VOID", "Ranks 23rd in crosses (1.2) — entirely a central orchestrator."),
    ],
    "Martin Ødegaard": [
        ("🎯", "THROUGH BALL ELITE", "Ranks 3rd in through balls per 90 (0.75) — one of the sharpest final passes in the dataset."),
        ("🏆", "TOP 3 OVERALL", "Sorcerer Score of 3.27 ranks 3rd overall — the most complete advanced playmaker profile in the database."),
        ("📉", "DEFENSIVE GHOST", "Ranks 20th in interceptions (0.8) — operates entirely in the final third."),
    ],
    "Kevin De Bruyne": [
        ("👑", "STAT LEADER", "Leads or co-leads in SCA (6.2), PPA (3.5), through balls (0.95), and xA (0.45) — the most complete creative profile in the dataset."),
        ("🏆", "HIGHEST SORCERER SCORE", "3.99 — the single highest score in the database. No player generates more attacking threat per pass."),
        ("📉", "DEFENSIVE FLOOR", "Ranks 25th in interceptions (0.4) — the price of being a pure creator."),
    ],
    "Vitinha": [
        ("🎯", "PASSING PERFECTION", "Ranks #1 in both long pass completion % (82%) and overall pass completion % (91%) — the most accurate passer in the database."),
        ("📉", "CREATIVITY LIMIT", "Despite precision, his low Sorcerer Score (1.52) reveals the contradiction: elite accuracy, low risk, low line-breaking output."),
        ("🧩", "DEEP ARCHITECT", "His role as PSG's deepest midfielder explains everything — positional, safe, and non-creative by design."),
    ],
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def get_image_url(url):
    if pd.isna(url) or str(url).strip() == "" or not str(url).strip().startswith("http"):
        return FALLBACK_IMAGE
    raw = str(url).strip()
    if "wikimedia" in raw and not any(raw.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
        return FALLBACK_IMAGE
    return raw

def safe_index(lst, name, default=0):
    try:
        return lst.index(name)
    except ValueError:
        return default

def compute_sorcerer_score(row):
    threat = row["SCA_p90"] + 2 * row["PPA_p90"] + row["xA_p90"]
    efficiency = (row["Prog_Passes_p90"] + 1.5 * row["Final_Third_Passes_p90"]) / max(row["Total_Passes_p90"], 1)
    return round(threat * efficiency, 2)

def compute_creativity_index(row):
    return round((row["Key_Passes_p90"] + row["xA_p90"] * 2 + row["Through_Balls_p90"] * 1.5) * row["Passing_Efficiency_Ratio"], 2)


# ─────────────────────────────────────────────
# D3 SCATTER WITH REMOVE/RESTORE
# ─────────────────────────────────────────────
def render_d3_scatter(players_data, x_col, y_col, x_label, y_label):
    players_json = json.dumps(players_data)
    html = f"""<!DOCTYPE html>
<html>
<head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: transparent; font-family: 'Segoe UI', sans-serif; overflow: hidden; }}
  #controls {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; background: #0e1117;
    border-bottom: 1px solid #222; flex-wrap: wrap;
  }}
  #controls span {{ color: #aaa; font-size: 13px; font-weight: 600; }}
  .toggle-btn {{
    padding: 6px 18px; border-radius: 20px;
    border: 1.5px solid #4da6ff; background: transparent;
    color: #4da6ff; font-size: 13px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
  }}
  .toggle-btn.active {{ background: #4da6ff; color: #000; }}
  #removed-panel {{
    display: none; align-items: center; gap: 8px;
    padding: 8px 16px; background: #161b22;
    border-bottom: 1px solid #222; flex-wrap: wrap;
  }}
  #removed-panel.visible {{ display: flex; }}
  #removed-panel span {{ color: #888; font-size: 12px; font-weight: 600; }}
  .restore-chip {{
    display: inline-flex; align-items: center; gap: 5px;
    background: #1e3a1e; border: 1px solid #238636;
    border-radius: 20px; padding: 3px 10px 3px 8px;
    color: #3fb950; font-size: 12px; cursor: pointer;
    transition: background 0.15s;
  }}
  .restore-chip:hover {{ background: #2ea04326; }}
  .restore-chip .plus {{ font-size: 14px; font-weight: 700; line-height: 1; }}
  .axis path, .axis line {{ stroke: #2a2a2a; }}
  .axis text {{ fill: #666; font-size: 11px; }}
  .grid line {{ stroke: #1a1a1a; stroke-dasharray: 3,3; }}
  .grid path {{ stroke: none; }}
  .axis-label {{ fill: #888; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; }}
  .remove-btn {{
    cursor: pointer; opacity: 0;
    transition: opacity 0.15s;
    pointer-events: none;
  }}
  .node:hover .remove-btn {{ opacity: 1; pointer-events: all; }}
  #tooltip {{
    position: fixed; pointer-events: none;
    background: #0d1117; border: 1.5px solid #4da6ff;
    border-radius: 14px; opacity: 0;
    transition: opacity 0.18s; z-index: 9999;
    width: 260px; box-shadow: 0 12px 40px rgba(0,0,0,0.7);
    overflow: hidden;
  }}
  #tooltip.visible {{ opacity: 1; }}
  #tt-header {{
    display: flex; align-items: center; gap: 12px;
    padding: 14px; background: #161b22;
    border-bottom: 1px solid #21262d;
  }}
  #tt-img {{
    width: 54px; height: 54px; border-radius: 50%;
    object-fit: cover; border: 2px solid #4da6ff;
    flex-shrink: 0; background: #1a1a1a;
  }}
  #tt-name {{ color: #e6edf3; font-size: 14px; font-weight: 700; line-height: 1.3; }}
  #tt-meta {{ color: #8b949e; font-size: 11px; margin-top: 3px; }}
  #tt-body {{ padding: 12px 14px; }}
  #tt-stats {{
    display: flex; justify-content: space-between;
    margin-bottom: 10px; padding-bottom: 10px;
    border-bottom: 1px solid #21262d;
  }}
  .tt-stat {{ text-align: center; flex: 1; }}
  .tt-stat-val {{ color: #4da6ff; font-size: 17px; font-weight: 800; }}
  .tt-stat-lbl {{ color: #8b949e; font-size: 9px; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.5px; }}
  #tt-insight {{ color: #c9d1d9; font-size: 12px; line-height: 1.55; font-style: italic; }}
  #tt-tier {{
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 8px; border-radius: 10px; margin-bottom: 8px;
    letter-spacing: 0.5px; text-transform: uppercase;
  }}
  #remove-hint {{
    padding: 4px 16px 6px; background: #0e1117;
    color: #555; font-size: 11px; font-style: italic;
  }}
</style>
</head>
<body>
<div id="controls">
  <span>Marker Style:</span>
  <button class="toggle-btn active" id="btn-icons" onclick="setMode('icons')">👤 Face Icons</button>
  <button class="toggle-btn" id="btn-circles" onclick="setMode('circles')">⬤ Circles</button>
</div>
<div id="removed-panel">
  <span>Removed:</span>
  <div id="removed-chips"></div>
</div>
<div id="remove-hint">Hover over a player icon → click ✕ to remove from chart</div>
<svg id="chart"></svg>

<div id="tooltip">
  <div id="tt-header">
    <img id="tt-img" src="" onerror="this.src='{FALLBACK_IMAGE}'"/>
    <div>
      <div id="tt-name"></div>
      <div id="tt-meta"></div>
    </div>
  </div>
  <div id="tt-body">
    <div id="tt-stats">
      <div class="tt-stat">
        <div class="tt-stat-val" id="tt-x"></div>
        <div class="tt-stat-lbl">{x_label}</div>
      </div>
      <div class="tt-stat">
        <div class="tt-stat-val" id="tt-y"></div>
        <div class="tt-stat-lbl">{y_label}</div>
      </div>
      <div class="tt-stat">
        <div class="tt-stat-val" id="tt-ss"></div>
        <div class="tt-stat-lbl">Sorcerer Score</div>
      </div>
    </div>
    <div id="tt-tier"></div>
    <div id="tt-insight"></div>
  </div>
</div>

<script>
const allPlayers = {players_json};
const xKey = "{x_col}";
const yKey = "{y_col}";
let mode = "icons";
let removedSet = new Set();

const ROLE_COLORS = {{
  "Wingback Creator":    "#4da6ff",
  "Possession Full-back":"#a78bfa",
  "Balanced Full-back":  "#34d399",
  "Advanced Playmaker":  "#fb923c",
  "Deep Playmaker":      "#f472b6",
  "Defensive Fullback":  "#94a3b8",
  "Carrying Wingback":   "#facc15",
  "Possession Fullback": "#a78bfa",
}};
const TIER_COLORS = {{ "Excellent":"#00e676","Good":"#4da6ff","Average":"#ffc107","Poor":"#ff5252" }};

function getTier(s) {{
  if (s >= 2.75) return "Excellent";
  if (s >= 2.00) return "Good";
  if (s >= 1.21) return "Average";
  return "Poor";
}}

const W = Math.min(document.documentElement.clientWidth, 1100);
const H = 530;
const margin = {{ top: 28, right: 28, bottom: 58, left: 68 }};
const innerW = W - margin.left - margin.right;
const innerH = H - margin.top - margin.bottom;

const svg = d3.select("#chart").attr("width", W).attr("height", H);
const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

const xVals = allPlayers.map(p => +p[xKey]);
const yVals = allPlayers.map(p => +p[yKey]);
const xPad = (d3.max(xVals) - d3.min(xVals)) * 0.09;
const yPad = (d3.max(yVals) - d3.min(yVals)) * 0.09;

const xScale = d3.scaleLinear()
  .domain([d3.min(xVals)-xPad, d3.max(xVals)+xPad]).range([0,innerW]);
const yScale = d3.scaleLinear()
  .domain([d3.min(yVals)-yPad, d3.max(yVals)+yPad]).range([innerH,0]);
const sizeScale = d3.scaleSqrt()
  .domain([d3.min(allPlayers, p=>p.Sorcerer_Score), d3.max(allPlayers, p=>p.Sorcerer_Score)])
  .range([20,44]);

// Grid & axes
g.append("g").attr("class","grid").attr("transform",`translate(0,${{innerH}})`)
  .call(d3.axisBottom(xScale).tickSize(-innerH).tickFormat(""));
g.append("g").attr("class","grid")
  .call(d3.axisLeft(yScale).tickSize(-innerW).tickFormat(""));
g.append("g").attr("class","axis").attr("transform",`translate(0,${{innerH}})`)
  .call(d3.axisBottom(xScale).ticks(6));
g.append("g").attr("class","axis").call(d3.axisLeft(yScale).ticks(6));
g.append("text").attr("class","axis-label")
  .attr("x",innerW/2).attr("y",innerH+46).attr("text-anchor","middle").text("{x_label}");
g.append("text").attr("class","axis-label")
  .attr("transform","rotate(-90)").attr("x",-innerH/2).attr("y",-52)
  .attr("text-anchor","middle").text("{y_label}");

// Defs
const defs = svg.append("defs");
allPlayers.forEach((p,i) => {{
  defs.append("clipPath").attr("id",`clip-${{i}}`)
    .append("circle").attr("r", sizeScale(p.Sorcerer_Score));
}});

// Tooltip
const tooltip = document.getElementById("tooltip");
function showTooltip(event, p) {{
  document.getElementById("tt-img").src = p.Icons_URL || "{FALLBACK_IMAGE}";
  document.getElementById("tt-name").textContent = p.Player_Name;
  document.getElementById("tt-meta").textContent = p.Team + " · " + p.Position + " · " + p.Role_Tag;
  document.getElementById("tt-x").textContent = (+p[xKey]).toFixed(2);
  document.getElementById("tt-y").textContent = (+p[yKey]).toFixed(2);
  document.getElementById("tt-ss").textContent = p.Sorcerer_Score;
  const tier = getTier(p.Sorcerer_Score);
  const tierEl = document.getElementById("tt-tier");
  tierEl.textContent = "● " + tier;
  tierEl.style.color = TIER_COLORS[tier];
  tierEl.style.background = TIER_COLORS[tier] + "22";
  document.getElementById("tt-insight").textContent = p.insight || "";
  tooltip.classList.add("visible");
  moveTooltip(event);
}}
function moveTooltip(event) {{
  const tw=260,th=220;
  let left=event.clientX+18, top=event.clientY-70;
  if(left+tw>window.innerWidth) left=event.clientX-tw-18;
  if(top+th>window.innerHeight) top=window.innerHeight-th-10;
  if(top<0) top=10;
  tooltip.style.left=left+"px"; tooltip.style.top=top+"px";
}}
function hideTooltip() {{ tooltip.classList.remove("visible"); }}

// Draw nodes
const nodes = g.selectAll(".node").data(allPlayers).enter()
  .append("g").attr("class","node")
  .attr("id", (p,i) => `node-${{i}}`)
  .attr("transform", p => `translate(${{xScale(+p[xKey])}},${{yScale(+p[yKey])}})`);

nodes.append("circle").attr("class","dot-circle")
  .attr("r", p=>sizeScale(p.Sorcerer_Score))
  .attr("fill", p=>ROLE_COLORS[p.Role_Tag]||"#4da6ff")
  .attr("fill-opacity",0.82).attr("stroke","#fff").attr("stroke-width",1.5)
  .style("opacity",0).style("cursor","pointer");

nodes.append("circle").attr("class","dot-ring")
  .attr("r", p=>sizeScale(p.Sorcerer_Score)+2.5)
  .attr("fill","none")
  .attr("stroke", p=>ROLE_COLORS[p.Role_Tag]||"#4da6ff")
  .attr("stroke-width",2.5).style("cursor","pointer");

nodes.append("image").attr("class","dot-image")
  .attr("href", p=>p.Icons_URL||"{FALLBACK_IMAGE}")
  .attr("x", p=>-sizeScale(p.Sorcerer_Score))
  .attr("y", p=>-sizeScale(p.Sorcerer_Score))
  .attr("width",  p=>sizeScale(p.Sorcerer_Score)*2)
  .attr("height", p=>sizeScale(p.Sorcerer_Score)*2)
  .attr("clip-path",(p,i)=>`url(#clip-${{i}})`)
  .attr("preserveAspectRatio","xMidYMid slice")
  .style("cursor","pointer");

// Remove ✕ button (appears on hover)
nodes.each(function(p, i) {{
  const r = sizeScale(p.Sorcerer_Score);
  const btn = d3.select(this).append("g")
    .attr("class","remove-btn")
    .attr("transform", `translate(${{r-4}},${{-r+4}})`)
    .style("cursor","pointer")
    .on("click", function(event) {{
      event.stopPropagation();
      removePlayer(i, p.Player_Name);
    }});
  btn.append("circle").attr("r",9).attr("fill","#ff5252").attr("stroke","#fff").attr("stroke-width",1.5);
  btn.append("text").attr("text-anchor","middle").attr("dy","0.35em")
    .attr("fill","#fff").attr("font-size","11px").attr("font-weight","700").text("✕");
}});

// Invisible hit area
nodes.append("circle")
  .attr("r", p=>sizeScale(p.Sorcerer_Score)+5)
  .attr("fill","transparent").style("cursor","pointer")
  .on("mouseover", function(event,p) {{
    if(removedSet.has(p.Player_Name)) return;
    d3.select(this.parentNode).raise();
    d3.select(this.parentNode).selectAll("image,.dot-circle,.dot-ring")
      .transition().duration(160).attr("transform","scale(1.22)");
    showTooltip(event,p);
  }})
  .on("mousemove", moveTooltip)
  .on("mouseout", function(event,p) {{
    d3.select(this.parentNode).selectAll("image,.dot-circle,.dot-ring")
      .transition().duration(160).attr("transform","scale(1)");
    hideTooltip();
  }});

// ── Remove / Restore logic ──
function removePlayer(i, name) {{
  hideTooltip();
  removedSet.add(name);
  d3.select(`#node-${{i}}`).transition().duration(300)
    .style("opacity", 0).style("pointer-events","none");
  updateRemovedPanel();
}}

function restorePlayer(name) {{
  removedSet.delete(name);
  allPlayers.forEach((p,i) => {{
    if(p.Player_Name === name) {{
      d3.select(`#node-${{i}}`).transition().duration(300)
        .style("opacity",1).style("pointer-events","all");
    }}
  }});
  updateRemovedPanel();
}}

function updateRemovedPanel() {{
  const panel = document.getElementById("removed-panel");
  const chips = document.getElementById("removed-chips");
  chips.innerHTML = "";
  if(removedSet.size === 0) {{
    panel.classList.remove("visible");
    return;
  }}
  panel.classList.add("visible");
  removedSet.forEach(name => {{
    const chip = document.createElement("div");
    chip.className = "restore-chip";
    chip.innerHTML = `<span class="plus">+</span><span>${{name}}</span>`;
    chip.onclick = () => restorePlayer(name);
    chips.appendChild(chip);
  }});
}}

function setMode(m) {{
  mode = m;
  document.getElementById("btn-icons").classList.toggle("active", m==="icons");
  document.getElementById("btn-circles").classList.toggle("active", m==="circles");
  g.selectAll(".dot-image").transition().duration(280).style("opacity", m==="icons"?1:0);
  g.selectAll(".dot-ring").transition().duration(280).style("opacity", m==="icons"?1:0);
  g.selectAll(".dot-circle").transition().duration(280).style("opacity", m==="circles"?0.82:0);
}}
</script>
</body>
</html>"""
    st.components.v1.html(html, height=660, scrolling=False)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    try:
        import soccerdata as sd
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        leagues = ["ENG-Premier League","ESP-La Liga","GER-Bundesliga","ITA-Serie A","FRA-Ligue 1"]
        fbref = sd.FBref(leagues=leagues, seasons="2425")
        passing = fbref.read_player_season_stats(stat_type="passing")
        shot_cr = fbref.read_player_season_stats(stat_type="shot_creation")
        carries = fbref.read_player_season_stats(stat_type="possession")
        defense = fbref.read_player_season_stats(stat_type="defense")
        for frame in [passing,shot_cr,carries,defense]:
            frame.reset_index(inplace=True)
        positions = ["MF","DF","FW,MF","MF,FW","DF,MF","MF,DF"]
        passing = passing[passing["pos"].isin(positions)].copy()
        merge_keys = ["player","team","pos","league","season"]
        df = passing.copy()
        for frame in [shot_cr,carries,defense]:
            cols = [c for c in frame.columns if c not in df.columns or c in merge_keys]
            valid_keys = [k for k in merge_keys if k in frame.columns and k in df.columns]
            df = df.merge(frame[cols], on=valid_keys, how="left")
        col_map = {
            "player":"Player_Name","team":"Team","pos":"Position","league":"League",
            "progressive_passes":"Prog_Passes_p90","passes_into_final_third":"Final_Third_Passes_p90",
            "passes_into_penalty_area":"PPA_p90","through_balls":"Through_Balls_p90",
            "key_passes":"Key_Passes_p90","passes_completed":"Total_Passes_p90",
            "pass_cmp_pct":"Pass_Cmp_Pct","sca":"SCA_p90","gca":"GCA_p90",
            "xa":"xA_p90","assists":"Assists_p90","progressive_carries":"Prog_Carries_p90",
            "carries_into_final_third":"Carries_Final_Third_p90",
            "carries_into_penalty_area":"Carries_Pen_Area_p90",
            "passes_long":"Long_Passes_Att_p90","long_pass_cmp_pct":"Long_Pass_Cmp_Pct",
            "switches":"Switches_p90","crosses":"Crosses_Att_p90",
            "tackles":"Tackles_p90","interceptions":"Interceptions_p90",
            "blocks":"Blocks_p90","minutes":"Minutes_Played",
        }
        df.rename(columns={k:v for k,v in col_map.items() if k in df.columns}, inplace=True)
        for col in col_map.values():
            if col not in df.columns: df[col] = 0.0
        df["Cross_Cmp_Pct"]=0.0; df["Image_URL"]=""; df["Icons_URL"]=""
        df["Passing_Efficiency_Ratio"] = (
            (df["Prog_Passes_p90"]+1.5*df["Final_Third_Passes_p90"])
            /df["Total_Passes_p90"].replace(0,np.nan)
        ).fillna(0).round(3)
        df["Sorcerer_Score"]   = df.apply(compute_sorcerer_score, axis=1)
        df["Creativity_Index"] = df.apply(compute_creativity_index, axis=1)
        cluster_features = ["Prog_Passes_p90","SCA_p90","PPA_p90","Prog_Carries_p90",
                            "Tackles_p90","Interceptions_p90","Passing_Efficiency_Ratio","xA_p90"]
        X = df[cluster_features].fillna(0)
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)
        centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=cluster_features)
        role_map = {}
        for i, row in centers.iterrows():
            if row["Tackles_p90"]>2.0 and row["SCA_p90"]<3.0: role_map[i]="Defensive Fullback"
            elif row["Prog_Passes_p90"]>6.5 and row["SCA_p90"]>4.0: role_map[i]="Advanced Playmaker"
            elif row["Prog_Carries_p90"]>5.0: role_map[i]="Carrying Wingback"
            elif row["Passing_Efficiency_Ratio"]>0.11: role_map[i]="Wingback Creator"
            else: role_map[i]="Possession Fullback"
        df["Role_Tag"] = df["cluster"].map(role_map)
        df.drop(columns=["cluster"],inplace=True)
        df = df[df["Sorcerer_Score"]>0].sort_values("Sorcerer_Score",ascending=False).reset_index(drop=True)
        return df, True
    except Exception:
        try:
            df = pd.read_csv("trent_sorcerer_stats.csv")
            return df, False
        except FileNotFoundError:
            st.error("⚠️ Neither live data nor 'trent_sorcerer_stats.csv' could be loaded.")
            st.stop()


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
df, is_live = load_data()
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
player_names = df["Player_Name"].tolist()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ System Navigation")
if is_live:
    st.sidebar.success("🟢 Live FBref Data Active")
else:
    st.sidebar.warning("🟡 Using Local CSV (FBref unavailable)")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Go To:", [
    "🏠 Home Screen",
    "📖 The Sorcerer Formula",
    "📊 Data Explorer",
    "⚖️ Tactical Comparison",
    "🧠 Creative Profiles"
])
if is_live and st.sidebar.button("🔄 Refresh Live Data"):
    st.cache_data.clear()
    st.rerun()


# ─────────────────────────────────────────────
# MODE 1: HOME SCREEN
# ─────────────────────────────────────────────
if app_mode == "🏠 Home Screen":
    st.markdown('<p class="big-font">The Sorcerer System</p>', unsafe_allow_html=True)
    st.subheader("Elite Playmaker & Fullback Tactical Profiling Dashboard")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    with col1: st.metric("Players Tracked", len(df))
    with col2: st.metric("Metrics Analyzed", len(numeric_cols))
    with col3:
        top_s = df.loc[df["Sorcerer_Score"].idxmax()]
        st.metric("Highest Sorcerer Score", top_s["Sorcerer_Score"], delta=top_s["Player_Name"])
    with col4:
        top_c = df.loc[df["Creativity_Index"].idxmax()]
        st.metric("Highest Creativity Index", top_c["Creativity_Index"], delta=top_c["Player_Name"])

    st.markdown("### 🏆 Top 10 by Sorcerer Score")
    top10 = df.sort_values("Sorcerer_Score", ascending=False).head(10)
    st.dataframe(top10[["Player_Name","Team","Position","Role_Tag","Sorcerer_Score","Passing_Efficiency_Ratio"]],
                 use_container_width=True, hide_index=True)
    st.markdown("---")
    st.markdown("### 📊 Full Sorcerer Score Rankings")
    bar_df = df.sort_values("Sorcerer_Score", ascending=True)
    fig_bar = px.bar(bar_df, x="Sorcerer_Score", y="Player_Name", color="Role_Tag", orientation="h",
                     template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel,
                     hover_data=["Team","Position","Creativity_Index"])
    fig_bar.update_layout(height=max(600, len(df)*26), yaxis_title="", xaxis_title="Sorcerer Score")
    st.plotly_chart(fig_bar, use_container_width=True)


# ─────────────────────────────────────────────
# MODE 2: THE SORCERER FORMULA
# ─────────────────────────────────────────────
elif app_mode == "📖 The Sorcerer Formula":
    st.title("📖 The Sorcerer Score Explained")
    st.markdown("The Sorcerer Score quantifies pure line-breaking playmaking gravity. It strips away safe-passing bias and exposes who actually takes tactical risks to generate attacking rewards.")
    st.markdown("---")
    st.subheader("The Master Equation")
    st.latex(r"\text{Sorcerer Score} = \left( \text{SCA} + 2(\text{PPA}) + \text{xA} \right) \times \left( \frac{\text{Prog Passes} + 1.5(\text{Final 1/3 Passes})}{\text{Total Passes}} \right)")
    st.markdown("---")
    st.subheader("🎚️ Score Tier Thresholds")
    st.caption("Thresholds derived from actual data: 25th pct = 1.21, median = 1.56, 75th pct = 2.47")
    tc1,tc2,tc3,tc4 = st.columns(4)
    tc1.markdown("<div class='score-poor'><div class='score-label'>Poor</div><div class='score-value' style='color:#ff5252'>< 1.21</div><div class='score-tag' style='color:#ff5252'>BOTTOM 25%</div></div>", unsafe_allow_html=True)
    tc2.markdown("<div class='score-average'><div class='score-label'>Average</div><div class='score-value' style='color:#ffc107'>1.21 – 1.99</div><div class='score-tag' style='color:#ffc107'>MID RANGE</div></div>", unsafe_allow_html=True)
    tc3.markdown("<div class='score-good'><div class='score-label'>Good</div><div class='score-value' style='color:#4da6ff'>2.00 – 2.74</div><div class='score-tag' style='color:#4da6ff'>ABOVE AVERAGE</div></div>", unsafe_allow_html=True)
    tc4.markdown("<div class='score-excellent'><div class='score-label'>Excellent</div><div class='score-value' style='color:#00e676'>≥ 2.75</div><div class='score-tag' style='color:#00e676'>TOP 25%</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("### 🗡️ Part 1: The Threat Base")
        st.info("**(SCA + 2×PPA + xA)** — volume and quality of final-third output.")
        with st.expander("SCA — Shot Creating Actions"):
            st.markdown("The two offensive actions directly leading to a shot.")
        with st.expander("PPA — Passes into Penalty Area (2×)"):
            st.markdown("Breaking the last line is the hardest thing in football — weighted double.")
        with st.expander("xA — Expected Assists"):
            st.markdown("Mathematical probability a pass results in an assist based on shot quality.")
    with col2:
        st.markdown("### ⚙️ Part 2: The Efficiency Multiplier")
        st.success("**((Prog + 1.5×Final 3rd) / Total)** — punishes safe, sideways passing.")
        with st.expander("Progressive Passes"):
            st.markdown("Passes that move the ball significantly closer to the opponent's goal.")
        with st.expander("Final 1/3 Passes (1.5×)"):
            st.markdown("Passes entering the attacking third — the most dangerous zone.")
        with st.expander("Total Passes — The Denominator"):
            st.markdown("High volume dilutes the score unless the passes are progressive.")


# ─────────────────────────────────────────────
# MODE 3: DATA EXPLORER
# ─────────────────────────────────────────────
elif app_mode == "📊 Data Explorer":
    st.title("📊 Interactive Metric Explorer")
    st.markdown("Hover over any player to see their face, stats, and a tactical insight. Toggle markers. **Hover → click ✕ to remove a player** from the chart. Click their name in the bar at the top to restore.")

    col1,col2 = st.columns(2)
    with col1:
        xi = numeric_cols.index("Prog_Passes_p90") if "Prog_Passes_p90" in numeric_cols else 0
        x_metric = st.selectbox("X-Axis Metric", numeric_cols, index=xi)
    with col2:
        yi = numeric_cols.index("PPA_p90") if "PPA_p90" in numeric_cols else 1
        y_metric = st.selectbox("Y-Axis Metric", numeric_cols, index=yi)

    icon_col = "Icons_URL" if "Icons_URL" in df.columns else "Image_URL"
    players_data = []
    for _, row in df.iterrows():
        entry = {
            "Player_Name":    row["Player_Name"],
            "Team":           row["Team"],
            "Position":       row["Position"],
            "Role_Tag":       row["Role_Tag"],
            "Sorcerer_Score": float(row["Sorcerer_Score"]),
            "Icons_URL":      str(row[icon_col]) if not pd.isna(row[icon_col]) else FALLBACK_IMAGE,
            "insight":        PLAYER_INSIGHTS.get(row["Player_Name"], ""),
            x_metric:         float(row[x_metric]),
        }
        if y_metric != x_metric:
            entry[y_metric] = float(row[y_metric])
        players_data.append(entry)

    x_label = x_metric.replace("_p90"," (p90)").replace("_"," ").title()
    y_label = y_metric.replace("_p90"," (p90)").replace("_"," ").title()
    render_d3_scatter(players_data, x_metric, y_metric, x_label, y_label)


# ─────────────────────────────────────────────
# MODE 4: TACTICAL COMPARISON
# ─────────────────────────────────────────────
elif app_mode == "⚖️ Tactical Comparison":
    st.title("⚖️ Master Tactical Comparison Panel")

    col1,col2 = st.columns(2)
    with col1:
        p1_name = st.selectbox("Player 1", player_names, index=safe_index(player_names,"João Cancelo",1))
    with col2:
        p2_name = st.selectbox("Player 2", player_names, index=safe_index(player_names,"Trent Alexander-Arnold",0))

    if p1_name and p2_name:
        p1 = df[df["Player_Name"]==p1_name].iloc[0]
        p2 = df[df["Player_Name"]==p2_name].iloc[0]

        compare_metrics = [m for m in numeric_cols if m != "Minutes_Played"]

        # Stats table
        table_html = "<table class='comp-table'>\n"
        table_html += (
            f"<tr><th>Metric</th>"
            f"<th>{p1['Player_Name']}<br><span style='font-size:13px;color:gray;font-weight:normal'>{p1['Team']} | {p1['Position']}</span></th>"
            f"<th>{p2['Player_Name']}<br><span style='font-size:13px;color:gray;font-weight:normal'>{p2['Team']} | {p2['Position']}</span></th></tr>\n"
        )
        all_stats_for_summary = []
        for m in compare_metrics:
            v1,v2 = float(p1[m]), float(p2[m])
            all_stats_for_summary.append((m.replace("_p90"," (p90)").replace("_"," ").title(), v1, v2))
            if v1>v2:
                c1=f"{v1:.2f} <span class='win-arrow'>↑</span>"
                c2=f"{v2:.2f} <span class='lose-arrow'>↓</span>"
            elif v2>v1:
                c1=f"{v1:.2f} <span class='lose-arrow'>↓</span>"
                c2=f"{v2:.2f} <span class='win-arrow'>↑</span>"
            else:
                c1=f"{v1:.2f} <span style='color:gray'>-</span>"
                c2=f"{v2:.2f} <span style='color:gray'>-</span>"
            label = m.replace("_p90"," (p90)").replace("_"," ").title().replace("Pct","%")
            table_html += f"<tr><td style='font-weight:bold;color:#e0e0e0'>{label}</td><td>{c1}</td><td>{c2}</td></tr>\n"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # Radar
        st.markdown("---")
        st.subheader("Radar Chart Overlay")
        radar_metrics = ["Prog_Passes_p90","PPA_p90","SCA_p90","Through_Balls_p90","xA_p90","Final_Third_Passes_p90"]
        radar_labels  = ["Progressive Passes","Passes into Pen Area","SCA","Through Balls","xA","Final Third Passes"]
        fig = go.Figure()
        for player in [p1_name, p2_name]:
            pd_ = df[df["Player_Name"]==player].iloc[0]
            fig.add_trace(go.Scatterpolar(
                r=[pd_[m] for m in radar_metrics],
                theta=radar_labels, fill="toself", name=player
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark", height=520, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        # ── TACTICAL SUMMARY ──
        st.markdown("---")
        has_key = bool(get_api_key())
        st.subheader("🤖 AI Tactical Summary" if has_key else "📋 Tactical Summary")

        cache_key = f"summary_{p1_name}_{p2_name}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = None

        col_gen1, col_gen2, _ = st.columns([1.5, 1.5, 5])
        with col_gen1:
            generate_btn = st.button("⚡ Generate Analysis", type="primary")
        with col_gen2:
            if st.session_state[cache_key]:
                if st.button("🔄 Regenerate"):
                    st.session_state[cache_key] = None
                    st.rerun()

        if generate_btn:
            label = f"Analysing {p1_name} vs {p2_name}..." if has_key else "Building tactical breakdown..."
            with st.spinner(label):
                result = generate_comparison_summary(p1.to_dict(), p2.to_dict(), all_stats_for_summary)
                st.session_state[cache_key] = result

        if st.session_state[cache_key]:
            mode, payload = st.session_state[cache_key]

            section_icons = {
                "Attacking Output":       "⚔️",
                "Passing Profile":        "🎯",
                "Ball Carrying":          "🏃",
                "Defensive Contribution": "🛡️",
                "Verdict":                "⚖️",
            }

            if mode == "ai":
                # Parse the raw AI text into sections
                sections = parse_summary_sections(payload)
            else:
                # Already structured list of (title, body) tuples
                sections = payload

            if sections:
                badge = "<span style='font-size:11px;background:#1e3a5f;color:#60a5fa;padding:2px 8px;border-radius:20px;margin-left:8px;font-weight:700;'>AI</span>" if mode == "ai" else ""
                st.markdown("<div class='summary-box'>", unsafe_allow_html=True)
                st.markdown(f"<h3>📋 {p1_name} vs {p2_name} — Full Tactical Breakdown{badge}</h3>", unsafe_allow_html=True)
                for title, body in sections:
                    icon = section_icons.get(title, "📌")
                    if title == "Verdict":
                        st.markdown(f"<div class='summary-verdict'><strong>{icon} Verdict:</strong> {body}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p><strong>{icon} {title}:</strong> {body}</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("Could not parse the summary. Please try regenerating.")
        elif not generate_btn:
            if has_key:
                st.info("Click **⚡ Generate Analysis** to get an AI-written tactical breakdown of this matchup.")
            else:
                st.info("Click **⚡ Generate Analysis** to get a stat-driven tactical breakdown. Add `ANTHROPIC_API_KEY` to your Railway variables to upgrade to full AI analysis.")


# ─────────────────────────────────────────────
# MODE 5: CREATIVE PROFILES
# ─────────────────────────────────────────────
elif app_mode == "🧠 Creative Profiles":
    st.title("🧠 Comprehensive Creative Profiles")

    selected = st.multiselect(
        "Search players (start typing...)",
        options=player_names,
        default=["Trent Alexander-Arnold"] if "Trent Alexander-Arnold" in player_names else [player_names[0]]
    )

    if selected:
        filtered = df[df["Player_Name"].isin(selected)]
        for _, p in filtered.iterrows():
            with st.expander(f"📂 {p['Player_Name']}", expanded=True):
                col_img,col_info = st.columns([1,4])
                with col_img:
                    st.image(get_image_url(p.get("Image_URL")), width=150)
                with col_info:
                    st.markdown(f"# {p['Player_Name']}")
                    st.markdown(f"### {p['Team']} | {p['Position']}")
                    st.markdown(f"**Role:** `{p['Role_Tag']}`")
                    tier_label,tier_color,tier_class = score_tier(p["Sorcerer_Score"])
                    st.markdown(
                        f"<div class='{tier_class}'>"
                        f"<div class='score-label'>Sorcerer Score</div>"
                        f"<div class='score-value' style='color:{tier_color}'>{p['Sorcerer_Score']}</div>"
                        f"<div class='score-tag' style='color:{tier_color}'>● {tier_label.upper()}</div>"
                        f"</div>", unsafe_allow_html=True
                    )
                st.markdown("---")
                facts = PLAYER_FACTS.get(p["Player_Name"])
                if facts:
                    st.markdown("### 🔍 Player Intel")
                    for icon,badge,text in facts:
                        st.markdown(
                            f"<div class='fact-box'>"
                            f"<div class='fact-badge'>{badge}</div><br>"
                            f"<span class='fact-icon'>{icon}</span>"
                            f"<span class='fact-text'>{text}</span>"
                            f"</div>", unsafe_allow_html=True
                        )
                st.markdown("---")
                tab1,tab2,tab3,tab4 = st.tabs(["🎯 Danger Zone","🚀 Progression & Passing","🏃 Ball Carrying","🛡️ Defense"])
                with tab1:
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("SCA p90", p["SCA_p90"]); c2.metric("GCA p90", p["GCA_p90"])
                    c3.metric("PPA p90", p["PPA_p90"]); c4.metric("Through Balls", p["Through_Balls_p90"])
                    c5,c6,c7,c8 = st.columns(4)
                    c5.metric("Key Passes", p["Key_Passes_p90"]); c6.metric("xA", p["xA_p90"])
                    c7.metric("Assists", p["Assists_p90"]); c8.metric("Crosses Att.", p["Crosses_Att_p90"])
                with tab2:
                    c1,c2,c3,c4 = st.columns(4)
                    c1.metric("Prog Passes", p["Prog_Passes_p90"]); c2.metric("Final 3rd Passes", p["Final_Third_Passes_p90"])
                    c3.metric("Total Passes", p["Total_Passes_p90"]); c4.metric("Pass Cmp %", f"{p['Pass_Cmp_Pct']}%")
                    c5,c6,c7,c8 = st.columns(4)
                    c5.metric("Long Passes Att.", p["Long_Passes_Att_p90"]); c6.metric("Long Pass Cmp %", f"{p['Long_Pass_Cmp_Pct']}%")
                    c7.metric("Switches", p["Switches_p90"]); c8.metric("Efficiency Ratio", p["Passing_Efficiency_Ratio"])
                with tab3:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Prog Carries", p["Prog_Carries_p90"])
                    c2.metric("Carries Final Third", p["Carries_Final_Third_p90"])
                    c3.metric("Carries into Box", p["Carries_Pen_Area_p90"])
                with tab4:
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Tackles", p["Tackles_p90"])
                    c2.metric("Interceptions", p["Interceptions_p90"])
                    c3.metric("Blocks", p["Blocks_p90"])
