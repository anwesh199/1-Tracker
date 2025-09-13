# app.py
# ==================== Fear → Top 1% Tracker (Advanced, Railway-ready) ====================
# - iOS-like UI with custom CSS widgets & Plotly animations
# - Domain/Task tracking with XP, streaks, unlocks, decay
# - Self-learning: adjusts goals/difficulty & domain weights from history
# - Smart Assistant: natural-language commands to modify data
# - SQLite persistence (safe restarts on Railway)
# Run locally:  streamlit run app.py
# Railway:     Procfile provided; uses $PORT

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import re
import sqlite3
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------- CONFIG --------------------------------
st.set_page_config(page_title="Fear → Top 1% Tracker", page_icon="🚀", layout="wide")
DB_PATH = Path("tracker.db")

DEFAULT_DOMAINS = {
    "Coding": ["SQL", "Python", "SAS", "Tableau", "Power BI"],
    "Driving": ["Parking", "Traffic", "Highway"],
    "Business": ["Learning", "Idea Generation", "Execution"],
    "Trading": ["Paper Trading", "Backtesting", "Real Trading"],
    "Body Discipline": ["Workout", "Diet Logging", "Advanced Diet"]
}

LEVELS = [("Beginner", 0, 99), ("Intermediate", 100, 299), ("Pro", 300, 699), ("Top 1%", 700, 10**9)]
BASE_XP = {"daily": 10, "weekly": 50, "monthly": 200}
UNLOCKS = {
    "Python":        "SELECT SUM(xp) FROM tasks WHERE task='SQL'      AND xp>=300",
    "Tableau":       "SELECT SUM(xp) FROM tasks WHERE task='Python'   AND xp>=300",
    "Power BI":      "SELECT SUM(xp) FROM tasks WHERE task='Python'   AND xp>=300",
    "SAS":           "SELECT SUM(xp) FROM tasks WHERE task='Python'   AND xp>=300",
    "Highway":       "SELECT SUM(xp) FROM tasks WHERE task='Traffic'  AND xp>=200",
    "Execution":     "SELECT SUM(xp) FROM tasks WHERE task='Idea Generation' AND xp>=150",
    "Real Trading":  "SELECT SUM(xp) FROM tasks WHERE task='Paper Trading'   AND xp>=200",
    "Advanced Diet": "SELECT MAX(streak) FROM tasks WHERE task='Workout'     AND streak>=14",
}
# Difficulty (1–3) influences XP; Self-learning will adjust slightly over time
DEFAULT_DIFFICULTY = {
    "SQL": 2, "Python": 3, "SAS": 2, "Tableau": 2, "Power BI": 2,
    "Parking": 1, "Traffic": 2, "Highway": 3,
    "Learning": 1, "Idea Generation": 2, "Execution": 3,
    "Paper Trading": 1, "Backtesting": 2, "Real Trading": 3,
    "Workout": 2, "Diet Logging": 1, "Advanced Diet": 2
}
# Daily minutes goals; Self-learning will raise/lower by ±10% when appropriate
DEFAULT_GOALS = {
    "SQL": 45, "Python": 45, "SAS": 30, "Tableau": 30, "Power BI": 30,
    "Parking": 20, "Traffic": 20, "Highway": 20,
    "Learning": 15, "Idea Generation": 15, "Execution": 30,
    "Paper Trading": 15, "Backtesting": 30, "Real Trading": 10,
    "Workout": 30, "Diet Logging": 5, "Advanced Diet": 10
}
# Domain weights (for recommender); Self-learning updates weekly based on misses
DEFAULT_WEIGHTS = {"Coding": 0.40, "Body Discipline": 0.20, "Driving": 0.15, "Business": 0.15, "Trading": 0.10}

CARD_CSS = """
<style>
/* iOS-like cards */
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {
  background: rgba(255,255,255,0.75);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 8px 26px rgba(0,0,0,0.08);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.6);
}
.badge {
  display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px;
  background: linear-gradient(135deg,#8ec5fc,#e0c3fc); color:#111; font-weight:600;
  margin-left:8px;
}
.kpill {
  display:inline-block; padding:6px 10px; border-radius:12px; font-size:12px;
  background:#111; color:#fff; margin-right:8px;
}
h1, h2, h3 { font-weight: 700; }
.small { color:#666; font-size:12px; }
.stButton>button { border-radius:12px; padding:10px 14px; font-weight:600; }
</style>
"""

st.markdown(CARD_CSS, unsafe_allow_html=True)

# ------------------------------- DB LAYER --------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def db_init():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      domain TEXT, task TEXT,
      xp INTEGER DEFAULT 0,
      streak INTEGER DEFAULT 0,
      last_done DATE,
      goal_min INTEGER,
      difficulty REAL,
      locked INTEGER DEFAULT 0
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS logs(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      ts TIMESTAMP, date DATE,
      domain TEXT, task TEXT,
      minutes INTEGER, xp_gain INTEGER, ratio REAL,
      note TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta(
      key TEXT PRIMARY KEY,
      value REAL
    )""")
    # Seed if empty
    cur.execute("SELECT COUNT(*) FROM tasks")
    if cur.fetchone()[0] == 0:
        for d, tasks in DEFAULT_DOMAINS.items():
            for t in tasks:
                cur.execute(
                    "INSERT INTO tasks(domain,task,goal_min,difficulty,locked) VALUES (?,?,?,?,?)",
                    (d, t, DEFAULT_GOALS.get(t, 20), float(DEFAULT_DIFFICULTY.get(t, 2)), 0)
                )
        # lock those gated by UNLOCKS initially
        gated = set(UNLOCKS.keys())
        cur.execute("UPDATE tasks SET locked=1 WHERE task IN ({})".format(",".join("?"*len(gated))), tuple(gated))
        # domain weights
        for k, v in DEFAULT_WEIGHTS.items():
            cur.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (f"weight:{k}", float(v)))
    conn.commit(); conn.close()

def fetch_df(query, params=()):
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def execute(query, params=()):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    conn.commit()
    conn.close()

db_init()

# ------------------------------- BUSINESS LOGIC --------------------------------
def get_level(xp:int) -> str:
    for name, lo, hi in LEVELS:
        if lo <= xp <= hi:
            return name
    return "Top 1%"

def get_domain_weights():
    rows = fetch_df("SELECT key,value FROM meta WHERE key LIKE 'weight:%'")
    if rows.empty:
        return DEFAULT_WEIGHTS.copy()
    return {r["key"].split(":",1)[1]: float(r["value"]) for _, r in rows.iterrows()}

def set_domain_weight(domain, value):
    execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (f"weight:{domain}", float(value)))

def is_unlocked(task) -> bool:
    if task not in UNLOCKS:  # not gated
        return True
    # check condition query returns any row
    cond_sql = UNLOCKS[task]
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(cond_sql)
        row = cur.fetchone()
        ok = row and row[0] is not None
    except Exception:
        ok = False
    conn.close()
    return bool(ok)

def calc_decay(last_done):
    if not last_done:
        return 1.0
    idle = (dt.date.today() - dt.datetime.strptime(last_done, "%Y-%m-%d").date()).days
    if idle <= 3: return 1.0
    return max(0.80, 1.0 - 0.01*(idle-3))  # cap −20%

def xp_gain_for(task:str, minutes:int, goal:int, diff:float, current_streak:int, met_goal:bool) -> int:
    base = BASE_XP["daily"]
    ratio_bonus = float(np.clip(minutes/goal, 0.2, 1.5))
    streak_bonus = 1 + min(0.30, current_streak*0.03)
    diff_bonus = 1 + (float(diff)-1)*0.25
    xp = round(base * ratio_bonus * streak_bonus * diff_bonus)
    if met_goal and current_streak in (5,10,20,30): xp += BASE_XP["weekly"]
    return int(xp)

def log_progress(domain, task, minutes, note=""):
    # read current
    row = fetch_df("SELECT * FROM tasks WHERE domain=? AND task=?", (domain, task)).iloc[0]
    last = row["last_done"]
    streak = int(row["streak"])
    today = dt.date.today().isoformat()
    # streak math
    if last == today:
        current_streak = streak  # multiple logs same day keep streak
    elif last == (dt.date.today() - dt.timedelta(days=1)).isoformat():
        current_streak = streak + 1
    else:
        current_streak = 1
    goal = int(row["goal_min"]); diff=float(row["difficulty"])
    met_goal = minutes >= goal
    gain = xp_gain_for(task, minutes, goal, diff, current_streak, met_goal)
    # decay then add
    decay = calc_decay(last) if last else 1.0
    new_xp = int(round(int(row["xp"])*decay) + gain)
    execute("UPDATE tasks SET xp=?, streak=?, last_done=? WHERE id=?",
            (new_xp, current_streak, today, int(row["id"])))
    ratio = minutes/goal if goal>0 else 1.0
    execute("INSERT INTO logs(ts,date,domain,task,minutes,xp_gain,ratio,note) VALUES(?,?,?,?,?,?,?,?)",
            (pd.Timestamp.utcnow().isoformat(), today, domain, task, int(minutes), int(gain), float(ratio), note))
    # attempt unlocks
    maybe_unlock_all()
    # self-learning updates (lightweight)
    self_learning_adjustments(task)

def maybe_unlock_all():
    gated = list(UNLOCKS.keys())
    if not gated: return
    df = fetch_df("SELECT task FROM tasks WHERE task IN ({})".format(",".join("?"*len(gated))), tuple(gated))
    for t in df["task"].tolist():
        if is_unlocked(t):
            execute("UPDATE tasks SET locked=0 WHERE task=?", (t,))

def self_learning_adjustments(task_name:str):
    """
    - If last 5 ratios for task >1.2 → raise goal by +10% (max +50% above default)
    - If last 5 ratios for task <0.6 → lower goal by -10% (min -40% below default)
    - If you frequently miss a domain (avg ratio <0.7 over 7 days) → boost domain weight slightly
    - Difficulty nudges: if ratio consistently >1.3, downshift difficulty by 0.1; if <0.5, upshift by 0.1 (clamp 1..3)
    """
    logs = fetch_df("SELECT * FROM logs WHERE task=? ORDER BY id DESC LIMIT 5", (task_name,))
    if len(logs) >= 3:
        ratios = logs["ratio"].astype(float).values
        avg = ratios.mean()
        # adjust goal
        task_row = fetch_df("SELECT domain,task,goal_min FROM tasks WHERE task=?", (task_name,)).iloc[0]
        base = DEFAULT_GOALS.get(task_name, 20)
        cur_goal = int(task_row["goal_min"])
        if avg > 1.2 and cur_goal < int(base*1.5):
            execute("UPDATE tasks SET goal_min=? WHERE task=?", (int(round(cur_goal*1.10)), task_name))
        elif avg < 0.6 and cur_goal > int(base*0.6):
            execute("UPDATE tasks SET goal_min=? WHERE task=?", (max(5, int(round(cur_goal*0.90))), task_name))
        # difficulty nudge
        task_now = fetch_df("SELECT difficulty FROM tasks WHERE task=?", (task_name,)).iloc[0]
        diff = float(task_now["difficulty"])
        if avg > 1.3 and diff > 1.0:
            execute("UPDATE tasks SET difficulty=? WHERE task=?", (round(max(1.0, diff-0.1), 2), task_name))
        elif avg < 0.5 and diff < 3.0:
            execute("UPDATE tasks SET difficulty=? WHERE task=?", (round(min(3.0, diff+0.1), 2), task_name))

    # domain weight weekly tuning
    last7 = fetch_df("""
        SELECT domain, AVG(ratio) AS avg_ratio
        FROM logs WHERE date >= ?
        GROUP BY domain
    """, ((dt.date.today()-dt.timedelta(days=7)).isoformat(),))
    if not last7.empty:
        weights = get_domain_weights()
        for _, r in last7.iterrows():
            dom = r["domain"]; avg_r = float(r["avg_ratio"])
            if dom not in weights: continue
            if avg_r < 0.70: weights[dom] = min(0.50, weights[dom]+0.02)
            elif avg_r > 1.10: weights[dom] = max(0.10, weights[dom]-0.02)
        for k, v in weights.items(): set_domain_weight(k, v)

# ------------------------------- SMART ASSISTANT --------------------------------
ASSIST_HELP = """
**Examples**
- `log 30 min sql`  → log 30 minutes to SQL
- `log 45 minutes Python "ETL practice"` → with note
- `set goal python 60`
- `add task "Meditation" under "Body Discipline"`
- `add domain "Finance"`
- `rename task "Diet Logging" to "Nutrition Log"`
- `lock task Highway` / `unlock task Highway`
- `reset task SQL` / `reset all`
- `show stats` (quick summary)
"""

def assistant_handle(cmd: str) -> str:
    s = cmd.strip().lower()

    # log minutes
    m = re.match(r"log\s+(\d+)\s*(min|mins|minutes|m)?\s+([a-zA-Z ]+?)(?:\s+\"(.+)\")?$", s)
    if m:
        minutes = int(m.group(1)); task = m.group(3).strip().title(); note = m.group(4) or ""
        row = fetch_df("SELECT domain,task,locked FROM tasks WHERE LOWER(task)=?", (task.lower(),))
        if row.empty: return f"Task '{task}' not found."
        if int(row.iloc[0]['locked'])==1: return f"Task '{task}' is locked."
        log_progress(row.iloc[0]['domain'], row.iloc[0]['task'], minutes, note)
        return f"Logged {minutes} min to {task}."

    # set goal
    m = re.match(r"set\s+goal\s+([a-zA-Z ]+)\s+(\d+)", s)
    if m:
        task = m.group(1).strip().title(); val = int(m.group(2))
        execute("UPDATE tasks SET goal_min=? WHERE LOWER(task)=?", (val, task.lower()))
        return f"Set goal for {task} to {val} min."

    # add domain
    m = re.match(r"add\s+domain\s+\"(.+)\"", s)
    if m:
        dom = m.group(1).strip().title()
        # weight default 0.12
        set_domain_weight(dom, 0.12)
        return f"Added domain '{dom}' (no tasks yet). Use: add task \"X\" under \"{dom}\"."

    # add task under domain
    m = re.match(r"add\s+task\s+\"(.+)\"\s+under\s+\"(.+)\"", s)
    if m:
        t = m.group(1).strip().title(); d = m.group(2).strip().title()
        # create with default goal 20/difficulty 2
        execute("INSERT INTO tasks(domain,task,xp,streak,goal_min,difficulty,locked) VALUES(?,?,?,?,?,?,0)",
                (d, t, 0, 0, 20, 2.0))
        return f"Added task '{t}' under '{d}'."

    # rename task
    m = re.match(r"rename\s+task\s+\"(.+)\"\s+to\s+\"(.+)\"", s)
    if m:
        old = m.group(1).strip().title(); new = m.group(2).strip().title()
        execute("UPDATE tasks SET task=? WHERE LOWER(task)=?", (new, old.lower()))
        execute("UPDATE logs SET task=? WHERE LOWER(task)=?", (new, old.lower()))
        return f"Renamed task '{old}' to '{new}'."

    # lock / unlock
    m = re.match(r"(lock|unlock)\s+task\s+([a-zA-Z ]+)", s)
    if m:
        action = m.group(1); task = m.group(2).strip().title()
        val = 1 if action=="lock" else 0
        execute("UPDATE tasks SET locked=? WHERE LOWER(task)=?", (val, task.lower()))
        return f"{action.title()}ed task '{task}'."

    # reset
    if s == "reset all":
        execute("DELETE FROM logs")
        execute("UPDATE tasks SET xp=0, streak=0, last_done=NULL, goal_min=goal_min, difficulty=difficulty")
        return "All progress reset."
    m = re.match(r"reset\s+task\s+([a-zA-Z ]+)", s)
    if m:
        task = m.group(1).strip().title()
        execute("UPDATE tasks SET xp=0, streak=0, last_done=NULL WHERE LOWER(task)=?", (task.lower(),))
        execute("DELETE FROM logs WHERE LOWER(task)=?", (task.lower(),))
        return f"Reset task '{task}'."

    if s == "show stats":
        snap = fetch_df("SELECT domain, SUM(xp) AS xp, AVG(streak) AS avg_streak FROM tasks GROUP BY domain ORDER BY xp DESC")
        return snap.to_string(index=False)

    return "Sorry, I couldn't parse that. Try examples below."

# ------------------------------- UI --------------------------------
st.markdown("<h1>🚀 Fear → Top 1% Tracker</h1>", unsafe_allow_html=True)
st.caption("Advanced, self-learning tracker with iOS-style widgets, progress rings, and a smart assistant.")

with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Dashboard", "Log Progress", "Tasks", "Smart Assistant", "History"], index=0)
    st.markdown("### Tip")
    st.write("Use the **Smart Assistant** to control the app with natural language.")
    st.markdown("<div class='small'>Deploying on Railway? Use the provided Procfile.</div>", unsafe_allow_html=True)

# ---------- helpers for visuals ----------
def gauge(domain, xp, goal=700):
    percent = int(min(100, round((xp/goal)*100)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': f"{domain}"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#32CD32"},
               'steps': [
                   {'range':[0,30], 'color':'#efefef'},
                   {'range':[30,70],'color':'#cfe8ff'},
                   {'range':[70,100],'color':'#d7f7d5'}]}
    ))
    fig.update_layout(height=220, margin=dict(l=10,r=10,t=40,b=10), transition_duration=500)
    return fig

def domain_summary():
    return fetch_df("SELECT domain, SUM(xp) AS xp, AVG(streak) AS avg_streak FROM tasks GROUP BY domain ORDER BY xp DESC")

def tasks_table():
    df = fetch_df("SELECT domain, task, xp, streak, last_done, goal_min, difficulty, locked FROM tasks ORDER BY domain, task")
    df["Level"] = df["xp"].apply(get_level)
    df["Status"] = df["locked"].map({0:"Unlocked",1:"Locked"})
    return df

# ---------- PAGES ----------
if page == "Dashboard":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Domain Overview")
    summary = domain_summary()
    if summary.empty:
        st.info("No data yet. Log something!")
    else:
        ccols = st.columns(len(summary))
        for i, r in summary.iterrows():
            with ccols[i]:
                st.plotly_chart(gauge(r["domain"], r["xp"]), use_container_width=True)
                st.markdown(f"**Level:** {get_level(int(r['xp']))}  &nbsp;&nbsp; <span class='badge'>Avg streak: {int(round(r['avg_streak']))}d</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📈 XP Growth (last 30 days)")
    logs = fetch_df("SELECT date, domain, SUM(xp_gain) AS xp FROM logs WHERE date >= ? GROUP BY date, domain ORDER BY date",
                    ((dt.date.today()-dt.timedelta(days=30)).isoformat(),))
    if logs.empty:
        st.info("No history yet — start logging to see trends.")
    else:
        pivot = logs.pivot(index="date", columns="domain", values="xp").fillna(0)
        fig = px.line(pivot, x=pivot.index, y=pivot.columns, markers=True)
        fig.update_layout(height=320, hovermode="x unified", transition_duration=400, legend_title="Domain")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Log Progress":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("✅ Log Today's Work")
    tdf = tasks_table()
    domains = tdf["domain"].unique().tolist()
    c1, c2, c3 = st.columns([1.2, 1.2, 1])
    domain = c1.selectbox("Domain", domains)
    tasks = tdf.query("domain==@domain")["task"].tolist()
    task = c2.selectbox("Task", tasks)
    rec = tdf[(tdf["domain"]==domain) & (tdf["task"]==task)].iloc[0]
    minutes = c3.number_input("Minutes", min_value=1, value=int(rec["goal_min"]), step=5)
    note = st.text_input("Note (optional)")
    if int(rec["locked"]) == 1:
        st.warning("🔒 This task is currently locked by a prerequisite. Work on pre-req tasks first.")
    if st.button("Log Progress", use_container_width=True):
        log_progress(domain, task, minutes, note)
        st.success(f"Logged {minutes} min to {domain} → {task}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🎯 Recommender (Self-Learning)")
    weights = get_domain_weights()
    # score combines XP rank, streak, idle, and weight
    detail = fetch_df("SELECT domain, task, xp, streak, last_done, goal_min FROM tasks")
    if not detail.empty:
        def score_row(r):
            idle = 999
            if r["last_done"]:
                idle = (dt.date.today()-dt.datetime.strptime(r["last_done"], "%Y-%m-%d").date()).days
            xp_term = 1/(1+r["xp"]); streak_term = 1/(1+r["streak"]); idle_term = min(2.0, idle/7)
            w = weights.get(r["domain"], 0.1)
            return (xp_term*0.5 + streak_term*0.3 + idle_term*0.2) * (1 + (0.4 - w))
        detail["score"] = detail.apply(score_row, axis=1)
        best = detail.sort_values("score", ascending=False).iloc[0]
        st.info(f"**Focus next:** {best['domain']} → {best['task']}  |  Suggested minutes: {int(best['goal_min'])}")
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Tasks":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔍 Tasks & Status")
    st.dataframe(tasks_table(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Smart Assistant":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 Smart Assistant")
    st.markdown("Type a command. The assistant parses it and updates your tracker.")
    st.markdown(ASSIST_HELP)
    cmd = st.text_input("Command")
    if st.button("Run", use_container_width=True) and cmd.strip():
        out = assistant_handle(cmd)
        st.success(out)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "History":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📜 Activity Log")
    history = fetch_df("SELECT ts, date, domain, task, minutes, xp_gain, ratio, note FROM logs ORDER BY id DESC LIMIT 500")
    if history.empty:
        st.info("No logs yet.")
    else:
        st.dataframe(history, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
