import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dedup Error Explorer", layout="wide")
st.title("🔍 Dedup Error Explorer")

# ── data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    gold = pd.read_csv("gold.csv")
    bgi  = pd.read_csv("../../../data/bgi_large.csv", sep="|")

    gold.columns = gold.columns.str.strip()
    bgi.columns  = bgi.columns.str.strip()

    # rename similar_to_id in each df before any merge so they don't collide
    # gold.similar_to_id  = what the LLM thought was the match
    # bgi.similar_to_id   = what the pipeline thought was the match
    gold = gold.rename(columns={"similar_to_id": "llm_similar_to_id",
                                "verdict":        "verdict"})   # keep verdict as-is
    bgi  = bgi.rename(columns={"similar_to_id": "bgi_similar_to_id"})

    # merge bgi's pipeline verdict + its similar_to_id into gold on id
    gold = gold.merge(
        bgi[["id", "bgi_similar_to_id", "verdict"]].rename(columns={"verdict": "bgi_verdict"}),
        on="id", how="left"
    )

    return gold, bgi

gold, bgi = load_data()

# ── scenario + error type (sidebar) ───────────────────────────────────────────

st.sidebar.header("⚙️ Configuration")

scenario = st.sidebar.radio(
    "Paraphrase treated as:",
    ["DUPLICATE", "NEW"],
    key="scenario",
)

if scenario == "DUPLICATE":
    gold["y_true"] = (gold["llm_verdict"] != "NEW").astype(int)
    gold["y_pred"] = (gold["bgi_verdict"] != "NEW").astype(int)
else:
    gold["y_true"] = (gold["llm_verdict"] == "DUPE").astype(int)
    gold["y_pred"] = (gold["bgi_verdict"] == "DUPE").astype(int)

error_type = st.sidebar.radio(
    "Error type:",
    ["False Positives (FP)", "False Negatives (FN)"],
    key="error_type",
)

if error_type == "False Positives (FP)":
    errors = gold[(gold["y_true"] == 0) & (gold["y_pred"] == 1)].copy()
    badge  = "FP"
    desc   = "Ground truth = NEW  but predicted = DUPE/PARAPHRASE"
else:
    errors = gold[(gold["y_true"] == 1) & (gold["y_pred"] == 0)].copy()
    badge  = "FN"
    desc   = "Ground truth = DUPE/PARAPHRASE  but predicted = NEW"

errors = errors.reset_index(drop=True)
st.sidebar.metric("Total errors", len(errors))

st.sidebar.divider()
sort_col = st.sidebar.selectbox("Sort by", ["id", "conf"])
sort_asc  = st.sidebar.radio("Order", ["↑ Asc", "↓ Desc"], horizontal=True) == "↑ Asc"
errors = errors.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

# ── layout ────────────────────────────────────────────────────────────────────

st.caption(f"**{badge}** — {desc}  |  {len(errors)} cases")
st.divider()

left, right = st.columns([1, 2], gap="large")

# ── left: dropdown selector ───────────────────────────────────────────────────

with left:
    st.subheader("📋 Select message")

    if errors.empty:
        st.info("No errors in this configuration.")
        st.stop()

    # build labels for the dropdown
    options = {
        row["id"]: f"ID {int(row['id'])}  |  conf {row['conf']:.3f}  |  {str(row['text'])[:55]}…"
        for _, row in errors.iterrows()
    }

    selected_id = st.selectbox(
        "Error messages",
        options=list(options.keys()),
        format_func=lambda x: options[x],
        label_visibility="collapsed",
    )

# ── right: detail view ────────────────────────────────────────────────────────

with right:
    msg = errors[errors["id"] == selected_id].iloc[0]

    # ── top metrics ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ID",           int(msg["id"]))
    c2.metric("Confidence",   f"{msg['conf']:.3f}")
    c3.metric("Ground Truth", msg["llm_verdict"])
    c4.metric("Prediction",   msg["verdict"])

    st.divider()

    # ── current message ───────────────────────────────────────────────────────
    st.markdown("#### 📝 Message text")
    st.text_area("msg_text", value=msg["text"], height=110,
                 disabled=True, label_visibility="collapsed")

    if pd.notna(msg.get("reason")):
        st.markdown("**💭 LLM reason**")
        st.text_area("msg_reason", value=msg["reason"], height=70,
                     disabled=True, label_visibility="collapsed")

    st.divider()

    # ── two reference columns: BGI match vs LLM match ─────────────────────────
    st.markdown("#### 🔗 What each system matched it to")

    bgi_sim_id = msg.get("similar_to_id")          # BGI's match (from pipeline)
    llm_sim_id = msg.get("similar_to_id")           # LLM's match (from gold)
    # gold has ONE similar_to_id column — rename clearly after load (see below)
    # we split them at load time; see cache block above
    bgi_sim_id = msg.get("bgi_similar_to_id")
    llm_sim_id = msg.get("llm_similar_to_id")

    ref_left, ref_right = st.columns(2, gap="medium")

    # helper to render a reference block
    def render_ref(col, label, sim_id, source_df, id_col="id", color="info"):
        with col:
            st.markdown(f"**{label}**")
            if pd.isna(sim_id):
                st.warning("No match recorded.")
                return
            sim_id = int(sim_id)
            row = source_df[source_df[id_col] == sim_id]
            if row.empty:
                st.error(f"ID {sim_id} not found.")
                return
            row = row.iloc[0]
            m1, m2 = st.columns(2)
            m1.metric("ID",      sim_id)
            m2.metric("Verdict", row.get("verdict", "—"))
            if "conf" in row:
                st.caption(f"conf: {row['conf']:.3f}")
            text = str(row.get("text", ""))
            if color == "info":
                st.info(text)
            else:
                st.success(text)

    render_ref(ref_left,  "🤖 BGI pipeline matched to", bgi_sim_id, bgi,  color="info")
    render_ref(ref_right, "🧠 LLM matched to",          llm_sim_id, gold, color="success")

    # ── current message vs both refs ──────────────────────────────────────────
    st.divider()
    st.markdown("#### 📝 Current message")
    st.text_area("msg_text_bottom", value=msg["text"], height=100,
                 disabled=True, label_visibility="collapsed")