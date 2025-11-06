"""
Streamlit app to:
1) Upload raw survey data (CSV/XLSX)
2) Inspect and map columns to Brand Health Tracking concepts
3) Transform into a dashboard‑ready schema (Awareness, Usage, NPS, CSAT, Demographics)
4) Preview charts and export transformed tables
5) Save/Load mapping configs for reuse on future waves

Run locally:
  pip install -U streamlit pandas numpy pyyaml plotly openpyxl
  streamlit run app.py
"""
import io
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="BHT ETL + Dashboard", layout="wide")
st.title("📊 Brand Health Tracking — Reusable ETL & Mini‑Dashboard")

# ----------------------------
# Utilities
# ----------------------------

def _norm(s: str) -> str:
    return str(s or "").strip().lower()


def _contains_any(s: str, keys) -> bool:
    s = _norm(s)
    return any(k in s for k in keys)


def guess_mapping(df: pd.DataFrame) -> Dict:
    """Heuristics to auto-detect likely columns for BHT mapping based on header names."""
    cols = list(df.columns)
    L = [_norm(c) for c in cols]
    idx = { _norm(c): c for c in cols }

    # Respondent id
    resp = next((idx[c] for c in L if _contains_any(c, ["respondent id", "resp_id", "rid", "id_responden"]) ), None)

    # Demographics
    demo_keys = ["gender", "age", "usia", "region", "province", "city", "kota", "occupation", "job", "sec", "income"]
    demos = [idx[c] for c in L if _contains_any(c, demo_keys)]

    # Awareness
    tom = next((idx[c] for c in L if _contains_any(c, ["tom", "top of mind", "top_of_mind", "first mention"]) ), None)
    unaided = [idx[c] for c in L if _contains_any(c, ["unaided", "spont", "open awareness", "ua_"]) and idx[c] != tom]
    aided = [idx[c] for c in L if _contains_any(c, ["aided", "prompted", "aa_"]) and idx[c] != tom]

    # Usage
    ever_used = [idx[c] for c in L if _contains_any(c, ["ever used", "ever_used", "ever tried", "pernah pakai", "pernah gunakan", "ever_buy"]) ]
    bumo = [idx[c] for c in L if _contains_any(c, ["bumo", "most often", "main brand", "usually use", "brand utama", "brand yang paling sering"]) ]
    consider = [idx[c] for c in L if _contains_any(c, ["consider", "consideration", "consider_set", "pertimbangkan"]) ]

    # CSAT & NPS
    csat = next((idx[c] for c in L if _contains_any(c, ["satisfaction", "osat", "kepuasan"]) ), None)
    nps = next((idx[c] for c in L if _contains_any(c, ["nps", "recommend", "rekomendasi", "would you recommend"]) ), None)

    return {
        "respondent_id": resp,
        "demographics": demos,
        "awareness": {"tom": tom, "unaided": unaided, "aided": aided},
        "usage": {"ever_used": ever_used, "bumo": bumo, "consider": consider},
        "satisfaction": {"csat": csat},
        "nps": {"score": nps},
    }

def read_table(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV or XLSX.")


def safe_value_counts(series: pd.Series) -> pd.DataFrame:
    """Compatibility-safe value_counts → DataFrame.
    Works across pandas versions that don't support Series.reset_index(names=...)."""
    s = series.dropna().astype(str).str.strip()
    s = s[s.ne("")]
    vc = s.value_counts(dropna=True)
    # Convert to DataFrame with standard column names
    df = vc.rename_axis("option").reset_index(name="count")
    return df


def infer_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric where possible (for NPS/CSAT)."""
    return pd.to_numeric(series, errors="coerce")

# ---------- Tabulation / Crosstab Helpers ----------

def get_weights(df: pd.DataFrame, weight_col: Optional[str]) -> pd.Series:
    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        return w
    return pd.Series(1, index=df.index, dtype="float64")


def freq_table(df: pd.DataFrame, col: str, weight_col: Optional[str] = None, decimals: int = 1) -> pd.DataFrame:
    w = get_weights(df, weight_col)
    g = df[col].astype(str).str.strip()
    counts = w.groupby(g).sum().rename("count").reset_index().rename(columns={col: "value"})
    total = counts["count"].sum()
    counts["pct_total"] = (counts["count"] / total * 100).round(decimals)
    counts.insert(0, "column", col)
    return counts.sort_values("count", ascending=False).reset_index(drop=True)


def crosstab_table(
    df: pd.DataFrame,
    rows: str,
    cols: str,
    weight_col: Optional[str] = None,
    percent: str = "total",  # 'total' | 'row' | 'col'
    include_totals: bool = True,
    decimals: int = 1,
) -> pd.DataFrame:
    w = get_weights(df, weight_col)
    key = [rows, cols]
    tmp = df[key].copy()
    tmp["__w__"] = w
    piv = pd.pivot_table(tmp, index=rows, columns=cols, values="__w__", aggfunc="sum", fill_value=0)
    # Percentages
    if percent == "row":
        denom = piv.sum(axis=1).replace(0, np.nan)
        pct = (piv.div(denom, axis=0) * 100)
    elif percent == "col":
        denom = piv.sum(axis=0).replace(0, np.nan)
        pct = (piv.div(denom, axis=1) * 100)
    else:  # total
        denom = piv.values.sum()
        denom = denom if denom != 0 else np.nan
        pct = (piv / denom * 100)
    pct = pct.round(decimals)
    # Add totals
    if include_totals:
        piv.loc["Total", :] = piv.sum(axis=0)
        piv.loc[:, "Total"] = piv.sum(axis=1)
        pct.loc["Total", :] = pct.sum(axis=0) if percent != "row" else 100.0
        pct.loc[:, "Total"] = pct.sum(axis=1) if percent != "col" else 100.0
    # Flatten to export-friendly
    piv = piv.reset_index()
    pct = pct.reset_index()
    piv["__type__"] = "count"
    pct["__type__"] = f"%_{percent}"
    out = pd.concat([piv, pct], ignore_index=True)
    return out


def multi_dim_tabulation(
    df: pd.DataFrame,
    dims: List[str],
    weight_col: Optional[str] = None,
    percent_by: str = "total",  # 'total' | name of a level in dims
    decimals: int = 1,
) -> pd.DataFrame:
    w = get_weights(df, weight_col)
    tmp = df[dims].copy()
    tmp["__w__"] = w
    grp = tmp.groupby(dims, dropna=False)["__w__"].sum().rename("count").reset_index()
    total = grp["count"].sum()
    if percent_by == "total":
        grp["pct"] = (grp["count"] / (total if total else np.nan) * 100).round(decimals)
    elif percent_by in dims:
        denom = grp.groupby(percent_by)["count"].transform("sum").replace(0, np.nan)
        grp["pct"] = (grp["count"] / denom * 100).round(decimals)
    else:
        grp["pct"] = np.nan
    return grp


# ----------------------------
# Sidebar — Save/Load Mapping Config
# ----------------------------
with st.sidebar:
    st.header("⚙️ Mapping Config")
    cfg_mode = st.radio("Config Mode", ["New/Update", "Load from JSON"], horizontal=True)
    loaded_cfg: Optional[Dict] = None
    if cfg_mode == "Load from JSON":
        cfg_file = st.file_uploader("Load mapping_config.json", type=["json"], key="cfg")
        if cfg_file is not None:
            loaded_cfg = json.load(cfg_file)
            st.success("Config loaded. You can tweak it below or use as‑is.")

# ----------------------------
# Step 1: Upload Raw Data
# ----------------------------
up = st.file_uploader("1) Upload raw data (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="raw")

if up is None:
    st.info("Upload your raw survey export to begin. Supported: CSV, XLSX.")
    st.stop()

raw_df = read_table(up)
st.success(f"Loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]:,} columns")

with st.expander("Preview raw data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("Weights & Codebook (optional)", expanded=False):
    st.caption("Select a weight column (numeric) and/or upload a codebook CSV: columns = column,value,label")
    columns = list(raw_df.columns)
    weight_opt = st.selectbox("Weight column (optional)", options=["<none>"] + columns)
    weight_col = None if weight_opt == "<none>" else weight_opt

    codebook_file = st.file_uploader("Upload codebook CSV (column,value,label)", type=["csv"]) 
    if codebook_file is not None:
        try:
            cb = pd.read_csv(codebook_file)
            if set(["column", "value", "label"]).issubset(set(cb.columns)):
                mappings = {}
                for colname, sub in cb.groupby("column"):
                    m = dict(zip(sub["value"].astype(str), sub["label"].astype(str)))
                    mappings[colname] = m
                # apply mapping per column if it exists in data
                for colname, m in mappings.items():
                    if colname in raw_df.columns:
                        raw_df[colname] = raw_df[colname].astype(str).map(m).fillna(raw_df[colname])
                st.success("Codebook applied to matching columns.")
            else:
                st.warning("Codebook must have columns: column,value,label")
        except Exception as e:
            st.error(f"Failed to read/apply codebook: {e}")

# ---------- Brand Extraction Helpers ----------
import re

COMMON_PREFIXES = [
    r"^ua[_-]?", r"^aa[_-]?", r"^aw[_-]?", r"^ever[_-]?", r"^everused[_-]?",
    r"^consider[_-]?", r"^consid[_-]?", r"^cs[_-]?", r"^used[_-]?", r"^brand[_-]?",
]
COMMON_SUFFIXES = [r"[_-]?brand$", r"[_-]?used$", r"[_-]?ever$", r"[_-]?consider$", r"[_-]?aided$", r"[_-]?unaided$"]

def extract_brand_from_column(colname: str) -> str:
    """Turn coded column like 'UA_Indomie' or 'consider-sedaap' into 'Indomie' / 'sedaap'."""
    raw = str(colname)
    s = raw
    for p in COMMON_PREFIXES:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    for p in COMMON_SUFFIXES:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    s = re.sub(r"[_-]+", " ", s).strip()
    return s if s else raw

def brands_from_binary_columns(cols: List[str]) -> List[str]:
    return sorted({extract_brand_from_column(c) for c in cols})

columns = list(raw_df.columns)

# Offer auto-detect button
col_aut1, col_aut2 = st.columns([1,2])
with col_aut1:
    if st.button("🔍 Auto-detect mapping from column names", help="Heuristics scan your headers and prefill the mapping fields."):
        st.session_state["auto_cfg"] = guess_mapping(raw_df)
        st.rerun()

auto_cfg = st.session_state.get("auto_cfg")

# ----------------------------
# Step 2–4: Define Mapping
# ----------------------------
st.subheader("2–4) Map columns to BHT concepts")

# Start from loaded/auto config if any
if loaded_cfg is None:
    loaded_cfg = {}
seed_cfg = auto_cfg or loaded_cfg or {}

# Respondent ID (optional but recommended)
resp_id_col = st.selectbox(
    "Respondent ID column (optional but recommended)",
    options=["<none>"] + columns,
    index=(columns.index(seed_cfg.get("respondent_id", "<none>")) + 1) if seed_cfg.get("respondent_id") in columns else 0,
)
if resp_id_col == "<none>":
    resp_id_col = None

# Demographics (multi‑select)
demo_cols = st.multiselect(
    "Demographic columns (gender, age, region, occupation, SEC, etc.)",
    options=columns,
    default=[c for c in seed_cfg.get("demographics", []) if c in columns],
)

# Awareness mapping
st.markdown("### Awareness")
tom_col = st.selectbox(
    "Top‑of‑Mind (single text/brand column)",
    options=["<none>"] + columns,
    index=(columns.index(seed_cfg.get("awareness", {}).get("tom", "<none>")) + 1) if seed_cfg.get("awareness", {}).get("tom") in columns else 0,
)
if tom_col == "<none>":
    tom_col = None

unaided_cols = st.multiselect(
    "Unaided awareness (multi‑select columns or multi‑select coded list)",
    options=columns,
    default=[c for c in seed_cfg.get("awareness", {}).get("unaided", []) if c in columns],
    help="Choose columns that, when marked/selected, mean the brand was recalled unaided."
)

aided_cols = st.multiselect(
    "Aided awareness (multi‑select columns for shown list)",
    options=columns,
    default=[c for c in seed_cfg.get("awareness", {}).get("aided", []) if c in columns],
)

# Usage behaviour / funnel
st.markdown("### Usage Behaviour / Funnel")
ever_used_cols = st.multiselect(
    "Ever used (columns indicating historical use)",
    options=columns,
    default=[c for c in seed_cfg.get("usage", {}).get("ever_used", []) if c in columns],
)
current_used_cols = st.multiselect(
    "Brand used most often / BUMO (columns indicating main brand)",
    options=columns,
    default=[c for c in seed_cfg.get("usage", {}).get("bumo", []) if c in columns],
)
consider_cols = st.multiselect(
    "Consideration set (columns indicating consideration)",
    options=columns,
    default=[c for c in seed_cfg.get("usage", {}).get("consider", []) if c in columns],
)

# Satisfaction and NPS
st.markdown("### Satisfaction & NPS")
csat_col = st.selectbox(
    "Overall Satisfaction (e.g., 1–5 or 1–10 scale)",
    options=["<none>"] + columns,
    index=(columns.index(seed_cfg.get("satisfaction", {}).get("csat", "<none>")) + 1) if seed_cfg.get("satisfaction", {}).get("csat") in columns else 0,
)
if csat_col == "<none>":
    csat_col = None

nps_col = st.selectbox(
    "NPS question (0–10)",
    options=["<none>"] + columns,
    index=(columns.index(seed_cfg.get("nps", {}).get("score", "<none>")) + 1) if seed_cfg.get("nps", {}).get("score") in columns else 0,
)
if nps_col == "<none>":
    nps_col = None

# Save config
cfg = {
    "respondent_id": resp_id_col,
    "demographics": demo_cols,
    "awareness": {"tom": tom_col, "unaided": unaided_cols, "aided": aided_cols},
    "usage": {"ever_used": ever_used_cols, "bumo": current_used_cols, "consider": consider_cols},
    "satisfaction": {"csat": csat_col},
    "nps": {"score": nps_col},
}

col_a, col_b = st.columns(2)
with col_a:
    if st.button("💾 Download mapping config (JSON)"):
        b = io.BytesIO()
        b.write(json.dumps(cfg, indent=2).encode("utf-8"))
        b.seek(0)
        st.download_button(
            label="Save mapping_config.json",
            data=b,
            file_name="mapping_config.json",
            mime="application/json",
        )
with col_b:
    st.caption("Tip: Save this and reuse for the next wave — no remapping needed.")

# ----------------------------
# Step 5: Transform to Dashboard‑Ready Data
# ----------------------------
st.subheader("5) Transform ➜ Dashboard‑ready tables")

# Helper builders

def build_awareness_tables(df: pd.DataFrame, tom: Optional[str], unaided: List[str], aided: List[str]):
    """Build awareness tables (TOM / unaided / aided) safely across pandas versions."""
    tables: Dict[str, pd.DataFrame] = {}

    # TOM
    if tom and tom in df.columns:
        tom_df = safe_value_counts(df[tom])
        tom_df.rename(columns={"option": "brand", "count": "count"}, inplace=True)
        tables["tom"] = tom_df

    # Unaided (initialize at top-level to avoid UnboundLocalError)
    una_list: List[pd.DataFrame] = []
    if unaided:
        for c in unaided:
            if c in df.columns:
                s = df[c]
                sel = s.notna() & (s.astype(str).str.strip().ne("")) & (s.astype(str).str.lower().ne("0"))
                una_list.append(pd.DataFrame([{"brand": c, "count": int(sel.sum())}]))
    if una_list:
        tables["unaided"] = pd.concat(una_list, ignore_index=True)

    # Aided (initialize at top-level to avoid UnboundLocalError)
    aid_list: List[pd.DataFrame] = []
    if aided:
        for c in aided:
            if c in df.columns:
                s = df[c]
                sel = s.notna() & (s.astype(str).str.strip().ne("")) & (s.astype(str).str.lower().ne("0"))
                aid_list.append(pd.DataFrame([{"brand": c, "count": int(sel.sum())}]))
    if aid_list:
        tables["aided"] = pd.concat(aid_list, ignore_index=True)

    return tables


def build_usage_tables(df: pd.DataFrame, ever_cols: List[str], bumo_cols: List[str], consider_cols: List[str]):
    tables = {}
    def aggregate(cols: List[str], name: str):
        items = []
        for c in cols:
            if c in df.columns:
                s = df[c]
                sel = s.notna() & (s.astype(str).str.strip().ne("")) & (s.astype(str).str.lower().ne("0"))
                items.append(pd.DataFrame([{"brand": c, "count": int(sel.sum())}]))
        if items:
            tables[name] = pd.concat(items, ignore_index=True)
    aggregate(ever_cols, "ever_used")
    aggregate(bumo_cols, "bumo")
    aggregate(consider_cols, "consider")
    return tables


def build_satisfaction_table(df: pd.DataFrame, csat_col: Optional[str]):
    if not csat_col or csat_col not in df.columns:
        return None
    s = infer_numeric(df[csat_col])
    return pd.DataFrame({
        "metric": ["mean", "top2_box", "n"],
        "value": [s.mean(skipna=True), (s >= s.max() - 1).mean() if s.notna().any() else np.nan, s.notna().sum()],
    })


def build_nps_table(df: pd.DataFrame, nps_col: Optional[str]):
    if not nps_col or nps_col not in df.columns:
        return None
    s = infer_numeric(df[nps_col])
    detractors = ((s >= 0) & (s <= 6)).sum()
    passives = ((s >= 7) & (s <= 8)).sum()
    promoters = ((s >= 9) & (s <= 10)).sum()
    n = s.notna().sum()
    if n == 0:
        return pd.DataFrame({"metric": ["nps", "n"], "value": [np.nan, 0]})
    nps = (promoters / n - detractors / n) * 100
    return pd.DataFrame({"metric": ["nps", "n", "promoters", "passives", "detractors"],
                         "value": [nps, n, promoters, passives, detractors]})


if st.button("🚀 Transform Data", type="primary"):
    # Build a brand dictionary from mapping + raw values
    tom_brands = []
    if tom_col and tom_col in raw_df.columns:
        tom_brands = (
            raw_df[tom_col].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
        )
    brand_dictionary = {
        "TOM": sorted(tom_brands),
        "Unaided": brands_from_binary_columns(unaided_cols),
        "Aided": brands_from_binary_columns(aided_cols),
        "Ever Used": brands_from_binary_columns(ever_used_cols),
        "BUMO": brands_from_binary_columns(current_used_cols),
        "Consideration": brands_from_binary_columns(consider_cols),
    }

    out = {}
    # Awareness
    aware = build_awareness_tables(raw_df, tom_col, unaided_cols, aided_cols)
    out.update({f"awareness_{k}": v for k, v in aware.items()})
    # Usage
    usage = build_usage_tables(raw_df, ever_used_cols, current_used_cols, consider_cols)
    out.update({f"usage_{k}": v for k, v in usage.items()})
    # CSAT
    csat_tbl = build_satisfaction_table(raw_df, csat_col)
    if csat_tbl is not None:
        out["satisfaction_summary"] = csat_tbl
    # NPS
    nps_tbl = build_nps_table(raw_df, nps_col)
    if nps_tbl is not None:
        out["nps_summary"] = nps_tbl

    # Add brand dictionary as a table for export/preview
    bd_rows = []
    for k, v in brand_dictionary.items():
        if isinstance(v, list):
            for item in v:
                bd_rows.append({"group": k, "brand": item})
        else:
            bd_rows.append({"group": k, "brand": str(v)})
    if bd_rows:
        out["brand_dictionary"] = pd.DataFrame(bd_rows)

    if not out:
        st.warning("No tables generated — please complete the mappings above.")
        st.stop()

    st.success("Transformation complete. Preview tables, brand dictionary, and charts below.")

    with st.expander("🔎 Detected Brand Dictionary", expanded=False):
        st.json(brand_dictionary)

    # ----------------------------
    # Step 6A: Full Tabulation Generator (New)
    # ----------------------------
    st.subheader("📊 6A) Full Tabulation (Frequency Table for Each Column)")

    tab_rows = []
    for col in raw_df.columns:
        ser = raw_df[col].astype(str).str.strip()
        counts = ser.value_counts(dropna=False)
        for val, cnt in counts.items():
            tab_rows.append({
                "column": col,
                "value": val,
                "count": int(cnt)
            })

    tab_df = pd.DataFrame(tab_rows)
    out["tabulation"] = tab_df

    with st.expander("📄 Tabulation Preview", expanded=False):
        st.dataframe(tab_df.head(200), use_container_width=True)

    # ----------------------------
    # Step 6B: Advanced Tabulation & Crosstab (New)
    # ----------------------------
    st.subheader("📐 6B) Advanced Tabulation & Crosstab")

    adv_cols = list(raw_df.columns)
    col1, col2, col3 = st.columns(3)
    with col1:
        row_var = st.selectbox("Row variable", options=["<none>"] + adv_cols, index=0)
    with col2:
        col_var = st.selectbox("Column variable", options=["<none>"] + adv_cols, index=0)
    with col3:
        percent_mode = st.selectbox("Percent base", options=["total", "row", "col"], index=0)
    col4, col5 = st.columns([1,1])
    with col4:
        include_totals = st.checkbox("Include totals", value=True)
    with col5:
        decimals = st.number_input("Decimals", min_value=0, max_value=4, value=1)

    if row_var != "<none>" and col_var != "<none>":
        xt = crosstab_table(
            raw_df, rows=row_var, cols=col_var,
            weight_col=weight_col, percent=percent_mode,
            include_totals=include_totals, decimals=decimals,
        )
        out["crosstab"] = xt
        with st.expander("🔢 Crosstab Preview", expanded=False):
            st.dataframe(xt.head(200), use_container_width=True)

    # Multi-level tabulation
    st.markdown("---")
    st.subheader("🧮 6C) Multi-level Tabulation")
    dims = st.multiselect("Dimensions (up to 3)", options=adv_cols, default=[])
    percent_by = st.selectbox("Percent by", options=["total"] + dims, index=0)
    if len(dims) >= 2:
        mt = multi_dim_tabulation(raw_df, dims=dims[:3], weight_col=weight_col, percent_by=percent_by, decimals=decimals)
        out["multi_tabulation"] = mt
        with st.expander("📑 Multi-level Tabulation Preview", expanded=False):
            st.dataframe(mt.head(200), use_container_width=True)

    # ----------------------------
    # Step 6C: Mini Visual Dashboard
    # ----------------------------
    st.subheader("6) Mini Dashboard Preview")
    tabs = st.tabs(list(out.keys()))
    for i, (name, df_) in enumerate(out.items()):
        with tabs[i]:
            st.dataframe(df_, use_container_width=True)
            # simple chart heuristics
            if "brand" in df_.columns and "count" in df_.columns:
                fig = px.bar(df_.sort_values("count", ascending=False), x="brand", y="count", title=name)
                st.plotly_chart(fig, use_container_width=True)
            elif "metric" in df_.columns and "value" in df_.columns:
                fig = px.bar(df_, x="metric", y="value", title=name)
                st.plotly_chart(fig, use_container_width=True)

    # Package all tables into a single Excel for BI tools
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, df_ in out.items():
            sheet = name[:31]  # Excel sheet name limit
            df_.to_excel(writer, index=False, sheet_name=sheet)
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download dashboard‑ready tables (Excel)",
        data=buffer,
        file_name="bht_dashboard_ready.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Also offer a JSON bundle
    json_bundle = {k: v.to_dict(orient="records") for k, v in out.items()}
    jb = io.BytesIO(json.dumps(json_bundle, indent=2).encode("utf-8"))
    st.download_button(
        label="⬇️ Download JSON bundle",
        data=jb,
        file_name="bht_dashboard_ready.json",
        mime="application/json",
    )

# Footer tips
st.caption(
    "Pro tip: Use the saved JSON mapping to process the next wave with identical structure — just load, upload, transform, and export.")

