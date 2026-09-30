from decimal import Decimal
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from databricks import sql

# ───────────────────────── Config & Secrets ─────────────────────────
# Read from .streamlit/secrets.toml (Local) or App Secrets (Streamlit Cloud)
SERVER_HOSTNAME = st.secrets["DATABRICKS_HOST"]
HTTP_PATH = st.secrets["DATABRICKS_HTTP_PATH"]
ACCESS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

# Unity Catalog: workspace (catalog) -> idp (schema)
SCHEMA = "workspace.idp"
T_INVOICES = f"{SCHEMA}.finance_invoices"
T_SEGMENTS = f"{SCHEMA}.buyer_segments"
T_CLV = f"{SCHEMA}.buyer_clv"
T_ANOMALIES = f"{SCHEMA}.line_item_anomalies"

st.set_page_config(page_title="Invoice IDP Analytics", page_icon="🧾", layout="wide")

# ───────────────────────── Color scheme ─────────────────────────
# Each color has one job so it means the same thing in every tab.
PRIMARY = "#4C72B0"   # magnitudes: revenue, spend, counts
ACCENT = "#DD8452"    # the second series (net vs tax, gross vs net)
GREEN = "#55A868"     # volume (invoice counts)
ALERT = "#C44E52"     # flags only: outliers, cumulative % line
MUTED = "#A9BCD9"     # incomplete or de-emphasised
TIER_COLORS = {"High": "#1F3A63", "Medium": "#4C72B0", "Low": "#A9BCD9"}  # darker = more valuable
BRAND_SEQ = px.colors.qualitative.Safe

px.defaults.color_discrete_sequence = BRAND_SEQ
px.defaults.color_continuous_scale = "Blues"


def get_brand(desc: str) -> str:
    """Brand guessed from the first word of the product description."""
    parts = str(desc).split()
    return parts[0] if parts else "Other"


def color_map(values) -> dict:
    """Stable category -> color mapping so a category keeps its color across charts."""
    return {v: BRAND_SEQ[i % len(BRAND_SEQ)] for i, v in enumerate(sorted(set(values)))}


# ───────────────────────── Data Access ─────────────────────────
def _decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            first = df[c].dropna()
            if len(first) and isinstance(first.iloc[0], Decimal):
                df[c] = df[c].astype(float)
    return df


@st.cache_data(ttl=600, show_spinner="Querying Unity Catalog…")
def load_table(table: str) -> pd.DataFrame:
    """Execute a query against Databricks SQL Warehouse using PAT."""
    with sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=HTTP_PATH,
        access_token=ACCESS_TOKEN,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {table}")
            df = cursor.fetchall_arrow().to_pandas()
            return _decimals_to_float(df)


def require(df: pd.DataFrame, cols: list, table: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"`{table}` is missing expected columns {missing}. Found: {list(df.columns)}")


def fmt_inr(x: float) -> str:
    if x >= 1e7:
        return f"₹{x/1e7:.2f} Cr"
    if x >= 1e5:
        return f"₹{x/1e5:.2f} L"
    return f"₹{x:,.0f}"

# ───────────────────────── Load & prepare ─────────────────────────
INVOICE_COLS = ["path", "invoice_number", "invoice_date", "seller_name", "buyer_name",
                "line_item_number", "line_item_description", "line_item_quantity",
                "line_item_net_price", "line_item_net_worth", "line_item_vat_percent",
                "line_item_gross_worth", "summary_net_worth", "summary_gross_worth", "currency"]
try:
    raw = load_table(T_INVOICES)
    require(raw, INVOICE_COLS, T_INVOICES)
except Exception as e:
    st.error(f"Could not load `{T_INVOICES}`. Check the app's service principal has USE CATALOG / "
             f"USE SCHEMA / SELECT on `{SCHEMA}`.\n\n{e}")
    st.stop()

raw = raw.copy()
raw["invoice_dt"] = pd.to_datetime(raw["invoice_date"], format="%d/%m/%Y", errors="coerce")
raw["tax_value"] = raw["line_item_gross_worth"] - raw["line_item_net_worth"]
n_bad_dates = int(raw["invoice_dt"].isna().sum())
raw_ok = raw[raw["invoice_dt"].notna()].copy()
if raw_ok.empty:
    st.error("No invoice dates could be parsed (expected DD/MM/YYYY).")
    st.stop()
raw_ok["month"] = raw_ok["invoice_dt"].dt.to_period("M").astype(str)

CITY_MAP = {
    "Mysuru": "Karnataka", "Prayagraj": "Uttar Pradesh", "Bhopal": "Madhya Pradesh",
    "Kolhapur": "Maharashtra", "Hyderabad": "Telangana", "Varanasi": "Uttar Pradesh",
    "Surat": "Gujarat", "Guwahati": "Assam", "Kochi": "Kerala", "Ludhiana": "Punjab",
    "Ahmedabad": "Gujarat", "Solapur": "Maharashtra", "Amritsar": "Punjab",
    "Udaipur": "Rajasthan", "Coimbatore": "Tamil Nadu", "Rajkot": "Gujarat",
    "Jodhpur": "Rajasthan", "Nagpur": "Maharashtra", "Indore": "Madhya Pradesh",
    "Patna": "Bihar", "Ranchi": "Jharkhand", "Bhubaneswar": "Odisha",
    "Mumbai": "Maharashtra", "Delhi": "Delhi", "Chennai": "Tamil Nadu",
    "Kolkata": "West Bengal", "Pune": "Maharashtra", "Jaipur": "Rajasthan",
    "Lucknow": "Uttar Pradesh", "Chandigarh": "Chandigarh", "Dehradun": "Uttarakhand",
    "Shimla": "Himachal Pradesh", "Gangtok": "Sikkim", "Agartala": "Tripura",
    "Raipur": "Chhattisgarh", "Panaji": "Goa", "Vadodara": "Gujarat",
    "Warangal": "Telangana", "Visakhapatnam": "Andhra Pradesh", "Madurai": "Tamil Nadu",
    "Mangaluru": "Karnataka", "Hubli": "Karnataka", "Tirupati": "Andhra Pradesh",
    "Gwalior": "Madhya Pradesh", "Jabalpur": "Madhya Pradesh", "Bareilly": "Uttar Pradesh",
    "Aligarh": "Uttar Pradesh", "Gorakhpur": "Uttar Pradesh", "Saharanpur": "Uttar Pradesh",
    "Jhansi": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Agra": "Uttar Pradesh",
    "Aurangabad": "Maharashtra", "Guntur": "Andhra Pradesh", "Meerut": "Uttar Pradesh",
    "Nashik": "Maharashtra", "Nellore": "Andhra Pradesh", "Salem": "Tamil Nadu",
    "Thiruvananthapuram": "Kerala", "Tiruchirappalli": "Tamil Nadu",
}


def get_region(name: str) -> str:
    for city, state in CITY_MAP.items():
        if city.lower() in str(name).lower():
            return state
    return "Unmapped"


# ───────────────────────── Sidebar filters ─────────────────────────
st.sidebar.title("🧾 Invoice IDP")
st.sidebar.caption(f"Source: `{SCHEMA}`")

dmin, dmax = raw_ok["invoice_dt"].min().date(), raw_ok["invoice_dt"].max().date()
date_range = st.sidebar.date_input("Invoice date range", (dmin, dmax), min_value=dmin, max_value=dmax)
if not (isinstance(date_range, tuple) and len(date_range) == 2):
    st.info("Pick an end date in the sidebar date range.")
    st.stop()
all_buyers = sorted(raw_ok["buyer_name"].dropna().unique())
sel_buyers = st.sidebar.multiselect("Buyers (blank = all)", all_buyers)
st.sidebar.caption("Filters apply to the invoice-based tabs. Segments, CLV and Isolation Forest "
                   "tables were computed in the notebook on all data, so only the buyer filter applies to them.")

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

df = raw_ok[(raw_ok["invoice_dt"].dt.date >= date_range[0]) & (raw_ok["invoice_dt"].dt.date <= date_range[1])]
if sel_buyers:
    df = df[df["buyer_name"].isin(sel_buyers)]
if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

inv = (
    df.groupby(["invoice_number", "buyer_name", "invoice_dt"], as_index=False, dropna=False)
    .agg(line_items=("line_item_number", "count"),
         distinct_products=("line_item_description", "nunique"),
         total_qty=("line_item_quantity", "sum"),
         net_value=("line_item_net_worth", "sum"),
         gross_value=("line_item_gross_worth", "sum"),
         tax_value=("tax_value", "sum"),
         stated_gross=("summary_gross_worth", "max"))
)
inv["avg_line_item_value"] = inv["gross_value"] / inv["line_items"]

# ───────────────────────── Header + intro + KPIs ─────────────────────────
st.title("Invoice IDP Analytics")
seller = df["seller_name"].mode().iat[0] if df["seller_name"].notna().any() else "—"
st.caption(f"Seller: **{seller}** · {df['invoice_dt'].min():%d %b %Y} – {df['invoice_dt'].max():%d %b %Y}")

with st.expander("About this dashboard", expanded=True):
    st.markdown(
        f"""
**The dataset.** Invoices issued by **{seller}**, extracted from PDFs by an Intelligent Document
Processing (IDP) pipeline into Unity Catalog. It currently covers **{inv['invoice_number'].nunique():,} invoices**,
**{len(df):,} line items** and **{df['buyer_name'].nunique():,} buyers** between
{df['invoice_dt'].min():%b %Y} and {df['invoice_dt'].max():%b %Y} (the sidebar filters change these numbers).
Each row is one invoice line: product, quantity, unit price, net value, VAT and gross value.

**The problem.** Invoices are processed manually, so the business has no clear view of who its best buyers
are, which products drive revenue, or whether the extracted numbers can be trusted.

**What we want to find out.**
- **Revenue, Buyers, Products:** where the money comes from, and how concentrated it is.
- **Market Basket, Segments & CLV:** how buyers group together and which products sell together.
- **Anomalies, Data Quality:** unusual prices or extraction errors worth checking against the source PDFs.

**Read with care.** This is about one year of data with two invoices per buyer, so segments, CLV and
association rules are rough guides for deciding where to look first, not validated forecasts.

**Colour guide.** Blue is the main measure (darker = more valuable). Orange is the second series
(e.g. net vs tax). Green is invoice volume. Red marks flags only. Light blue means incomplete or lower value.
"""
    )

if n_bad_dates:
    st.warning(f"{n_bad_dates} line item(s) have an unparseable invoice date and are excluded from all charts.")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Invoices", f"{inv['invoice_number'].nunique():,}")
k2.metric("Line items", f"{len(df):,}")
k3.metric("Buyers", f"{df['buyer_name'].nunique():,}")
k4.metric("Gross revenue", fmt_inr(inv["gross_value"].sum()))
k5.metric("Tax (VAT)", fmt_inr(inv["tax_value"].sum()))

if inv["stated_gross"].notna().all():
    delta = inv["gross_value"].sum() - inv["stated_gross"].sum()

tabs = st.tabs(["📈 Revenue", "🏢 Buyers", "📦 Products", "🛒 Market Basket",
                "🎯 Segments & CLV", "🚨 Anomalies", "✅ Data Quality", "🧾 Invoices"])

# ───────────────────────── Revenue ─────────────────────────
with tabs[0]:
    df_m = df.assign(month=df["invoice_dt"].dt.to_period("M").astype(str))
    monthly = (df_m.groupby("month")
               .agg(invoices=("invoice_number", "nunique"), gross=("line_item_gross_worth", "sum"),
                    net=("line_item_net_worth", "sum"), tax=("tax_value", "sum"))
               .reset_index().sort_values("month"))
    edge = {monthly["month"].iloc[0], monthly["month"].iloc[-1]} if len(monthly) > 1 else set()
    colors = [MUTED if m in edge else PRIMARY for m in monthly["month"]]

    fig = go.Figure()
    fig.add_bar(x=monthly["month"], y=monthly["gross"] / 1e6, name="Gross (₹M)", marker_color=colors)
    fig.add_scatter(x=monthly["month"], y=monthly["net"] / 1e6, name="Net (₹M)",
                    mode="lines+markers", line=dict(color=ACCENT, width=3))
    fig.add_scatter(x=monthly["month"], y=monthly["invoices"], name="Invoices", yaxis="y2",
                    mode="lines+markers", line=dict(color=GREEN, dash="dash"))
    fig.update_layout(title="Monthly Revenue — Gross vs Net", yaxis_title="₹ Millions",
                      yaxis2=dict(title="Invoices", overlaying="y", side="right"),
                      legend=dict(orientation="h", y=-0.2), height=430)
    st.plotly_chart(fig)
    if edge:
        cnt = monthly.set_index("month")["invoices"]
        st.caption("Lighter bars are the first and last month in the selected range and may be incomplete: "
                   + ", ".join(f"{m} ({int(cnt[m])} invoices)" for m in sorted(edge))
                   + ". Don't read them as real dips.")

    peak = monthly.loc[monthly["gross"].idxmax()]
    st.info(f"Peak month: **{peak['month']}** with {fmt_inr(peak['gross'])} gross revenue "
            f"({int(peak['invoices'])} invoices). This is a single year of data, so seasonal "
            "explanations (e.g. festive stocking) are hypotheses, not findings.")

    c1, c2 = st.columns(2)
    with c1:
        f2 = go.Figure()
        f2.add_scatter(x=monthly["month"], y=monthly["net"] / 1e6, stackgroup="one", name="Net",
                       line=dict(color=PRIMARY))
        f2.add_scatter(x=monthly["month"], y=monthly["tax"] / 1e6, stackgroup="one", name="Tax",
                       line=dict(color=ACCENT))
        f2.update_layout(title="Revenue Composition: Net vs Tax (₹M)", height=380)
        st.plotly_chart(f2)
    with c2:
        st.plotly_chart(px.histogram(inv, x=inv["gross_value"] / 1e5, nbins=20,
                                     labels={"x": "Invoice value (₹ Lakhs)"},
                                     title="Invoice Size Distribution",
                                     color_discrete_sequence=[PRIMARY]).update_layout(height=380))
    st.caption("Forecasting is intentionally not included: 13 months (with two partial edge months) is too little "
               "history to beat a simple average.")

# ───────────────────────── Buyers ─────────────────────────
with tabs[1]:
    top_n = st.slider("Top N buyers", 5, 30, 15)
    buyers = (df.groupby("buyer_name")
              .agg(invoices=("invoice_number", "nunique"), spend=("line_item_gross_worth", "sum"),
                   line_items=("line_item_number", "count"))
              .reset_index().sort_values("spend", ascending=False))
    buyers["cum_pct"] = buyers["spend"].cumsum() / buyers["spend"].sum() * 100
    buyers["rank"] = range(1, len(buyers) + 1)

    c1, c2 = st.columns(2)
    with c1:
        t = buyers.head(top_n).iloc[::-1]
        # One color: bar length already shows spend, so a gradient would only repeat it.
        st.plotly_chart(px.bar(t, x=t["spend"] / 1e6, y="buyer_name", orientation="h",
                               labels={"x": "Total spend (₹M)", "buyer_name": ""},
                               title=f"Top {top_n} Buyers by Spend",
                               color_discrete_sequence=[PRIMARY])
                        .update_layout(height=520))
    with c2:
        p = go.Figure()
        p.add_bar(x=buyers["rank"], y=buyers["spend"] / 1e6, name="Spend (₹M)", marker_color=PRIMARY)
        p.add_scatter(x=buyers["rank"], y=buyers["cum_pct"], name="Cumulative %", yaxis="y2",
                      line=dict(color=ALERT, width=3))
        p.add_hline(y=80, line_dash="dash", line_color="gray", yref="y2")
        p.update_layout(title="Buyer Concentration (Pareto)", xaxis_title="Buyer rank",
                        yaxis_title="₹M", yaxis2=dict(overlaying="y", side="right", range=[0, 105],
                                                      title="Cumulative %"),
                        height=520, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(p)
    n80 = int((buyers["cum_pct"] < 80).sum() + 1)
    st.info(f"**{n80}** of {len(buyers)} buyers account for 80% of revenue. "
            f"Top buyer: **{buyers.iloc[0]['buyer_name']}** "
            f"({fmt_inr(buyers.iloc[0]['spend'])}, {buyers.iloc[0]['spend']/buyers['spend'].sum()*100:.1f}% of total).")

    st.subheader("By state")
    buyers["region"] = buyers["buyer_name"].apply(get_region)
    unmapped = buyers[buyers["region"] == "Unmapped"]
    st.caption("State is guessed from the city name inside the buyer name (there is no state column), "
               "so treat it as approximate.")
    if len(unmapped):
        st.warning(f"{len(unmapped)} buyer(s) have no recognised city and are left out of the state charts: "
                   + ", ".join(unmapped["buyer_name"].head(8)) + ("…" if len(unmapped) > 8 else ""))
    state = (buyers[buyers["region"] != "Unmapped"].groupby("region")
             .agg(buyers=("buyer_name", "nunique"), invoices=("invoices", "sum"), spend=("spend", "sum"))
             .reset_index().sort_values("spend", ascending=False))
    if state.empty:
        st.info("No buyers could be mapped to a state.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            t = state.head(10).iloc[::-1]
            st.plotly_chart(px.bar(t, x=t["spend"] / 1e6, y="region", orientation="h",
                                   labels={"x": "₹M", "region": ""}, title="Top 10 States by Spend",
                                   color_discrete_sequence=[PRIMARY])
                            .update_layout(height=400))
        with c2:
            pie = state.head(6)[["region", "spend"]]
            rest = state.iloc[6:]["spend"].sum()
            if rest > 0:
                pie = pd.concat([pie, pd.DataFrame({"region": ["All other states"], "spend": [rest]})])
            st.plotly_chart(px.pie(pie, names="region", values="spend", hole=0.4,
                                   title="Revenue Share by State",
                                   color_discrete_sequence=px.colors.sequential.Blues_r)
                            .update_layout(height=400))

# ───────────────────────── Products ─────────────────────────
with tabs[2]:
    prod = (df.groupby("line_item_description")
            .agg(times_ordered=("invoice_number", "count"), qty=("line_item_quantity", "sum"),
                 avg_price=("line_item_net_price", "mean"), revenue=("line_item_gross_worth", "sum"))
            .reset_index().rename(columns={"line_item_description": "product"})
            .sort_values("revenue", ascending=False))
    prod["brand"] = prod["product"].apply(get_brand)
    brand_colors = color_map(prod["brand"])

    c1, c2 = st.columns(2)
    with c1:
        t = prod.head(15).iloc[::-1]
        # Color = brand (legend shown); bar length already shows revenue.
        st.plotly_chart(px.bar(t, x=t["revenue"] / 1e6, y="product", orientation="h",
                               color="brand", color_discrete_map=brand_colors,
                               labels={"x": "Revenue (₹M)", "product": "", "brand": "Brand"},
                               title="Top 15 Products by Revenue")
                        .update_layout(height=520))
    with c2:
        # Bubble size already encodes revenue, so color encodes brand instead.
        st.plotly_chart(px.scatter(prod, x="qty", y=prod["avg_price"] / 1e3, size="revenue",
                                   color="brand", color_discrete_map=brand_colors,
                                   hover_name="product",
                                   labels={"qty": "Total quantity", "y": "Avg unit price (₹K)",
                                           "brand": "Brand"},
                                   title="Price vs Volume (bubble = revenue, colour = brand)")
                        .update_layout(height=520))
    st.caption("Brand is guessed from the first word of the product description. Products are grouped by the "
               "exact description text, so the same item spelled two ways would appear as two products.")
    st.dataframe(prod.round(2), hide_index=True)

# ───────────────────────── Market Basket ─────────────────────────
with tabs[3]:
    baskets = df.groupby("invoice_number")["line_item_description"].apply(lambda s: sorted(set(s.dropna())))
    n = len(baskets)
    st.caption(f"Association rules mined over {n} invoices (each invoice = one basket). "
               "With this few baskets, a pair seen 3–5 times can look strong by chance — "
               "check the co-occurrence count before acting on a rule.")
    c1, c2, c3 = st.columns(3)
    min_sup = c1.slider("Min support", 0.01, 0.30, 0.05, 0.01)
    min_conf = c2.slider("Min confidence", 0.05, 1.0, 0.30, 0.05)
    min_cnt = c3.slider("Min co-occurrences", 2, 10, 3)

    item_c, pair_c = Counter(), Counter()
    for b in baskets:
        item_c.update(b)
        pair_c.update(combinations(b, 2))

    rows = []
    for (a, b), cnt in pair_c.items():
        sup = cnt / n
        if sup < min_sup or cnt < min_cnt:
            continue
        for ant, con in ((a, b), (b, a)):
            conf = cnt / item_c[ant]
            if conf >= min_conf:
                rows.append({"antecedent": ant, "consequent": con, "support": sup,
                             "confidence": conf, "lift": conf / (item_c[con] / n), "co_occurrence": cnt})
    rules = pd.DataFrame(rows)

    top_pairs = pd.DataFrame([{"pair": f"{a} + {b}", "freq": c} for (a, b), c in pair_c.most_common(10)])
    c1, c2 = st.columns(2)
    with c1:
        if not top_pairs.empty:
            st.plotly_chart(px.bar(top_pairs.iloc[::-1], x="freq", y="pair", orientation="h",
                                   title="Top 10 Product Pairs (all, ignoring thresholds)",
                                   labels={"freq": "Co-occurrences", "pair": ""},
                                   color_discrete_sequence=[PRIMARY]).update_layout(height=450))
        else:
            st.info("No product pairs (every invoice has a single product).")
    with c2:
        if rules.empty:
            st.info("No rules at these thresholds — lower support, confidence or co-occurrences.")
        else:
            st.plotly_chart(px.scatter(rules, x="support", y="confidence", color="lift",
                                       hover_data=["antecedent", "consequent", "co_occurrence"],
                                       color_continuous_scale="Blues",
                                       labels={"lift": "Lift (1 = chance level)"},
                                       title="Rules: Support vs Confidence (darker = higher lift)")
                            .update_layout(height=450))
    if not rules.empty:
        st.subheader(f"{len(rules)} rules (sorted by lift)")
        st.dataframe(rules.sort_values("lift", ascending=False).round(3), hide_index=True)

# ───────────────────────── Segments & CLV ─────────────────────────
with tabs[4]:
    seg_tab, clv_tab = st.tabs(["K-Means segments", "CLV (heuristic)"])

    with seg_tab:
        try:
            seg = load_table(T_SEGMENTS)
            require(seg, ["buyer_name", "recency_days", "frequency", "total_spend", "segment_name"], T_SEGMENTS)
            if sel_buyers:
                seg = seg[seg["buyer_name"].isin(sel_buyers)]
            if seg.empty:
                st.info("No segment rows for the selected buyers.")
            else:
                if seg["frequency"].nunique() == 1:
                    st.warning(f"Every buyer has the same purchase frequency ({seg['frequency'].iloc[0]:g} invoices), "
                               "so that feature carries no information. These segments are effectively driven by "
                               "recency and spend only, from two invoices per buyer. Treat them as a rough grouping.")
                summ = (seg.groupby("segment_name")
                        .agg(buyers=("buyer_name", "count"), avg_recency_days=("recency_days", "mean"),
                             avg_spend=("total_spend", "mean"), total_revenue=("total_spend", "sum"))
                        .reset_index())
                summ["revenue_share_%"] = summ["total_revenue"] / summ["total_revenue"].sum() * 100
                med_r, med_s = seg["recency_days"].median(), seg["total_spend"].median()
                rec_lbl = summ["avg_recency_days"].le(med_r).map({True: "recent", False: "lapsed"})
                spd_lbl = summ["avg_spend"].ge(med_s).map({True: "high spend", False: "low spend"})
                summ["profile"] = rec_lbl + " · " + spd_lbl
                seg_colors = color_map(seg["segment_name"])  # same segment, same color in every chart
                st.caption("Segment names come from a notebook heuristic (spend ÷ recency rank). Read the "
                           "**profile** column — a segment called 'Key Accounts' is not necessarily the one "
                           "with the most revenue.")
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.bar(summ, x="segment_name", y="buyers", color="segment_name",
                                       color_discrete_map=seg_colors,
                                       title="Segment sizes").update_layout(showlegend=False))
                c2.plotly_chart(px.bar(summ, x="segment_name", y=summ["total_revenue"] / 1e6,
                                       color="segment_name", color_discrete_map=seg_colors,
                                       labels={"y": "₹M"},
                                       title="Revenue by segment").update_layout(showlegend=False))
                st.plotly_chart(px.scatter(seg, x="recency_days", y=seg["total_spend"] / 1e5,
                                           color="segment_name", color_discrete_map=seg_colors,
                                           hover_name="buyer_name",
                                           labels={"y": "Total spend (₹L)", "segment_name": "Segment"},
                                           title="Buyers: recency vs spend"))
                st.dataframe(summ.round(1), hide_index=True)
                with st.expander("Buyer-level segments"):
                    st.dataframe(seg.sort_values("total_spend", ascending=False), hide_index=True)
        except Exception as e:
            st.error(f"Segments unavailable: {e}")

    with clv_tab:
        try:
            clv = load_table(T_CLV).rename(columns={
                "customer_id": "buyer_name", "prob_alive": "p_alive",
                "clv_segment": "clv_tier", "num_invoices": "frequency"})  # two notebook versions exist
            require(clv, ["buyer_name", "clv_6m", "p_alive"], T_CLV)
            if "clv_tier" in clv.columns:
                clv["clv_tier"] = (clv["clv_tier"].astype(str)
                                   .str.replace(" CLV", "", regex=False).str.replace(" Value", "", regex=False))
            else:
                clv["clv_tier"] = pd.qcut(clv["clv_6m"].rank(method="first"), 3, labels=["Low", "Medium", "High"]).astype(str)
            if sel_buyers:
                clv = clv[clv["buyer_name"].isin(sel_buyers)]
            if clv.empty:
                st.info("No CLV rows for the selected buyers.")
            else:
                st.warning("**Heuristic, not a validated forecast.** Each buyer has only two invoices, the "
                           "'probability alive' decay is a fixed formula rather than a fitted model, and the "
                           "predictions were never back-tested. Use it to rank buyers for follow-up, not to "
                           "budget revenue. Tiers are always thirds of the ranking by construction. "
                           "Recency is measured from the end of the data, not today.")
                a, b, c, d = st.columns(4)
                a.metric("Sum of estimates (6 mo)", fmt_inr(clv["clv_6m"].sum()))
                b.metric("Median per buyer", fmt_inr(clv["clv_6m"].median()))
                c.metric("Avg P(alive)", f"{clv['p_alive'].mean():.2f}")
                d.metric("Buyers with P<0.5", int((clv["p_alive"] < 0.5).sum()))
                tier_order = {"clv_tier": ["High", "Medium", "Low"]}
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.histogram(clv, x=clv["clv_6m"] / 1e5, nbins=15,
                                             labels={"x": "6-month estimate (₹L)"}, title="Estimate distribution",
                                             color_discrete_sequence=[PRIMARY]))
                c2.plotly_chart(px.scatter(clv, x="p_alive", y=clv["clv_6m"] / 1e5, color="clv_tier",
                                           color_discrete_map=TIER_COLORS, category_orders=tier_order,
                                           hover_name="buyer_name",
                                           labels={"y": "Estimate (₹L)", "clv_tier": "CLV tier"},
                                           title="P(alive) vs estimate"))
                t = clv.nlargest(15, "clv_6m").iloc[::-1]
                st.plotly_chart(px.bar(t, x=t["clv_6m"] / 1e5, y="buyer_name", orientation="h",
                                       color="clv_tier", color_discrete_map=TIER_COLORS,
                                       category_orders=tier_order,
                                       labels={"x": "Estimate (₹L)", "buyer_name": "", "clv_tier": "CLV tier"},
                                       title="Top 15 buyers by estimate").update_layout(height=520))
                st.dataframe(clv.sort_values("clv_6m", ascending=False).round(2), hide_index=True)
        except Exception as e:
            st.error(f"CLV unavailable: {e}")

# ───────────────────────── Anomalies ─────────────────────────
with tabs[5]:
    try:
        st.subheader("1 · Price deviation from each product's typical price")
        st.caption("Rule-based and easy to verify: compares each line's unit price with the median price of the "
                   "same product. A deviation is a lead to check against the source PDF, not proof of an error.")
        pp = df[["invoice_number", "buyer_name", "line_item_description", "line_item_quantity",
                 "line_item_net_price", "line_item_net_worth"]].copy()
        pp.columns = ["invoice_number", "buyer_name", "product", "quantity", "unit_price", "net_value"]
        pp = pp.dropna(subset=["unit_price"])
        med = pp.groupby("product")["unit_price"].transform("median")
        cnt_p = pp.groupby("product")["unit_price"].transform("count")
        pp["median_price"] = med
        pp["deviation_%"] = (pp["unit_price"] / med - 1) * 100
        pp = pp[cnt_p >= 3]
        thr = st.slider("Flag if price differs from product median by more than (%)", 1, 50, 10)
        flagged_p = pp[pp["deviation_%"].abs() > thr].sort_values("deviation_%", key=abs, ascending=False)
        st.metric("Lines flagged", f"{len(flagged_p)} of {len(pp)}")
        st.plotly_chart(px.histogram(pp, x="deviation_%", nbins=30,
                                     title="How far unit prices sit from their product median (%)",
                                     color_discrete_sequence=[PRIMARY]))
        st.dataframe(flagged_p.round(2), hide_index=True)

        st.subheader("2 · Isolation Forest outliers")
        an = load_table(T_ANOMALIES)
        require(an, ["invoice_number", "buyer_name", "product", "quantity", "unit_price", "net_value",
                     "gross_value", "anomaly_label", "anomaly_score"], T_ANOMALIES)
        if sel_buyers:
            an = an[an["buyer_name"].isin(sel_buyers)]
        an = an.assign(status=np.where(an["anomaly_label"] == -1, "Outlier", "Normal"))
        flagged = an[an["status"] == "Outlier"].sort_values("anomaly_score")
        if an.empty:
            st.info("No rows for the selected buyers.")
        else:
            top_decile = an["gross_value"].quantile(0.9)
            share_big = (flagged["gross_value"] >= top_decile).mean() * 100 if len(flagged) else 0
            st.warning("The model was told to flag about 5% of lines, so the flag rate says nothing about data "
                       f"quality. Here, {share_big:.0f}% of flagged lines are in the top 10% by value — mostly it "
                       "is picking out large orders, not mistakes. Use section 1 for price errors.")
            a, b = st.columns(2)
            a.metric("Line items scored", f"{len(an):,}")
            b.metric("Flagged as outliers", f"{len(flagged):,}")
            cmap = {"Normal": MUTED, "Outlier": ALERT}
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.scatter(an, x="quantity", y="unit_price", color="status",
                                       color_discrete_map=cmap, hover_data=["invoice_number", "product"],
                                       title="Quantity vs unit price"))
            c2.plotly_chart(px.scatter(an, x="net_value", y="gross_value", color="status",
                                       color_discrete_map=cmap, hover_data=["invoice_number", "product"],
                                       title="Net vs gross value"))
            st.dataframe(flagged[["invoice_number", "buyer_name", "product", "quantity", "unit_price",
                                  "net_value", "gross_value", "anomaly_score"]].round(2), hide_index=True)
    except Exception as e:
        st.error(f"Anomaly view unavailable: {e}")

# ───────────────────────── Data Quality ─────────────────────────
with tabs[6]:
    st.caption("Validation checks on the full `finance_invoices` table (ignores the sidebar filters).")
    q = raw
    checks = []

    def add(cat, name, ok, detail, warn=False):
        checks.append({"category": cat, "check": name,
                       "status": "PASS" if ok else ("WARN" if warn else "FAIL"), "details": detail})

    for col in ["invoice_number", "invoice_date", "buyer_name", "seller_name", "line_item_description",
                "line_item_quantity", "line_item_net_price", "line_item_net_worth",
                "line_item_gross_worth", "currency"]:
        s = q[col]
        bad = int(s.isna().sum())
        if not pd.api.types.is_numeric_dtype(s):
            bad += int((s.dropna().astype(str).str.strip() == "").sum())
        add("Completeness", f"{col} not null", bad == 0, f"{bad} null/empty of {len(q)} rows")

    m1 = int(((q["line_item_quantity"] * q["line_item_net_price"] - q["line_item_net_worth"]).abs() > 0.01).sum())
    add("Accuracy", "qty × net_price = net_worth", m1 == 0, f"{m1} mismatches of {len(q)} rows")
    m2 = int(((q["tax_value"] - q["line_item_net_worth"] * q["line_item_vat_percent"] / 100).abs() > 1.0).sum())
    add("Accuracy", "net_worth + tax = gross_worth", m2 == 0, f"{m2} mismatches of {len(q)} rows")

    tot = q.groupby("invoice_number").agg(cn=("line_item_net_worth", "sum"), cg=("line_item_gross_worth", "sum"),
                                          sn=("summary_net_worth", "max"), sg=("summary_gross_worth", "max"))
    mn, mg = int(((tot.cn - tot.sn).abs() > 1).sum()), int(((tot.cg - tot.sg).abs() > 1).sum())
    add("Consistency", "SUM(line items) net = invoice summary net", mn == 0, f"{mn} mismatches of {len(tot)} invoices")
    add("Consistency", "SUM(line items) gross = invoice summary gross", mg == 0, f"{mg} mismatches of {len(tot)} invoices")
    nv = q["line_item_vat_percent"].nunique()
    add("Consistency", "uniform VAT rate", nv == 1, f"{nv} distinct VAT rate(s)", warn=True)

    dup_inv = int((q.groupby("invoice_number")["path"].nunique() > 1).sum())
    add("Uniqueness", "each invoice number comes from one file", dup_inv == 0, f"{dup_inv} invoice(s) on multiple files")
    dup_li = int(q.duplicated(["path", "invoice_number", "line_item_number"]).sum())
    add("Uniqueness", "no duplicate line items", dup_li == 0, f"{dup_li} duplicate line item(s)", warn=True)

    add("Validity", "invoice_date parseable (DD/MM/YYYY)", n_bad_dates == 0, f"{n_bad_dates} unparseable")
    neg = int((q[["line_item_quantity", "line_item_net_price", "line_item_net_worth",
                  "line_item_gross_worth"]] < 0).sum().sum())
    add("Validity", "non-negative numeric values", neg == 0, f"{neg} negative value(s)")
    nc = q["currency"].nunique()
    add("Validity", "single currency", nc == 1, f"{nc} distinct currency value(s)", warn=True)
    zq = int((q["line_item_quantity"] == 0).sum())
    add("Validity", "no zero-quantity line items", zq == 0, f"{zq} zero-quantity row(s)", warn=True)

    dq = pd.DataFrame(checks)
    a, b, c = st.columns(3)
    a.metric("✅ PASS", int((dq.status == "PASS").sum()))
    b.metric("⚠️ WARN", int((dq.status == "WARN").sum()))
    c.metric("❌ FAIL", int((dq.status == "FAIL").sum()))
    _color = lambda v: {"PASS": "color:#2e7d32", "WARN": "color:#ef6c00", "FAIL": "color:#c62828"}.get(v, "")
    _sty = dq.style
    _sty = (_sty.map if hasattr(_sty, "map") else _sty.applymap)(_color, subset=["status"])  # pandas <2.1 fallback
    st.dataframe(_sty, hide_index=True)
    st.caption("These checks confirm the extracted numbers are internally consistent. They can't detect a value "
               "that was misread but still adds up; spot-check a few invoices against the source PDFs.")

# ───────────────────────── Invoices ─────────────────────────
with tabs[7]:
    show = inv.rename(columns={"line_items": "line_item_count", "total_qty": "total_quantity",
                               "tax_value": "tax_amount"})
    fig = px.scatter(show, x="line_item_count", y=show["gross_value"] / 1e5, hover_name="invoice_number",
                     labels={"y": "Gross value (₹L)", "line_item_count": "Line items"},
                     title="Invoice value vs line item count", color_discrete_sequence=[PRIMARY])
    if show["line_item_count"].nunique() > 1 and len(show) >= 3:
        slope, icpt = np.polyfit(show["line_item_count"], show["gross_value"] / 1e5, 1)
        xs = np.array([show["line_item_count"].min(), show["line_item_count"].max()])
        fig.add_scatter(x=xs, y=slope * xs + icpt, mode="lines", name="Linear fit",
                        line=dict(dash="dash", color="gray"))
        r = np.corrcoef(show["line_item_count"], show["gross_value"])[0, 1]
        st.caption(f"Correlation between line items and invoice value: {r:.2f}.")
    st.plotly_chart(fig)
    st.dataframe(show.sort_values("gross_value", ascending=False), hide_index=True)
