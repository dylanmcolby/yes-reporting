# combine_orders_customers.py
# RUN COMMAND:  python3 -m streamlit run combine_orders_customers.py
#
# Streamlit app that:
# 1. Lets you upload an **Orders** CSV and a **Customers** CSV.
# 2. Flattens multi‑line‑item orders into a single row with a
#    "Line items" column:
#       Line item name (SKU - $00.00), Line item name 2 (SKU - $00.00)
# 3. Appends the matching customer record (by e‑mail, case‑insensitive) to each
#    order row.
# 4. Provides multiple downloadable reports:
#       • **Combined Order + Customer CSV** (column‑picker)  
#       • **Average Lifetime Value (LTV)** reports with flexible customer filters.

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st


# ───────────────────────── helpers ──────────────────────────────────────────
def _find_col(df: pd.DataFrame, *candidates: str) -> str:
    for cand in candidates:
        for col in df.columns:
            if cand in col.lower():
                return col
    raise ValueError(f"None of {candidates} found in {df.columns.tolist()}")


def _format_line(row, name_col: str, sku_col: str, price_col: str) -> str:
    name, sku, price = map(str.strip, (row[name_col], row[sku_col], row[price_col]))
    price = price if price.startswith("$") else f"${price}"
    return f"{name} ({sku} - {price})"


def _clean_email(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower()


_money_re = re.compile(r"[^\d.\-]")


def _money_to_float(val) -> float:
    try:
        return float(_money_re.sub("", str(val) or "0"))
    except ValueError:
        return 0.0


# ───────────────────────── CSV combine ──────────────────────────────────────
def combine(
    orders_csv: BytesIO,
    customers_csv: BytesIO,
    selected_columns: List[str] | None,
) -> pd.DataFrame:
    orders = pd.read_csv(orders_csv, dtype=str).fillna("")
    customers = pd.read_csv(customers_csv, dtype=str).fillna("")

    order_id_col = _find_col(orders, "name")
    email_col = _find_col(orders, "email")
    item_name_col = _find_col(orders, "lineitem name", "line item name")
    sku_col = _find_col(orders, "lineitem sku", "line item sku")
    price_col = _find_col(orders, "lineitem price", "line item price")

    cust_email_col = _find_col(customers, "email")
    orders["_email_key"] = _clean_email(orders[email_col])
    customers["_email_key"] = _clean_email(customers[cust_email_col])

    orders["Line items"] = orders.apply(
        _format_line, axis=1, args=(item_name_col, sku_col, price_col)
    )

    base_cols = [
        c for c in orders.columns if c not in {item_name_col, sku_col, price_col, "Line items"}
    ]
    base_df = orders[base_cols].drop_duplicates(subset=[order_id_col])
    li_df = (
        orders[[order_id_col, "Line items"]]
        .groupby(order_id_col, as_index=False)["Line items"]
        .agg(", ".join)
    )
    flat_orders = base_df.merge(li_df, on=order_id_col, how="left")

    merged = (
        flat_orders.merge(customers, on="_email_key", how="left", suffixes=("", "_cust"))
        .drop(columns="_email_key")
        .copy()
    )

    if selected_columns:
        merged = merged[[c for c in selected_columns if c in merged.columns]]

    return merged


# ─────────────────────── LTV helpers / filters ──────────────────────────────
def _parse_date_ranges(text: str) -> List[Tuple[date, date]]:
    ranges: List[Tuple[date, date]] = []
    for chunk in (text or "").split(";"):
        m = re.match(r"\s*(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})\s*", chunk)
        if not m:
            continue
        try:
            start = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            end = datetime.strptime(m.group(2), "%Y-%m-%d").date()
            if start <= end:
                ranges.append((start, end))
        except ValueError:
            pass
    return ranges


def _attach_created_date(df: pd.DataFrame) -> None:
    if "_created_at_dt" in df.columns or "Created at" not in df.columns:
        return
    dt = pd.to_datetime(df["Created at"].astype(str).str.strip(), utc=True, errors="coerce")
    if not dt.isna().all():
        df["_created_at_dt"] = dt.dt.date


def _emails_in_ranges(df: pd.DataFrame, ranges: List[Tuple[date, date]]) -> set[str]:
    if not ranges or "Created at" not in df.columns:
        return set()
    _attach_created_date(df)
    if "_created_at_dt" not in df.columns:
        st.warning("Could not parse 'Created at' dates. Date filters skipped.")
        return set()
    mask = False
    for start, end in ranges:
        mask |= df["_created_at_dt"].between(start, end, inclusive="both")
    return set(df.loc[mask, "Email"])


# ─────────────────────── LTV report builder ─────────────────────────────────
def generate_ltv_report(df_orders: pd.DataFrame, filters: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = df_orders.copy()

    # ― order date filter ―
    if filters.get("order_date_ranges"):
        _attach_created_date(df)
        if "_created_at_dt" not in df.columns:
            st.warning("Could not parse 'Created at' dates. Order date filter skipped.")
        else:
            mask = False
            for start, end in filters["order_date_ranges"]:
                mask |= df["_created_at_dt"].between(start, end, inclusive="both")
            df = df[mask]

    # ― customer date filters ―
    inc_emails = _emails_in_ranges(df, filters["include_ranges"])
    exc_emails = _emails_in_ranges(df, filters["exclude_ranges"])
    if inc_emails:
        df = df[df["Email"].isin(inc_emails)]
    if exc_emails:
        df = df[~df["Email"].isin(exc_emails)]

    # ― tag filters ―
    tag_col = "Tags_cust" if "Tags_cust" in df.columns else "Tags"
    if tag_col in df.columns:
        tin = [t.lower() for t in filters["tag_includes"] if t]
        tex = [t.lower() for t in filters["tag_excludes"] if t]
        if tin:
            df = df[
                df[tag_col]
                .str.lower()
                .fillna("")
                .apply(lambda x: any(tok in x for tok in tin))
            ]
        if tex:
            df = df[
                ~df[tag_col]
                .str.lower()
                .fillna("")
                .apply(lambda x: any(tok in x for tok in tex))
            ]

    # ― line‑item exclusion ―
    li_exc = [t.lower() for t in filters["lineitem_excludes"] if t]
    if li_exc and "Line items" in df.columns:
        bad = df["Line items"].str.lower().fillna("").apply(lambda x: any(t in x for t in li_exc))
        df = df[~bad]

    # ― numeric helpers (needed before $0 filter) ―
    df["_Subtotal_f"] = df["Subtotal"].apply(_money_to_float)
    df["_Total_f"] = df["Total"].apply(_money_to_float)
    df["_Discount_f"] = df["Discount Amount"].apply(_money_to_float)
    df["_Gross_f"] = df["_Total_f"] + df["_Discount_f"]

    # ― exclude $0 orders if requested ―
    if filters.get("exclude_zero_orders"):
        df = df[(df["_Total_f"] > 0) & (df["_Subtotal_f"] > 0)]

    if df.empty:
        raise ValueError("No data left after applying filters.")

    # ― aggregate per customer ―
    grp = df.groupby("Email", dropna=False)
    report = pd.DataFrame(
        {
            "Customer Email": grp.size().index,
            "Total Number of Orders": grp.size().values,
            "Subtotal Total Spend": grp["_Subtotal_f"].sum().values,
            "Order Total Spend": grp["_Total_f"].sum().values,
            "Gross Total Spend": grp["_Gross_f"].sum().values,
        }
    )
    report["Subtotal AOV"] = report["Subtotal Total Spend"] / report["Total Number of Orders"]
    report["Order Total AOV"] = report["Order Total Spend"] / report["Total Number of Orders"]
    report["Gross Total AOV"] = report["Gross Total Spend"] / report["Total Number of Orders"]

    # ― overall purchase‑frequency (orders per customer in period) ―
    purchase_frequency = report["Total Number of Orders"].sum() / len(report)

    # add LTV columns per customer
    report["Subtotal LTV"] = report["Subtotal AOV"] * report["Total Number of Orders"]
    report["Order LTV"] = report["Order Total AOV"] * report["Total Number of Orders"]
    report["Gross LTV"] = report["Gross Total AOV"] * report["Total Number of Orders"]

    # ― summary metrics (means of numeric cols) ―
    summary: Dict[str, float] = report.mean(numeric_only=True).to_dict()
    summary["Purchase Frequency"] = purchase_frequency
    summary["Subtotal LTV"] = summary["Subtotal AOV"] * purchase_frequency
    summary["Order LTV"] = summary["Order Total AOV"] * purchase_frequency
    summary["Gross LTV"] = summary["Gross Total AOV"] * purchase_frequency

    return report, summary


# ─────────────────────────── UI ─────────────────────────────────────────────
st.set_page_config(page_title="YES Society Order Reports", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛠️ Order + Customer CSV Combiner")

st.markdown(
    "Upload your **Shopify Orders** and **Customers** CSVs.  \n\n"
    "• **Combined CSV** – merge files & choose columns.  \n"
    "• **LTV reports** – analyse customer value with flexible filters."
)

orders_file = st.file_uploader("Orders CSV", type="csv")
customers_file = st.file_uploader("Customers CSV", type="csv")

if orders_file and customers_file:
    # keep un‑filtered combined DF in session
    if "full_combined_df" not in st.session_state:
        orders_file.seek(0)
        customers_file.seek(0)
        st.session_state.full_combined_df = combine(orders_file, customers_file, [])

    full_df: pd.DataFrame = st.session_state.full_combined_df

    # ────────── Part A – Combined CSV ──────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Combined CSV")

    try:
        orders_file.seek(0)
        customers_file.seek(0)
        orders_sample = pd.read_csv(orders_file, nrows=1, dtype=str)
        customers_sample = pd.read_csv(customers_file, nrows=1, dtype=str)
        all_cols = list(
            dict.fromkeys(orders_sample.columns.tolist() + customers_sample.columns.tolist() + ["Line items"])
        )
    except Exception as e:
        st.error(f"Could not inspect column names: {e}")
        all_cols = []

    default_cols = {
        "Name",
        "Email",
        "Created at",
        "Subtotal",
        "Total",
        "Discount Amount",
        "Tags",
        "Line items",
        "Customer ID",
        "First Name",
        "Last Name",
        "Total Spent",
        "Total Orders",
        "Tags_cust",
    }

    if all_cols:
        show_picker = st.checkbox("Choose columns manually", value=True)
        if show_picker:
            sel = st.multiselect("Columns to include", all_cols, default=[c for c in all_cols if c in default_cols])
        else:
            sel = [c for c in all_cols if c in default_cols]
            st.info("Using default column set.")

        if st.button("Generate combined CSV"):
            orders_file.seek(0)
            customers_file.seek(0)
            combined_df = combine(orders_file, customers_file, sel)
            st.download_button(
                f"⬇️ Download {len(combined_df):,}‑row CSV",
                combined_df.to_csv(index=False).encode(),
                "combined_orders_customers.csv",
                "text/csv",
            )
            st.success("Combined CSV ready!")

    # ────────── Part B – LTV reports ───────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Average Lifetime Value (LTV) Reports")

    if "ltv_reports" not in st.session_state:
        st.session_state.ltv_reports = {}

    with st.expander("➕ Create a new LTV report", expanded=False):
        with st.form("ltv_form"):
            name = st.text_input("Report name", "LTV Report")

            st.markdown("**Order date filter** (semicolon‑separated `YYYY‑MM‑DD to YYYY‑MM‑DD`)")
            order_date_raw = st.text_input("Only include orders between")
            
            st.markdown("**Customer date filters** (semicolon‑separated `YYYY‑MM‑DD to YYYY‑MM‑DD`)")
            inc_raw = st.text_input("Include customers with ≥1 order between")
            exc_raw = st.text_input("Exclude customers with any order between")

            st.markdown("**Customer tag filters** (comma‑separated)")
            tag_inc_raw = st.text_input("Tags **include**")
            tag_exc_raw = st.text_input("Tags **exclude**")

            st.markdown("**Line‑item exclusion**")
            li_exc_raw = st.text_input("Exclude orders containing", "membership, bottle box")

            excl_zero = st.checkbox("Exclude $0 orders", value=True)

            if st.form_submit_button("Add report"):
                try:
                    filters = {
                        "order_date_ranges": _parse_date_ranges(order_date_raw),
                        "include_ranges": _parse_date_ranges(inc_raw),
                        "exclude_ranges": _parse_date_ranges(exc_raw),
                        "tag_includes": [t.strip() for t in tag_inc_raw.split(",") if t.strip()],
                        "tag_excludes": [t.strip() for t in tag_exc_raw.split(",") if t.strip()],
                        "lineitem_excludes": [t.strip() for t in li_exc_raw.split(",") if t.strip()],
                        "exclude_zero_orders": excl_zero,
                    }
                    rep_df, summary = generate_ltv_report(full_df, filters)
                    rep_id = uuid.uuid4().hex[:8]
                    st.session_state.ltv_reports[rep_id] = {"df": rep_df, "summary": summary, "name": name}
                    st.success(f"Report **{name}** added.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    # ───────── display reports ─────────────────────────────────────────────
    for rid, r in list(st.session_state.ltv_reports.items()):
        col1, col2 = st.columns([8, 1])
        with col1:
            with st.expander(f"📄 {r['name']}"):
                st.dataframe(r["df"])
                st.download_button(
                    "⬇️ Download CSV",
                    r["df"].to_csv(index=False).encode(),
                    f"{r['name'].replace(' ', '_').lower()}.csv",
                    "text/csv",
                    key=f"dl_{rid}",
                )

                s = r["summary"]
                # ─ metrics row 1 ─
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Purchase Frequency", f"{s['Purchase Frequency']:.3f}")
                c1.caption("Orders per customer in selected period")
                c2.metric("Avg Subtotal AOV", f"${s['Subtotal AOV']:.2f}")
                c2.caption("Pre‑discount (excl. tax+shipping) avg order value")
                c3.metric("Avg Order AOV", f"${s['Order Total AOV']:.2f}")
                c3.caption("Post‑discount order value")
                c4.metric("Avg Gross AOV", f"${s['Gross Total AOV']:.2f}")
                c4.caption("Order total excluding discounts")

                # ─ metrics row 2 ─
                d1, d2, d3 = st.columns(3)
                d1.metric("Subtotal LTV", f"${s['Subtotal LTV']:.2f}")
                d1.caption("Customer value using subtotal AOV")
                d2.metric("Order LTV", f"${s['Order LTV']:.2f}")
                d2.caption("Customer value using order AOV")
                d3.metric("Gross LTV", f"${s['Gross LTV']:.2f}")
                d3.caption("Customer value using gross AOV")
        with col2:
            if st.button("🗑️", key=f"rm_{rid}"):
                st.session_state.ltv_reports.pop(rid)
                st.experimental_rerun()
