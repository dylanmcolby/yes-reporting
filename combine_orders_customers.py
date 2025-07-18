# combine_orders_customers.py
# RUN COMMAND: python3 -m streamlit run combine_orders_customers.py
#
# Streamlit app that:
# 1. Lets you upload an **Orders** CSV and a **Customers** CSV.
# 2. Flattens multi-line-item orders into a single "Line items" column.
# 3. Appends the matching customer record (by e-mail, case-insensitive) to each
#    order row.
# 4. Provides three downloadable report types:
#       • Combined Order + Customer CSV  (column-picker)
#       • Average Lifetime Value (LTV)   (existing)
#       • Purchases report               (NEW)

from __future__ import annotations

import re
import uuid
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ───────────────────────── helpers ──────────────────────────────────────────
_money_re = re.compile(r"[^\d.\-]")


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


def _money_to_float(val) -> float:
    try:
        return float(_money_re.sub("", str(val) or "0"))
    except ValueError:
        return 0.0


def _parse_date_ranges(text: str) -> List[Tuple[date, date]]:
    """Return list[(start, end)] parsed from 'YYYY-MM-DD to YYYY-MM-DD' chunks."""
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
    """Parse Shopify 'Created at' (once) into df['_created_at_dt'] as date."""
    if "_created_at_dt" in df.columns or "Created at" not in df.columns:
        return
    dt = pd.to_datetime(df["Created at"].astype(str).str.strip(), utc=True, errors="coerce")
    if not dt.isna().all():
        df["_created_at_dt"] = dt.dt.date


def _emails_with_orders(
    df: pd.DataFrame,
    date_ranges: List[Tuple[date, date]],
    texts: List[str],
) -> set[str]:
    """Return set of customer emails that have ≥1 order matching date + text."""
    if not date_ranges and not texts:
        return set()

    _attach_created_date(df)
    mask = True

    if date_ranges and "_created_at_dt" in df.columns:
        date_mask = False
        for start, end in date_ranges:
            date_mask |= df["_created_at_dt"].between(start, end, inclusive="both")
        mask &= date_mask

    if texts and "Line items" in df.columns:
        lc_texts = [t.lower() for t in texts]
        mask &= df["Line items"].str.lower().fillna("").apply(lambda x: any(t in x for t in lc_texts))

    return set(df.loc[mask, "Email"])


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

    orders["Line items"] = orders.apply(_format_line, axis=1, args=(item_name_col, sku_col, price_col))

    base_cols = [c for c in orders.columns if c not in {item_name_col, sku_col, price_col, "Line items"}]
    base_df = orders[base_cols].drop_duplicates(subset=[order_id_col])
    li_df = (
        orders[[order_id_col, "Line items"]]
        .groupby(order_id_col, as_index=False)["Line items"].agg(", ".join)
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

# ─────────────────────── LTV report generator ───────────────────────────────
def generate_ltv_report(df_orders: pd.DataFrame, filters: Dict) -> Tuple[pd.DataFrame, Dict]:
    df = df_orders.copy()

    # order‑level pre‑filter (date range)
    if filters.get("order_date_ranges"):
        _attach_created_date(df)
        if "_created_at_dt" in df.columns:
            mask = False
            for s, e in filters["order_date_ranges"]:
                mask |= df["_created_at_dt"].between(s, e, inclusive="both")
            df = df[mask]

    # customer include / exclude (date + li text)
    inc_emails = _emails_with_orders(df, filters["include_ranges"], filters["inc_texts"])
    exc_emails = _emails_with_orders(df, filters["exclude_ranges"], filters["exc_texts"])
    if inc_emails:
        df = df[df["Email"].isin(inc_emails)]
    if exc_emails:
        df = df[~df["Email"].isin(exc_emails)]

    # tag filters
    tag_col = "Tags_cust" if "Tags_cust" in df.columns else "Tags"
    if tag_col in df.columns:
        inc_tags = [t.lower() for t in filters["tag_includes"] if t]
        exc_tags = [t.lower() for t in filters["tag_excludes"] if t]
        if inc_tags:
            df = df[
                df[tag_col].str.lower().fillna("").apply(lambda x: any(t in x for t in inc_tags))
            ]
        if exc_tags:
            df = df[
                ~df[tag_col].str.lower().fillna("").apply(lambda x: any(t in x for t in exc_tags))
            ]

    # Freeze the valid customer pool after customer-level filters but before order-level filters
    valid_customers = df["Email"].nunique()

    # line‑item exclusion
    li_exc = [t.lower() for t in filters["lineitem_excludes"] if t]
    if li_exc and "Line items" in df.columns:
        df = df[
            ~df["Line items"].str.lower().fillna("").apply(lambda x: any(t in x for t in li_exc))
        ]

    # money helpers
    df["_Subtotal_f"] = df["Subtotal"].apply(_money_to_float)
    df["_Total_f"] = df["Total"].apply(_money_to_float)
    df["_Discount_f"] = df["Discount Amount"].apply(_money_to_float)
    df["_Gross_f"] = df["_Total_f"] + df["_Discount_f"]

    if filters.get("exclude_zero_orders"):
        df = df[(df["_Total_f"] > 0) & (df["_Subtotal_f"] > 0)]

    if df.empty:
        raise ValueError("No data left after applying filters.")

    # aggregate
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

    purchase_frequency = report["Total Number of Orders"].sum() / valid_customers

    report["Subtotal LTV"] = report["Subtotal AOV"] * purchase_frequency
    report["Order LTV"] = report["Order Total AOV"] * purchase_frequency
    report["Gross LTV"] = report["Gross Total AOV"] * purchase_frequency

    # Calculate customer retention percentage
    final_customers = report["Customer Email"].nunique()
    customer_retention_pct = (final_customers / valid_customers * 100) if valid_customers else 0.0

    summary: Dict[str, float] = report.mean(numeric_only=True).to_dict()
    summary["Purchase Frequency"] = purchase_frequency
    summary["Subtotal LTV"] = summary["Subtotal AOV"] * purchase_frequency
    summary["Order LTV"] = summary["Order Total AOV"] * purchase_frequency
    summary["Gross LTV"] = summary["Gross Total AOV"] * purchase_frequency
    summary["Valid Customers"] = valid_customers
    summary["Final Customers"] = final_customers
    summary["% Customers Retained"] = customer_retention_pct

    return report, summary


# ─────────────────────────── UI ─────────────────────────────────────────────
st.set_page_config(page_title="YES Society Order Reports", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1100px; padding-left:2rem; padding-right:2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍷 YES Society Order Reports")

st.markdown(
    "Upload your **Shopify Orders** and **Customers** CSVs, then build any of the reports below."
)

# File Upload Section
st.markdown("### 📁 Upload Data Files")
col1, col2 = st.columns(2)

with col1:
    orders_file = st.file_uploader(
        "Orders CSV", 
        type="csv",
        help="Upload your Shopify orders export CSV file. Default: YS Full Orders.csv"
    )

with col2:
    customers_file = st.file_uploader(
        "Customers CSV", 
        type="csv",
        help="Upload your Shopify customers export CSV file. Default: Customers Full Export.csv"
    )

if orders_file and customers_file:
    # keep combined DF
    if "full_combined_df" not in st.session_state:
        orders_file.seek(0)
        customers_file.seek(0)
        st.session_state.full_combined_df = combine(orders_file, customers_file, [])
    full_df: pd.DataFrame = st.session_state.full_combined_df

    # ───────── Combined CSV ────────────────────────────────────────────────
    st.markdown("## 📋 Combined Order+Customers CSV")
    st.markdown("""
    **What this report does:**
    Combines your orders and customers data into a single CSV file. Each order row includes the matching customer information, and multi-line orders are flattened into a single "Line items" column. Perfect for data analysis in Excel or other tools.
    """)
    with st.expander("Generate combined CSV", expanded=False):
        # Get actual column names from the combined dataframe
        all_cols = full_df.columns.tolist()
        
        # Create display names for columns (rename Tags_cust to Customer Tags)
        display_cols = []
        for col in all_cols:
            if col == "Tags_cust":
                display_cols.append("Customer Tags")
            else:
                display_cols.append(col)

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
            "Customer Tags",
        }

        sel_display = st.multiselect(
            "Columns to include in combined CSV",
            display_cols,
            default=[c for c in display_cols if c in default_cols],
            help="Select which columns to include in the combined CSV download."
        )
        
        # Map display names back to actual column names
        sel = []
        for display_col in sel_display:
            if display_col == "Customer Tags":
                sel.append("Tags_cust")
            else:
                sel.append(display_col)

        if st.button("Download combined CSV"):
            # Reset file positions for combining
            if hasattr(orders_file, 'seek'):
                orders_file.seek(0)
            if hasattr(customers_file, 'seek'):
                customers_file.seek(0)
            combined = combine(orders_file, customers_file, sel)
            st.download_button(
                f"⬇️ Download {len(combined):,}‑row CSV",
                combined.to_csv(index=False).encode(),
                "combined_orders_customers.csv",
                "text/csv",
            )

    # ───────── LTV reports ────────────────────────────────────────────────
    st.markdown("## 📈 Average Lifetime Value (LTV) Reports")
    st.markdown("""
    **What this report does:**
    Calculates the average lifetime value (LTV) of your customers based on their order history. You can filter by order date, customer tags, and line-item text. Useful for understanding customer value and segmentation.
    """)
    if "ltv_reports" not in st.session_state:
        st.session_state.ltv_reports = {}

    with st.expander("➕ Create LTV report", expanded=False):
        with st.form("ltv_form"):
            st.markdown("#### Report Name")
            name = st.text_input("Name for this LTV report", "LTV Report")
            st.markdown(":grey[Give your report a descriptive name.]")

            st.markdown("---")
            st.markdown("#### Order Date Filters")
            st.markdown("""
            **Order date ranges to include:**  
            Enter one or more date ranges in the format YYYY-MM-DD to YYYY-MM-DD, separated by semicolons.<br>
            Example: `2023-01-01 to 2023-12-31;2024-01-01 to 2024-06-30`
            """, unsafe_allow_html=True)
            order_date_raw = st.text_input("Order date ranges to include", "")

            st.markdown("---")
            st.markdown("#### Customer Filters")
            st.markdown("""
            **Customer date filters:**  
            - *Include customers* who placed orders in these date ranges:
            """)
            inc_raw = st.text_input("Customer include date ranges", "")
            st.markdown(":grey[Same format as above. Leave blank to include all.]")
            st.markdown("*Exclude customers* who placed orders in these date ranges:")
            exc_raw = st.text_input("Customer exclude date ranges", "")
            st.markdown(":grey[Same format as above. Leave blank to exclude none.]")

            st.markdown("**Customer line-item filters:**  ")
            st.markdown("Only include customers who purchased items containing these keywords (comma-separated):")
            inc_texts_raw = st.text_input("Customer include line-item keywords", "")
            st.markdown(":grey[Example: Chardonnay, Pinot Noir]")
            st.markdown("Exclude customers who purchased items containing these keywords (comma-separated):")
            exc_texts_raw = st.text_input("Customer exclude line-item keywords", "")
            st.markdown(":grey[Example: Gift Card, Membership]")

            st.markdown("**Customer tag filters:**  ")
            st.markdown("Only include customers with these tags (comma-separated):")
            tag_inc_raw = st.text_input("Customer tags to include", "")
            st.markdown(":grey[Example: VIP, Club]")
            st.markdown("Exclude customers with these tags (comma-separated):")
            tag_exc_raw = st.text_input("Customer tags to exclude", "")
            st.markdown(":grey[Example: Wholesale]")

            st.markdown("---")
            st.markdown("#### Order Line-Item Exclusions")
            st.markdown("Exclude orders containing these keywords in any line item (comma-separated):")
            li_exc_raw = st.text_input("Order line-item keywords to exclude", "membership, bottle box")
            st.markdown(":grey[Example: Membership, Bottle Box]")

            excl_zero = st.checkbox(
                "Exclude $0 orders",
                value=True,
                help=None
            )
            st.markdown(":grey[Exclude orders where the total or subtotal is $0.]")

            if st.form_submit_button("Add LTV report"):
                try:
                    filters = {
                        "order_date_ranges": _parse_date_ranges(order_date_raw),
                        "include_ranges": _parse_date_ranges(inc_raw),
                        "exclude_ranges": _parse_date_ranges(exc_raw),
                        "inc_texts": [t.strip() for t in inc_texts_raw.split(",") if t.strip()],
                        "exc_texts": [t.strip() for t in exc_texts_raw.split(",") if t.strip()],
                        "tag_includes": [t.strip() for t in tag_inc_raw.split(",") if t.strip()],
                        "tag_excludes": [t.strip() for t in tag_exc_raw.split(",") if t.strip()],
                        "lineitem_excludes": [t.strip() for t in li_exc_raw.split(",") if t.strip()],
                        "exclude_zero_orders": excl_zero,
                    }
                    df_rep, summ = generate_ltv_report(full_df, filters)
                    rid = uuid.uuid4().hex[:8]
                    st.session_state.ltv_reports[rid] = {"df": df_rep, "summary": summ, "name": name}
                    st.success(f"Added **{name}**")
                except Exception as e:
                    st.error(e)

    # Collect delete button presses for LTV reports
    ltv_to_delete = []
    for rid, rep in list(st.session_state.ltv_reports.items()):
        with st.expander(f"📄 {rep['name']}"):
            st.dataframe(rep["df"])
            st.download_button(
                "⬇️ Download CSV",
                rep["df"].to_csv(index=False).encode(),
                f"{rep['name'].replace(' ', '_').lower()}.csv",
                "text/csv",
                key=f"dl_ltv_{rid}",
            )

            s = rep["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Purchase Freq", f"{s['Purchase Frequency']:.3f}")
            c2.metric("Avg Subt AOV", f"${s['Subtotal AOV']:.2f}")
            c3.metric("Avg Ord AOV", f"${s['Order Total AOV']:.2f}")
            c4.metric("Avg Gross AOV", f"${s['Gross Total AOV']:.2f}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Subtotal LTV", f"${s['Subtotal LTV']:.2f}")
            d2.metric("Order LTV", f"${s['Order LTV']:.2f}")
            d3.metric("Gross LTV", f"${s['Gross LTV']:.2f}")

            # Customer retention metrics
            e1, e2, e3 = st.columns(3)
            e1.metric("Valid Customers", f"{s['Valid Customers']:,}")
            e2.metric("Valid Customers with Valid Orders", f"{s['Final Customers']:,}")
            e3.metric("% Valid Customers with Valid Orders", f"{s['% Customers Retained']:.1f}%")

            if st.button("🗑️ Delete", key=f"rm_ltv_{rid}"):
                ltv_to_delete.append(rid)

    # After rendering all expanders, pop the flagged reports in one batch
    rerun_needed = False
    for rid in ltv_to_delete:
        st.session_state.ltv_reports.pop(rid, None)
        rerun_needed = True
    if rerun_needed:
        st.experimental_rerun()
