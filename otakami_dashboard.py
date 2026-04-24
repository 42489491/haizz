from pathlib import Path

_streamlit_dir = Path(__file__).parent / ".streamlit"
_streamlit_dir.mkdir(exist_ok=True)
_config = _streamlit_dir / "config.toml"
if not _config.exists():
    _config.write_text(
        '[theme]\nprimaryColor="#e8734a"\n'
        'backgroundColor="#ffffff"\n'
        'secondaryBackgroundColor="#faf6f1"\n'
        'textColor="#2e3440"\n'
    )

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from io import BytesIO
from urllib.request import urlopen
import os, re, warnings

warnings.filterwarnings("ignore")

# Optional: put direct .xlsx URLs here (SharePoint direct download links).
# If a URL is provided, app will try URL first, then fallback to local file.
REMOTE_ORDER_URLS = {
    "T10": "",
    "T11": "",
    "T12": "",
    "T1_2026": "",
    "T2_2026": "",
    "T3_2026": "",
}

REMOTE_STATS_URLS = {
    "T10": "",
    "T11": "",
    "T12": "",
    "T1_2026": "",
    "T2_2026": "",
    "T3_2026": "",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OTAKAMI — Shopee Business Analytics",
    page_icon="O",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAL = {
    "pri": "#e8734a", "sec": "#2c5f7c", "acc": "#3d8eb9",
    "ok": "#4a9c6d", "warn": "#d4a843", "bad": "#c0504d",
    "muted": "#7b8794", "txt": "#2e3440",
    "orange": "#e8734a", "teal": "#2c8c99", "purple": "#7c5295",
    "pink": "#d4577a", "lime": "#7cb342",
}
TRAFFIC_COLORS = {
    "Product Card": "#2c5f7c",
    "Livestream": "#e8734a",
    "Video": "#4a9c6d",
    "Affiliate": "#d4a843",
    "Shopee Ads": "#7c5295",
    "Other": "#7b8794",
}

st.html("""<style>
section[data-testid="stSidebar"]{background:#faf6f1;font-size:13px!important}
section[data-testid="stSidebar"] label{font-size:13px!important}
section[data-testid="stSidebar"] .stMarkdown p{font-size:13px!important}
section[data-testid="stSidebar"] .stMarkdown h3{font-size:1rem!important}
.main .block-container{padding-top:1.2rem;max-width:1360px}
html, body, [class*="css"]{font-size:15px!important;line-height:1.6!important}
div[data-testid="stSlider"] [role="slider"]{background:#e8734a!important}
div[data-testid="stSlider"] [data-testid="stThumbValue"]{color:#e8734a!important}
div[data-testid="stSlider"] div[role="progressbar"]{background:#e8734a!important}
h1{color:#e8734a!important;font-weight:700!important;font-size:1.9rem!important}
h3{color:#2c5f7c!important;font-weight:700!important;font-size:1.25rem!important}
.stTabs [data-baseweb="tab-list"]{gap:6px}
.stTabs [data-baseweb="tab"]{background:#fff;border-radius:6px 6px 0 0;border:1px solid #e2e8f0;border-bottom:none;padding:8px 20px;font-weight:600;font-size:.92rem}
.stTabs [aria-selected="true"]{background:#e8734a!important;color:#fff!important}
.rq-box{background:linear-gradient(135deg,#fef5f0,#faf6f1);border-left:5px solid #e8734a;border-radius:6px;padding:18px 22px;margin:10px 0 18px 0}
.rq-box h2{color:#e8734a;font-size:1.3rem;font-weight:700;margin:0 0 6px 0}
.rq-box .rq-question{color:#2e3440;font-size:1.02rem;font-style:italic;margin:0}
.concept-tip{background:#f8f9fa;border:1px solid #e2e8f0;border-radius:6px;padding:12px 16px;margin:8px 0 14px 0;font-size:.93rem;color:#4a5568;line-height:1.55}
.concept-tip b{color:#e8734a}
div[data-testid="stMarkdownContainer"] p{font-size:.96rem!important;line-height:1.65!important}
div[data-testid="stMarkdownContainer"] li{font-size:.95rem!important;line-height:1.6!important}
</style>""")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _parse_vnd(val):
    """Convert Vietnamese formatted numbers like '36.776.556' to float."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("%", "")
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return np.nan
    dot_count = s.count(".")
    comma_count = s.count(",")
    if dot_count > 1:
        s = s.replace(".", "")
        if comma_count == 1:
            s = s.replace(",", ".")
    elif comma_count > 1:
        s = s.replace(",", "")
        if dot_count == 1:
            pass
    elif dot_count == 1 and comma_count == 1:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif comma_count == 1:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    elif dot_count == 1:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            s = s.replace(".", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_pct(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().replace("%", "").replace(",", ".")
    try:
        return float(s) / 100
    except ValueError:
        return np.nan


def _read_excel_any(local_path: Path, remote_url: str = "", sheet_name=0):
    """Read Excel from URL first (if provided), fallback to local path."""
    if remote_url:
        try:
            with urlopen(remote_url, timeout=45) as resp:
                raw = resp.read()
            return pd.read_excel(BytesIO(raw), engine="openpyxl", sheet_name=sheet_name)
        except Exception:
            pass
    if local_path.exists():
        # Do not swallow local read errors: surface root cause in Streamlit logs/UI.
        return pd.read_excel(local_path, engine="openpyxl", sheet_name=sheet_name)
    return None


def _excel_file_any(local_path: Path, remote_url: str = ""):
    """Open ExcelFile from URL first (if provided), fallback to local path."""
    if remote_url:
        try:
            with urlopen(remote_url, timeout=45) as resp:
                raw = resp.read()
            return pd.ExcelFile(BytesIO(raw), engine="openpyxl")
        except Exception:
            pass
    if local_path.exists():
        # Do not swallow local read errors: surface root cause in Streamlit logs/UI.
        return pd.ExcelFile(local_path, engine="openpyxl")
    return None


def _resolve_data_path(base: Path, folder: str, fname: str) -> Path:
    """
    Support both layouts:
    1) base/folder/file.xlsx
    2) base/file.xlsx
    """
    nested = base / folder / fname
    if nested.exists():
        return nested
    flat = base / fname
    return flat


@st.cache_data(show_spinner="Loading order data...")
def load_orders() -> pd.DataFrame:
    base = Path(__file__).parent
    order_map = {
        "T10": "Order.all.20251001_20251031.xlsx",
        "T11": "Order.all.20251101_20251130.xlsx",
        "T12": "Order.all.20251201_20251231.xlsx",
        "T1_2026": "Order.all.20260101_20260131 (1).xlsx",
        "T2_2026": "Tất cả đơn hàng T2 (1).xlsx",
        "T3_2026": "Tất cả đơn hàng T3.xlsx",
    }
    frames = []
    checked_paths = []
    for folder, fname in order_map.items():
        fp = _resolve_data_path(base, folder, fname)
        checked_paths.append(str(fp))
        remote_url = REMOTE_ORDER_URLS.get(folder, "").strip()
        df = _read_excel_any(fp, remote_url=remote_url)
        if isinstance(df, pd.DataFrame):
            df["_source_month"] = folder
            frames.append(df)
    st.session_state["_order_checked_paths"] = checked_paths
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    df["order_date"] = pd.to_datetime(df["Ngày đặt hàng"], format="mixed", errors="coerce")
    df["complete_date"] = pd.to_datetime(df["Thời gian hoàn thành đơn hàng"], format="mixed", errors="coerce")

    for col in ["Giá gốc", "Giá ưu đãi", "Số lượng", "Tổng giá bán (sản phẩm)",
                 "Tổng giá trị đơn hàng (VND)", "Tổng số tiền người mua thanh toán",
                 "Người bán trợ giá", "Được Shopee trợ giá", "Phí cố định",
                 "Phí Dịch Vụ", "Phí thanh toán", "Mã giảm giá của Shop",
                 "Hoàn Xu", "Mã giảm giá của Shopee",
                 "Phí vận chuyển (dự kiến)", "Phí vận chuyển mà người mua trả"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_cancelled"] = df["Trạng Thái Đơn Hàng"].str.contains("Đã hủy|hủy", case=False, na=False).astype(int)
    df["is_completed"] = (~df["Trạng Thái Đơn Hàng"].str.contains("Đã hủy|hủy|Đang giao", case=False, na=True)).astype(int)
    df["is_returned"] = df["Trạng thái Trả hàng/Hoàn tiền"].notna().astype(int)
    df["is_returned"] = df["is_returned"] & (df["Trạng thái Trả hàng/Hoàn tiền"] != 0)

    df["order_hour"] = df["order_date"].dt.hour
    df["order_dow"] = df["order_date"].dt.dayofweek
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)

    df["is_combo"] = df["Tên sản phẩm"].str.contains(r"[Cc]ombo", na=False).astype(int)

    df["product_short"] = df["Tên sản phẩm"].apply(_shorten_product_name)

    df["cancel_reason_short"] = df["Lý do hủy"].apply(_shorten_cancel_reason)

    region_map = _build_region_map()
    df["region"] = df["Tỉnh/Thành phố"].map(region_map).fillna("Other")

    # Revenue = SUM(Gross sales) - Shop voucher (once per order)
    voucher_per_order = (
        df.sort_values("order_date")
        .groupby("Mã đơn hàng")["Mã giảm giá của Shop"]
        .first()
        .fillna(0)
        .rename("_voucher_shop_once")
    )
    df = df.merge(voucher_per_order, on="Mã đơn hàng", how="left")

    gross_per_order = (
        df.groupby("Mã đơn hàng")["Tổng giá bán (sản phẩm)"]
        .transform("sum")
    )
    df["revenue"] = gross_per_order - df["_voucher_shop_once"].fillna(0)

    df["net_revenue"] = df["revenue"] - df["Phí Dịch Vụ"].fillna(0) - df["Phí thanh toán"].fillna(0) - df["Phí cố định"].fillna(0)

    df["seller_discount"] = df["Người bán trợ giá"].fillna(0) + df["Mã giảm giá của Shop"].fillna(0)
    df["discount_pct"] = np.where(
        df["Giá gốc"] > 0,
        df["seller_discount"] / (df["Giá gốc"] * df["Số lượng"].fillna(1)),
        0,
    )

    return df


def _shorten_product_name(name):
    if pd.isna(name):
        return "N/A"
    s = str(name)
    s = re.sub(r"\[Combo \d+\]\s*", "CB-", s)
    s = re.sub(r"Combo \d+\s*", "CB-", s)
    s = re.sub(r"Giấy Vệ Sinh|GVS", "GVS", s)
    s = re.sub(r"Cao Cấp|cao cấp", "", s)
    s = re.sub(r"OTAKAMI\s*", "", s)
    s = re.sub(r"Tái Chế\s*", "TC ", s)
    s = re.sub(r"Không Lõi|không lõi", "KL", s)
    s = re.sub(r"Có Lõi|có lõi", "CL", s)
    s = re.sub(r"Sống Xanh.*", "", s)
    s = re.sub(r"Chuẩn Xuất Khẩu.*", "", s)
    s = re.sub(r"Tan Nhanh.*", "", s)
    s = re.sub(r"TAN NHANH.*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 55:
        s = s[:52] + "..."
    return s


def _shorten_cancel_reason(reason):
    if pd.isna(reason):
        return None
    s = str(reason)
    if "Thay đổi đơn hàng" in s:
        return "Order modification"
    if "Lý do khác" in s or "Lí do khác" in s:
        return "Other reasons"
    if "Giao hàng thất bại" in s:
        return "Delivery failed"
    if "Voucher" in s or "voucher" in s:
        return "Voucher change"
    if "thay đổi sản phẩm" in s:
        return "Product change"
    if "Chưa được Thanh Toán" in s:
        return "Unpaid"
    if "thay đổi địa chỉ" in s or "delivery address" in s:
        return "Address change"
    if "Đổi ý" in s:
        return "Changed mind"
    if "Thủ tục thanh toán" in s:
        return "Payment issues"
    if "giá rẻ hơn" in s:
        return "Found cheaper"
    if "Hết hàng" in s:
        return "Out of stock"
    if "không gửi hàng" in s:
        return "Not shipped"
    if "thất lạc" in s:
        return "Lost in transit"
    return "Other"


def _build_region_map():
    north = ["Hà Nội", "Hải Phòng", "Quảng Ninh", "Hải Dương", "Hưng Yên",
             "Thái Bình", "Nam Định", "Hà Nam", "Ninh Bình", "Vĩnh Phúc",
             "Bắc Ninh", "Bắc Giang", "Phú Thọ", "Thái Nguyên", "Lạng Sơn",
             "Cao Bằng", "Bắc Kạn", "Tuyên Quang", "Hà Giang", "Lào Cai",
             "Yên Bái", "Sơn La", "Điện Biên", "Lai Châu", "Hòa Bình"]
    central = ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị",
               "Thừa Thiên Huế", "Đà Nẵng", "Quảng Nam", "Quảng Ngãi",
               "Bình Định", "Phú Yên", "Khánh Hòa", "Ninh Thuận", "Bình Thuận",
               "Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng"]
    south = ["TP. Hồ Chí Minh", "Bình Dương", "Đồng Nai", "Bà Rịa - Vũng Tàu",
             "Tây Ninh", "Bình Phước", "Long An", "Tiền Giang", "Bến Tre",
             "Vĩnh Long", "Trà Vinh", "Đồng Tháp", "An Giang", "Kiên Giang",
             "Cần Thơ", "Hậu Giang", "Sóc Trăng", "Bạc Liêu", "Cà Mau"]
    m = {}
    for p in north:
        m[p] = "North"
    for p in central:
        m[p] = "Central"
    for p in south:
        m[p] = "South"
    return m


@st.cache_data(show_spinner="Loading shop stats...")
def load_shop_stats() -> pd.DataFrame:
    base = Path(__file__).parent
    stats_map = {
        "T10": "otakamitissuevn.shopee-shop-stats.20251001-20251031.xlsx",
        "T11": "otakamitissuevn.shopee-shop-stats.20251101-20251130.xlsx",
        "T12": "otakamitissuevn.shopee-shop-stats.20251201-20251231.xlsx",
        "T1_2026": "otakamitissuevn.shopee-shop-stats.20260101-20260131 (1).xlsx",
    }
    stats_simple = {
        "T2_2026": "Phân tích bán hàng T2.xlsx",
        "T3_2026": "Phân tích bán hàng T3.xlsx",
    }

    frames = []

    for folder, fname in stats_map.items():
        fp = _resolve_data_path(base, folder, fname)
        try:
            remote_url = REMOTE_STATS_URLS.get(folder, "").strip()
            xl = _excel_file_any(fp, remote_url=remote_url)
            if xl is None:
                continue
            sheet_name = "Đơn hàng đã đặt" if "Đơn hàng đã đặt" in xl.sheet_names else xl.sheet_names[0]
            df_s = xl.parse(sheet_name)
            df_s = df_s.dropna(how="all")
            if "Ngày" in df_s.columns:
                header_mask = df_s["Ngày"] == "Ngày"
                if header_mask.any():
                    idx = header_mask.idxmax()
                    df_s.columns = df_s.loc[idx]
                    df_s = df_s.loc[idx + 1:].reset_index(drop=True)

                summary_mask = df_s["Ngày"].astype(str).str.contains(r"\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{4}", na=False)
                df_s = df_s[~summary_mask].copy()

                df_s["Ngày"] = pd.to_datetime(df_s["Ngày"], format="mixed", errors="coerce")
                df_s = df_s.dropna(subset=["Ngày"])

            for c in df_s.columns:
                if c == "Ngày":
                    continue
                if "lệ" in str(c).lower() or "tỉ" in str(c).lower() or "%" in str(c).lower():
                    df_s[c] = df_s[c].apply(_parse_pct)
                else:
                    df_s[c] = df_s[c].apply(_parse_vnd)
            frames.append(df_s)
        except Exception:
            continue

    for folder, fname in stats_simple.items():
        fp = _resolve_data_path(base, folder, fname)
        try:
            remote_url = REMOTE_STATS_URLS.get(folder, "").strip()
            df_s = _read_excel_any(fp, remote_url=remote_url)
            if not isinstance(df_s, pd.DataFrame):
                continue
            df_s = df_s.dropna(how="all")
            header_mask = df_s.iloc[:, 0].astype(str) == "Ngày"
            if header_mask.any():
                idx = header_mask.idxmax()
                df_s.columns = df_s.loc[idx]
                df_s = df_s.loc[idx + 1:].reset_index(drop=True)

            summary_mask = df_s.iloc[:, 0].astype(str).str.contains(r"\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{4}", na=False)
            df_s = df_s[~summary_mask].copy()

            first_col = df_s.columns[0]
            df_s[first_col] = pd.to_datetime(df_s[first_col], format="mixed", errors="coerce")
            df_s = df_s.dropna(subset=[first_col])
            if first_col != "Ngày":
                df_s = df_s.rename(columns={first_col: "Ngày"})

            for c in df_s.columns:
                if c == "Ngày":
                    continue
                if "lệ" in str(c).lower() or "tỉ" in str(c).lower():
                    df_s[c] = df_s[c].apply(_parse_pct)
                else:
                    df_s[c] = df_s[c].apply(_parse_vnd)

            common_cols = ["Ngày", "Tổng doanh số (VND)", "Tổng số đơn hàng",
                           "Doanh số trên mỗi đơn hàng", "Lượt nhấp vào sản phẩm",
                           "Số lượt truy cập", "Tỷ lệ chuyển đổi đơn hàng",
                           "Đơn đã hủy", "Doanh số đơn hủy",
                           "Đơn đã hoàn trả / hoàn tiền",
                           "Doanh số các đơn Trả hàng/Hoàn tiền",
                           "số người mua", "số người mua mới",
                           "số người mua hiện tại",
                           "số người mua tiềm năng",
                           "Tỉ lệ quay lại của người mua"]
            available = [c for c in common_cols if c in df_s.columns]
            df_s = df_s[available]
            frames.append(df_s)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    stats = pd.concat(frames, ignore_index=True)
    stats = stats.sort_values("Ngày").reset_index(drop=True)
    return stats


@st.cache_data(show_spinner="Loading traffic sources...")
def load_traffic_sources() -> pd.DataFrame:
    base = Path(__file__).parent
    stats_map = {
        "T10": "otakamitissuevn.shopee-shop-stats.20251001-20251031.xlsx",
        "T11": "otakamitissuevn.shopee-shop-stats.20251101-20251130.xlsx",
        "T12": "otakamitissuevn.shopee-shop-stats.20251201-20251231.xlsx",
        "T1_2026": "otakamitissuevn.shopee-shop-stats.20260101-20260131 (1).xlsx",
    }
    frames = []
    for folder, fname in stats_map.items():
        fp = _resolve_data_path(base, folder, fname)
        try:
            remote_url = REMOTE_STATS_URLS.get(folder, "").strip()
            xl = _excel_file_any(fp, remote_url=remote_url)
            if xl is None:
                continue
            sheet_name = None
            for sn in xl.sheet_names:
                if "Nguồn truy cập cho" in sn or "Nguồn truy cập" in sn:
                    sheet_name = sn
                    break
            if not sheet_name:
                continue
            df_t = xl.parse(sheet_name)
            df_t = df_t.dropna(how="all")
            summary_mask = df_t.iloc[:, 0].astype(str).str.contains(r"\d{2}-\d{2}-\d{4}-\d{2}-\d{2}-\d{4}", na=False)
            if summary_mask.any():
                idx_s = summary_mask.idxmax()
                row = df_t.loc[idx_s]
                frames.append({
                    "period": folder,
                    "total": _parse_vnd(row.iloc[2]) if len(row) > 2 else 0,
                    "the_sp": _parse_vnd(row.iloc[3]) if len(row) > 3 else 0,
                    "livestream": _parse_vnd(row.iloc[4]) if len(row) > 4 else 0,
                    "video": _parse_vnd(row.iloc[5]) if len(row) > 5 else 0,
                    "affiliate": _parse_vnd(row.iloc[6]) if len(row) > 6 else 0,
                })
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _layout(fig, h=420):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=PAL["txt"]),
        margin=dict(l=50, r=30, t=50, b=50),
        height=h, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#e2e8f0", borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def fmt_vnd(val):
    if pd.isna(val) or val == 0:
        return "0"
    if abs(val) >= 1e9:
        return f"{val / 1e9:.1f}B"
    if abs(val) >= 1e6:
        return f"{val / 1e6:.1f}M"
    if abs(val) >= 1e3:
        return f"{val / 1e3:.0f}K"
    return f"{val:,.0f}"


def kpi_card(label, value, delta=None, delta_good=True):
    delta_html = ""
    if delta is not None:
        color = PAL["ok"] if delta_good else PAL["bad"]
        arrow = "▲" if delta_good else "▼"
        delta_html = f'<div style="font-size:.82rem;color:{color};margin-top:2px">{arrow} {delta}</div>'
    return f'''
    <div style="flex:1;background:#fff;border:1px solid #cbd5e0;border-top:4px solid #e8734a;
                border-radius:8px;padding:16px 18px;text-align:center;
                box-shadow:0 1px 4px rgba(0,0,0,.06);min-width:140px">
        <div style="font-size:.82rem;color:#6b7280;font-weight:600;letter-spacing:.5px;
                    margin-bottom:4px">{label}</div>
        <div style="font-size:1.9rem;color:#2c5f7c;font-weight:800;line-height:1.1">{value}</div>
        {delta_html}
    </div>'''


# ---------------------------------------------------------------------------
# Page 1: Executive Overview
# ---------------------------------------------------------------------------
def page_executive(df, stats):
    st.html("""
    <div class="rq-box">
        <h2>RQ1 — Overall Business Performance</h2>
        <p class="rq-question">How is OTAKAMI's overall business performance across core KPIs?
        Which metrics need improvement?</p>
    </div>
    """)

    completed = df[df["is_completed"] == 1]
    total_orders = len(df)
    total_gmv = df.drop_duplicates("Mã đơn hàng")["revenue"].sum()
    total_net = completed.drop_duplicates("Mã đơn hàng")["net_revenue"].sum()
    aov = completed.drop_duplicates("Mã đơn hàng")["revenue"].mean()
    cancel_rate = df["is_cancelled"].mean()
    fulfillment_rate = df["is_completed"].mean()

    kpi_html = '<div style="display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap">'
    kpi_html += kpi_card("TOTAL ORDERS", f"{total_orders:,}")
    kpi_html += kpi_card("GMV", fmt_vnd(total_gmv))
    kpi_html += kpi_card("NET REVENUE", fmt_vnd(total_net))
    kpi_html += kpi_card("AOV", fmt_vnd(aov))
    kpi_html += kpi_card("CANCEL RATE", f"{cancel_rate:.1%}", delta=f"{cancel_rate:.1%}", delta_good=cancel_rate < 0.15)
    kpi_html += kpi_card("FULFILLMENT", f"{fulfillment_rate:.1%}", delta=f"{fulfillment_rate:.1%}", delta_good=fulfillment_rate > 0.80)
    kpi_html += '</div>'
    st.html(kpi_html)

    col1, col2 = st.columns(2)

    with col1:
        if not stats.empty and "Ngày" in stats.columns:
            s = stats.copy()
            s["month"] = s["Ngày"].dt.to_period("M").astype(str)
            monthly = s.groupby("month").agg({
                "Tổng doanh số (VND)": "sum",
                "Tổng số đơn hàng": "sum",
            }).reset_index()
            monthly.columns = ["Month", "Revenue", "Orders"]

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly["Month"], y=monthly["Revenue"],
                                 name="Revenue (VND)", marker_color=PAL["sec"],
                                 opacity=0.7), secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Orders"],
                                     name="Orders", mode="lines+markers",
                                     line=dict(color=PAL["pri"], width=3),
                                     marker=dict(size=8)), secondary_y=True)
            fig.update_layout(title="Monthly Revenue & Orders Trend",
                              barmode="group")
            fig.update_yaxes(title_text="Revenue (VND)", secondary_y=False)
            fig.update_yaxes(title_text="Orders", secondary_y=True)
            st.plotly_chart(_layout(fig), width="stretch", key="trend_rev")
        else:
            monthly_o = df.drop_duplicates("Mã đơn hàng").groupby("order_month").agg(
                gmv=("revenue", "sum"),
                orders=("Mã đơn hàng", "nunique"),
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=monthly_o["order_month"], y=monthly_o["gmv"],
                                 name="GMV", marker_color=PAL["sec"], opacity=0.7),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=monthly_o["order_month"], y=monthly_o["orders"],
                                     name="Orders", mode="lines+markers",
                                     line=dict(color=PAL["pri"], width=3)), secondary_y=True)
            fig.update_layout(title="Monthly GMV & Orders Trend")
            fig.update_yaxes(title_text="GMV (VND)", secondary_y=False)
            fig.update_yaxes(title_text="Orders", secondary_y=True)
            st.plotly_chart(_layout(fig), width="stretch", key="trend_rev_o")

    with col2:
        total_placed = len(df)
        confirmed = len(df[df["is_cancelled"] == 0])
        completed_n = len(df[df["is_completed"] == 1])
        returned = df["is_returned"].sum()
        paid = completed_n - returned

        funnel_data = pd.DataFrame({
            "Stage": ["Placed", "Confirmed", "Completed", "Paid"],
            "Count": [total_placed, confirmed, completed_n, paid],
        })

        fig = go.Figure(go.Funnel(
            y=funnel_data["Stage"],
            x=funnel_data["Count"],
            textinfo="value+percent previous",
            marker=dict(color=[PAL["sec"], PAL["acc"], PAL["ok"], PAL["pri"]]),
            connector=dict(line=dict(color="#e2e8f0", dash="dot", width=1)),
        ))
        fig.update_layout(title="Order Funnel")
        st.plotly_chart(_layout(fig, h=420), width="stretch", key="funnel")

    if not stats.empty and "Ngày" in stats.columns:
        st.markdown("### Daily Trends")
        metric_choice = st.selectbox(
            "Select metric:",
            ["Tổng doanh số (VND)", "Tổng số đơn hàng", "Lượt nhấp vào sản phẩm",
             "Số lượt truy cập", "Đơn đã hủy"],
            format_func=lambda x: {
                "Tổng doanh số (VND)": "Total Revenue (VND)",
                "Tổng số đơn hàng": "Total Orders",
                "Lượt nhấp vào sản phẩm": "Product Clicks",
                "Số lượt truy cập": "Visits",
                "Đơn đã hủy": "Cancelled Orders",
            }.get(x, x),
            key="daily_metric",
        )
        display_name = {
            "Tổng doanh số (VND)": "Total Revenue (VND)",
            "Tổng số đơn hàng": "Total Orders",
            "Lượt nhấp vào sản phẩm": "Product Clicks",
            "Số lượt truy cập": "Visits",
            "Đơn đã hủy": "Cancelled Orders",
        }.get(metric_choice, metric_choice)
        s = stats.dropna(subset=[metric_choice])
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=s["Ngày"], y=s[metric_choice],
            mode="lines", line=dict(color=PAL["pri"], width=2),
            fill="tozeroy", fillcolor="rgba(232,115,74,.08)",
        ))
        ma7 = s[metric_choice].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=s["Ngày"], y=ma7,
            mode="lines", name="MA-7",
            line=dict(color=PAL["sec"], width=2.5, dash="dash"),
        ))
        fig.update_layout(title=f"{display_name} — Daily Trend",
                          xaxis_title="Date", yaxis_title=display_name,
                          showlegend=True)
        st.plotly_chart(_layout(fig, 380), width="stretch", key="daily_trend")


# ---------------------------------------------------------------------------
# Page 2: Product Performance
# ---------------------------------------------------------------------------
def page_product(df):
    st.html("""
    <div class="rq-box">
        <h2>RQ2 — Product Performance</h2>
        <p class="rq-question">Which products drive the most revenue? Are Combos more effective
        than single items (AOV, cancel rate, CVR)?</p>
    </div>
    """)

    prod_rev = df.groupby("product_short").agg(
        revenue=("Tổng giá bán (sản phẩm)", "sum"),
        orders=("Mã đơn hàng", "count"),
        qty=("Số lượng", "sum"),
        cancel_rate=("is_cancelled", "mean"),
        aov=("Tổng giá bán (sản phẩm)", "mean"),
    ).reset_index().sort_values("revenue", ascending=False)
    prod_rev["cum_pct"] = prod_rev["revenue"].cumsum() / prod_rev["revenue"].sum()

    top_n = min(20, len(prod_rev))
    pr = prod_rev.head(top_n)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=pr["product_short"], y=pr["revenue"],
        name="Revenue", marker_color=PAL["sec"], opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pr["product_short"], y=pr["cum_pct"],
        name="Cumulative %", mode="lines+markers",
        line=dict(color=PAL["pri"], width=3), marker=dict(size=7),
    ), secondary_y=True)
    fig.add_hline(y=0.8, line_dash="dot", line_color=PAL["muted"],
                  annotation_text="80%", secondary_y=True)
    fig.update_layout(title="Pareto — Revenue by Product (Top 20)",
                      xaxis_tickangle=-45)
    fig.update_yaxes(title_text="Revenue (VND)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", tickformat=".0%", secondary_y=True)
    st.plotly_chart(_layout(fig, 480), width="stretch", key="pareto")

    st.markdown("### Combo vs. Single Items")
    st.html("""
    <div class="concept-tip">
        Comparing performance between <b>Combo</b> (bundle) and <b>Single</b> products.
        Combos typically have higher AOV but need to be checked for cancel rates and volume.
    </div>
    """)

    combo_stats = df.groupby("is_combo").agg(
        orders=("Mã đơn hàng", "count"),
        revenue=("Tổng giá bán (sản phẩm)", "sum"),
        aov=("Tổng giá bán (sản phẩm)", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_qty=("Số lượng", "mean"),
    ).reset_index()
    combo_stats["is_combo"] = combo_stats["is_combo"].map({0: "Single", 1: "Combo"})

    c1, c2, c3 = st.columns(3)
    metrics_combo = [("AOV (VND)", "aov"), ("Cancel Rate", "cancel_rate"), ("Avg Quantity", "avg_qty")]
    for col_ui, (label, metric) in zip([c1, c2, c3], metrics_combo):
        with col_ui:
            fig = go.Figure(go.Bar(
                x=combo_stats["is_combo"],
                y=combo_stats[metric],
                marker_color=[PAL["sec"], PAL["pri"]],
                text=[f"{v:,.0f}" if metric != "cancel_rate" else f"{v:.1%}" for v in combo_stats[metric]],
                textposition="outside",
            ))
            fmt = ".0%" if metric == "cancel_rate" else ",.0f"
            fig.update_layout(title=label, yaxis_tickformat=fmt)
            st.plotly_chart(_layout(fig, 320), width="stretch", key=f"combo_{metric}")

    st.markdown("### Product x KPI Heatmap")
    prod_kpi = df.groupby("product_short").agg(
        revenue=("Tổng giá bán (sản phẩm)", "sum"),
        orders=("Mã đơn hàng", "count"),
        aov=("Tổng giá bán (sản phẩm)", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_qty=("Số lượng", "mean"),
        discount_pct=("discount_pct", "mean"),
    ).reset_index()
    prod_kpi = prod_kpi.sort_values("revenue", ascending=False).head(15)

    from sklearn.preprocessing import MinMaxScaler
    num_cols = ["revenue", "orders", "aov", "cancel_rate", "avg_qty", "discount_pct"]
    scaler = MinMaxScaler()
    norm = pd.DataFrame(
        scaler.fit_transform(prod_kpi[num_cols]),
        columns=num_cols,
    )

    fig = go.Figure(go.Heatmap(
        z=norm.values,
        x=["Revenue", "Orders", "AOV", "Cancel Rate", "Avg Qty", "Discount %"],
        y=prod_kpi["product_short"].values,
        colorscale="YlOrRd",
        text=[[f"{prod_kpi.iloc[i][c]:,.0f}" if c != "cancel_rate" and c != "discount_pct"
               else f"{prod_kpi.iloc[i][c]:.1%}" for c in num_cols]
              for i in range(len(prod_kpi))],
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(title="Top 15 Products — Normalized KPI Heatmap", yaxis_autorange="reversed")
    st.plotly_chart(_layout(fig, max(400, len(prod_kpi) * 30 + 100)), width="stretch", key="heatmap_prod")

    st.markdown("### K-Means Product Clustering")
    st.html("""
    <div class="concept-tip">
        <b>K-Means Clustering</b> groups products based on performance features (revenue, orders, AOV,
        cancel rate, avg quantity, discount %). Optimal k is determined via <b>Elbow Method</b>
        and <b>Silhouette Score</b>. Data is standardized (z-score) before clustering.
    </div>
    """)

    prod_cluster = df.groupby("product_short").agg(
        revenue=("Tổng giá bán (sản phẩm)", "sum"),
        orders=("Mã đơn hàng", "count"),
        aov=("Tổng giá bán (sản phẩm)", "mean"),
        cancel_rate=("is_cancelled", "mean"),
        avg_qty=("Số lượng", "mean"),
        discount_pct=("discount_pct", "mean"),
    ).reset_index()
    prod_cluster = prod_cluster.dropna()

    if len(prod_cluster) >= 4:
        features = ["revenue", "orders", "aov", "cancel_rate", "avg_qty", "discount_pct"]
        X = StandardScaler().fit_transform(prod_cluster[features])

        k_range = range(2, min(8, len(prod_cluster)))
        wcss = []
        sil_scores = []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_))

        best_k = list(k_range)[np.argmax(sil_scores)]

        c_elbow, c_sil = st.columns(2)
        with c_elbow:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                       line=dict(color=PAL["sec"], width=2.5),
                                       marker=dict(size=8)))
            fig_e.update_layout(title="Elbow Method (WCSS)", xaxis_title="k", yaxis_title="WCSS")
            st.plotly_chart(_layout(fig_e, 320), width="stretch", key="elbow")
        with c_sil:
            fig_s = go.Figure()
            fig_s.add_trace(go.Bar(x=list(k_range), y=sil_scores,
                                   marker_color=[PAL["pri"] if k == best_k else PAL["muted"] for k in k_range]))
            fig_s.update_layout(title=f"Silhouette Score (best k={best_k})", xaxis_title="k", yaxis_title="Score")
            st.plotly_chart(_layout(fig_s, 320), width="stretch", key="silhouette")

        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        prod_cluster["cluster"] = km_final.fit_predict(X)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        prod_cluster["PC1"] = coords[:, 0]
        prod_cluster["PC2"] = coords[:, 1]

        cluster_colors = [PAL["pri"], PAL["sec"], PAL["ok"], PAL["warn"], PAL["purple"], PAL["teal"]]
        fig_pca = go.Figure()
        for cl in sorted(prod_cluster["cluster"].unique()):
            sub = prod_cluster[prod_cluster["cluster"] == cl]
            fig_pca.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"], mode="markers+text",
                marker=dict(size=12, color=cluster_colors[cl % len(cluster_colors)]),
                text=sub["product_short"],
                textposition="top center", textfont=dict(size=9),
                name=f"Cluster {cl} (n={len(sub)})",
            ))
        var_exp = pca.explained_variance_ratio_
        fig_pca.update_layout(
            title=f"PCA 2D — Product Clusters (k={best_k}, explained var: {var_exp.sum():.1%})",
            xaxis_title=f"PC1 ({var_exp[0]:.1%})",
            yaxis_title=f"PC2 ({var_exp[1]:.1%})",
        )
        st.plotly_chart(_layout(fig_pca, 520), width="stretch", key="pca_product")

        st.markdown("#### Cluster Profile")
        cluster_profile = prod_cluster.groupby("cluster")[features].mean().round(2)
        cluster_profile.columns = ["Avg Revenue", "Avg Orders", "AOV", "Cancel Rate", "Avg Qty", "Discount %"]
        st.dataframe(cluster_profile.style.format({
            "Avg Revenue": "{:,.0f}",
            "Avg Orders": "{:,.0f}",
            "AOV": "{:,.0f}",
            "Cancel Rate": "{:.2%}",
            "Avg Qty": "{:.1f}",
            "Discount %": "{:.2%}",
        }).background_gradient(cmap="YlOrRd", axis=0), width="stretch")
    else:
        st.info("Not enough products to run K-Means Clustering.")


# ---------------------------------------------------------------------------
# Page 3: Customer & Operations
# ---------------------------------------------------------------------------
def page_customer_ops(df):
    st.html("""
    <div class="rq-box">
        <h2>RQ3 — Buying Behavior & Operations</h2>
        <p class="rq-question">What are the main drivers of order cancellation (~18%)? How do product,
        shipping, geography, and payment relate to cancellations? How does buying behavior change over time?</p>
    </div>
    """)

    st.markdown("### Order Heatmap: Hour x Day of Week")
    st.html("""
    <div class="concept-tip">
        Heatmap shows <b>order volume</b> by hour and day of week.
        Dark cells indicate <b>peak times</b> — ideal for flash sales or push notifications.
    </div>
    """)

    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_data = df.groupby(["order_dow", "order_hour"]).size().reset_index(name="count")
    pivot_hm = heatmap_data.pivot(index="order_dow", columns="order_hour", values="count").fillna(0)
    pivot_hm.index = [dow_labels[i] for i in pivot_hm.index]

    fig = go.Figure(go.Heatmap(
        z=pivot_hm.values,
        x=[f"{h}h" for h in pivot_hm.columns],
        y=pivot_hm.index,
        colorscale="YlOrRd",
        text=pivot_hm.values.astype(int),
        texttemplate="%{text}",
        textfont=dict(size=10),
    ))
    fig.update_layout(title="Order Distribution: Hour x Day of Week",
                      xaxis_title="Hour", yaxis_title="", yaxis_autorange="reversed")
    st.plotly_chart(_layout(fig, 380), width="stretch", key="heatmap_time")

    st.markdown("### Geographic Distribution")
    c_geo1, c_geo2 = st.columns(2)

    with c_geo1:
        region_stats = df.groupby("region").agg(
            orders=("Mã đơn hàng", "count"),
            revenue=("Tổng giá bán (sản phẩm)", "sum"),
            cancel_rate=("is_cancelled", "mean"),
        ).reset_index().sort_values("orders", ascending=False)

        fig = go.Figure(go.Pie(
            labels=region_stats["region"],
            values=region_stats["orders"],
            marker=dict(colors=[PAL["sec"], PAL["pri"], PAL["ok"], PAL["muted"]]),
            textinfo="label+percent",
            hole=0.4,
        ))
        fig.update_layout(title="Orders by Region")
        st.plotly_chart(_layout(fig, 380), width="stretch", key="region_pie")

    with c_geo2:
        top_prov = df.groupby("Tỉnh/Thành phố").agg(
            orders=("Mã đơn hàng", "count"),
            cancel_rate=("is_cancelled", "mean"),
        ).reset_index().sort_values("orders", ascending=False).head(15)

        fig = go.Figure(go.Bar(
            y=top_prov["Tỉnh/Thành phố"], x=top_prov["orders"],
            orientation="h", marker_color=PAL["sec"],
            text=[f"{n:,}" for n in top_prov["orders"]],
            textposition="outside",
        ))
        fig.update_layout(title="Top 15 Provinces by Orders",
                          yaxis_autorange="reversed", xaxis_title="Orders")
        st.plotly_chart(_layout(fig, max(380, len(top_prov) * 28 + 80)),
                        width="stretch", key="province_bar")

    st.markdown("### Cancellation Analysis")
    st.html("""
    <div class="concept-tip">
        High cancellation rates hurt shop ratings and operational costs.
        Analyzing cancellation reasons helps identify <b>root causes</b> for improvement.
    </div>
    """)

    cancelled = df[df["is_cancelled"] == 1]

    c_can1, c_can2 = st.columns(2)

    with c_can1:
        reason_count = cancelled["cancel_reason_short"].value_counts().reset_index()
        reason_count.columns = ["Reason", "Count"]
        fig = go.Figure(go.Bar(
            y=reason_count["Reason"], x=reason_count["Count"],
            orientation="h",
            marker_color=[PAL["bad"] if n > reason_count["Count"].median() else PAL["muted"]
                          for n in reason_count["Count"]],
            text=[f"{n:,}" for n in reason_count["Count"]],
            textposition="outside",
        ))
        fig.update_layout(title="Cancellation Reasons", yaxis_autorange="reversed",
                          xaxis_title="Orders")
        st.plotly_chart(_layout(fig, max(350, len(reason_count) * 30 + 80)),
                        width="stretch", key="cancel_reason")

    with c_can2:
        df_pay = df[df["Phương thức thanh toán"].apply(lambda x: isinstance(x, str))].copy()
        payment_cancel = df_pay.groupby("Phương thức thanh toán").agg(
            total=("Mã đơn hàng", "count"),
            cancelled=("is_cancelled", "sum"),
        ).reset_index()
        payment_cancel["rate"] = payment_cancel["cancelled"] / payment_cancel["total"]
        payment_cancel = payment_cancel.sort_values("rate", ascending=True)
        overall_rate = df["is_cancelled"].mean()

        fig = go.Figure(go.Bar(
            y=payment_cancel["Phương thức thanh toán"],
            x=payment_cancel["rate"],
            orientation="h",
            marker_color=[PAL["bad"] if r > overall_rate else PAL["ok"] for r in payment_cancel["rate"]],
            text=[f"{r:.1%} (n={n:,})" for r, n in zip(payment_cancel["rate"], payment_cancel["total"])],
            textposition="outside",
        ))
        fig.add_vline(x=overall_rate, line_dash="dash", line_color=PAL["muted"],
                      annotation_text=f"Avg: {overall_rate:.1%}")
        fig.update_layout(title="Cancel Rate by Payment Method",
                          xaxis_tickformat=".0%", yaxis_autorange="reversed",
                          xaxis_range=[0, max(payment_cancel["rate"]) * 1.4])
        st.plotly_chart(_layout(fig, max(350, len(payment_cancel) * 35 + 80)),
                        width="stretch", key="cancel_payment")

    st.markdown("### Cancel Rate by Dimension")
    dim_choice = st.selectbox(
        "Select dimension:",
        ["Đơn Vị Vận Chuyển", "Tỉnh/Thành phố", "product_short", "region"],
        format_func=lambda x: {"Đơn Vị Vận Chuyển": "Shipping Carrier",
                                "Tỉnh/Thành phố": "Province",
                                "product_short": "Product",
                                "region": "Region"}.get(x, x),
        key="cancel_dim",
    )
    dim_stats = df.groupby(dim_choice).agg(
        total=("Mã đơn hàng", "count"),
        cancelled=("is_cancelled", "sum"),
    ).reset_index()
    dim_stats["rate"] = dim_stats["cancelled"] / dim_stats["total"]
    dim_stats = dim_stats[dim_stats["total"] >= 10].sort_values("rate", ascending=True).tail(20)

    fig = go.Figure(go.Bar(
        y=dim_stats[dim_choice].astype(str),
        x=dim_stats["rate"],
        orientation="h",
        marker_color=[PAL["bad"] if r > overall_rate else PAL["sec"] for r in dim_stats["rate"]],
        text=[f"{r:.1%} (n={n:,})" for r, n in zip(dim_stats["rate"], dim_stats["total"])],
        textposition="outside",
    ))
    fig.add_vline(x=overall_rate, line_dash="dash", line_color=PAL["muted"],
                  annotation_text=f"Avg: {overall_rate:.1%}")
    dim_label = {"Đơn Vị Vận Chuyển": "Shipping Carrier", "Tỉnh/Thành phố": "Province",
                  "product_short": "Product", "region": "Region"}.get(dim_choice, dim_choice)
    fig.update_layout(title=f"Cancel Rate by {dim_label}",
                      xaxis_tickformat=".0%", yaxis_autorange="reversed",
                      xaxis_range=[0, max(dim_stats["rate"].max() * 1.4, overall_rate * 1.5)])
    st.plotly_chart(_layout(fig, max(380, len(dim_stats) * 28 + 80)),
                    width="stretch", key="cancel_dim_chart")


# ---------------------------------------------------------------------------
# Page 4: Traffic & Conversion
# ---------------------------------------------------------------------------
def page_traffic(df, stats, traffic):
    st.html("""
    <div class="rq-box">
        <h2>RQ4 — Traffic Source Effectiveness & Conversion</h2>
        <p class="rq-question">Which traffic channel delivers the highest CVR and ROAS?
        Is Livestream/Affiliate more effective than Product Card?</p>
    </div>
    """)

    if not traffic.empty:
        st.markdown("### Revenue Distribution by Traffic Source")
        st.html("""
        <div class="concept-tip">
            Shopee splits revenue into 4 main sources: <b>Product Card</b> (organic search),
            <b>Livestream</b>, <b>Video</b>, and <b>Affiliate</b> (partner links).
            The radar chart compares revenue structure across months.
        </div>
        """)

        traffic["other"] = traffic["total"] - traffic["the_sp"] - traffic["livestream"] - traffic["video"] - traffic["affiliate"]
        traffic["other"] = traffic["other"].clip(lower=0)

        categories = ["Product Card", "Livestream", "Video", "Affiliate", "Other"]

        fig = go.Figure()
        colors_radar = [PAL["sec"], PAL["pri"], PAL["ok"], PAL["warn"], PAL["purple"]]
        for i, row in traffic.iterrows():
            values = [row["the_sp"], row["livestream"], row["video"], row["affiliate"], row["other"]]
            total = sum(v for v in values if v and not np.isnan(v))
            if total == 0:
                continue
            pcts = [v / total * 100 if v and not np.isnan(v) else 0 for v in values]
            pcts.append(pcts[0])
            cats_closed = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=pcts, theta=cats_closed,
                fill="toself", name=row["period"],
                line=dict(width=2),
                opacity=0.6,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")),
            title="Radar — Traffic Source Structure (% of Revenue)",
        )
        st.plotly_chart(_layout(fig, 480), width="stretch", key="radar_traffic")

        fig2 = go.Figure()
        for src, col_name, color in [
            ("Product Card", "the_sp", TRAFFIC_COLORS["Product Card"]),
            ("Livestream", "livestream", TRAFFIC_COLORS["Livestream"]),
            ("Video", "video", TRAFFIC_COLORS["Video"]),
            ("Affiliate", "affiliate", TRAFFIC_COLORS["Affiliate"]),
            ("Other", "other", TRAFFIC_COLORS["Other"]),
        ]:
            fig2.add_trace(go.Bar(
                x=traffic["period"], y=traffic[col_name],
                name=src, marker_color=color,
            ))
        fig2.update_layout(barmode="stack", title="Revenue by Traffic Source (Stacked)",
                           xaxis_title="Month", yaxis_title="Revenue (VND)")
        st.plotly_chart(_layout(fig2, 420), width="stretch", key="traffic_stacked")

    if not stats.empty:
        st.markdown("### Traffic & Conversion Trends")
        s = stats.dropna(subset=["Lượt nhấp vào sản phẩm"]).copy()
        if "Tỷ lệ chuyển đổi đơn hàng" in s.columns and len(s) > 0:
            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(go.Bar(
                x=s["Ngày"], y=s["Lượt nhấp vào sản phẩm"],
                name="Clicks", marker_color=PAL["sec"], opacity=0.5,
            ), secondary_y=False)
            fig3.add_trace(go.Scatter(
                x=s["Ngày"], y=s["Tỷ lệ chuyển đổi đơn hàng"],
                name="CVR", mode="lines",
                line=dict(color=PAL["pri"], width=2.5),
            ), secondary_y=True)
            fig3.update_layout(title="Clicks vs. Conversion Rate (CVR)")
            fig3.update_yaxes(title_text="Clicks", secondary_y=False)
            fig3.update_yaxes(title_text="CVR", tickformat=".1%", secondary_y=True)
            st.plotly_chart(_layout(fig3), width="stretch", key="traffic_cvr")

    if not stats.empty:
        st.markdown("### New vs. Returning Buyers")
        s = stats.copy()
        if "số người mua mới" in s.columns and "số người mua hiện tại" in s.columns:
            s["month"] = s["Ngày"].dt.to_period("M").astype(str)
            buyer_monthly = s.groupby("month").agg(
                new=("số người mua mới", "sum"),
                returning=("số người mua hiện tại", "sum"),
            ).reset_index()

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(x=buyer_monthly["month"], y=buyer_monthly["new"],
                                  name="New Buyers", marker_color=PAL["sec"]))
            fig4.add_trace(go.Bar(x=buyer_monthly["month"], y=buyer_monthly["returning"],
                                  name="Returning Buyers", marker_color=PAL["pri"]))
            fig4.update_layout(barmode="group", title="New vs. Returning Buyers by Month",
                               xaxis_title="Month", yaxis_title="Buyers")
            st.plotly_chart(_layout(fig4), width="stretch", key="buyer_new_ret")

        if "Tỉ lệ quay lại của người mua" in s.columns:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=s["Ngày"], y=s["Tỉ lệ quay lại của người mua"],
                mode="lines", line=dict(color=PAL["ok"], width=2),
                fill="tozeroy", fillcolor="rgba(74,156,109,.08)",
            ))
            fig5.update_layout(title="Buyer Return Rate",
                               xaxis_title="Date", yaxis_title="Return Rate",
                               yaxis_tickformat=".1%")
            st.plotly_chart(_layout(fig5, 340), width="stretch", key="return_rate")


# ---------------------------------------------------------------------------
# Page 5: K-Means Order Clustering
# ---------------------------------------------------------------------------
def page_order_clustering(df):
    st.html("""
    <div class="rq-box">
        <h2>RQ5 — Order Segmentation (K-Means)</h2>
        <p class="rq-question">How many order segments does K-Means identify?
        What are the characteristics and strategies for each segment?</p>
    </div>
    """)

    st.html("""
    <div class="concept-tip">
        Order segmentation via K-Means uses 6 features: order value, product quantity, discount %,
        order hour, shipping fee, and payment method (encoded). This helps identify distinct
        buying behavior groups for targeted marketing strategies.
    </div>
    """)

    order_df = df.dropna(subset=["revenue", "Số lượng", "order_hour"]).copy()

    order_df["payment_code"] = order_df["Phương thức thanh toán"].apply(
        lambda x: 0 if "nhận hàng" in str(x).lower() else 1
    )

    shipping_col = "Phí vận chuyển mà người mua trả"
    if shipping_col not in order_df.columns:
        shipping_col = "Phí vận chuyển (dự kiến)"
    if shipping_col not in order_df.columns:
        order_df["_shipping"] = 0
        shipping_col = "_shipping"

    features = ["revenue", "Số lượng", "discount_pct",
                "order_hour", shipping_col, "payment_code"]
    feature_labels = ["Order Value", "Quantity", "Discount %",
                      "Order Hour", "Shipping Fee", "Online Payment"]

    X_raw = order_df[features].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    k_range = range(2, 8)
    wcss = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    best_k = list(k_range)[np.argmax(sil_scores)]

    c1, c2 = st.columns(2)
    with c1:
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=list(k_range), y=wcss, mode="lines+markers",
                                   line=dict(color=PAL["sec"], width=2.5),
                                   marker=dict(size=8)))
        fig_e.update_layout(title="Elbow Method (WCSS)", xaxis_title="k", yaxis_title="WCSS")
        st.plotly_chart(_layout(fig_e, 320), width="stretch", key="order_elbow")
    with c2:
        fig_s = go.Figure()
        fig_s.add_trace(go.Bar(x=list(k_range), y=sil_scores,
                               marker_color=[PAL["pri"] if k == best_k else PAL["muted"] for k in k_range]))
        fig_s.update_layout(title=f"Silhouette Score (best k={best_k})", xaxis_title="k", yaxis_title="Score")
        st.plotly_chart(_layout(fig_s, 320), width="stretch", key="order_silhouette")

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    order_df["cluster"] = km_final.fit_predict(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    order_df["PC1"] = coords[:, 0]
    order_df["PC2"] = coords[:, 1]

    cluster_colors = [PAL["pri"], PAL["sec"], PAL["ok"], PAL["warn"], PAL["purple"], PAL["teal"]]
    fig_pca = go.Figure()
    for cl in sorted(order_df["cluster"].unique()):
        sub = order_df[order_df["cluster"] == cl]
        fig_pca.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers",
            marker=dict(size=5, color=cluster_colors[cl % len(cluster_colors)], opacity=0.6),
            name=f"Cluster {cl} (n={len(sub):,})",
        ))
    var_exp = pca.explained_variance_ratio_
    fig_pca.update_layout(
        title=f"PCA 2D — Order Clusters (k={best_k}, explained var: {var_exp.sum():.1%})",
        xaxis_title=f"PC1 ({var_exp[0]:.1%})",
        yaxis_title=f"PC2 ({var_exp[1]:.1%})",
    )
    st.plotly_chart(_layout(fig_pca, 500), width="stretch", key="pca_order")

    st.markdown("#### Cluster Profile")
    cluster_profile = order_df.groupby("cluster")[features].mean().round(2)
    cluster_profile.columns = feature_labels

    extra = order_df.groupby("cluster").agg(
        n=("cluster", "size"),
        cancel_rate=("is_cancelled", "mean"),
        combo_pct=("is_combo", "mean"),
    ).round(3)
    extra.columns = ["Orders", "Cancel Rate", "Combo %"]
    profile = pd.concat([extra, cluster_profile], axis=1)

    st.dataframe(profile.style.format({
        "Orders": "{:,.0f}",
        "Cancel Rate": "{:.1%}",
        "Combo %": "{:.1%}",
        "Order Value": "{:,.0f}",
        "Quantity": "{:.1f}",
        "Discount %": "{:.2%}",
        "Order Hour": "{:.1f}",
        "Shipping Fee": "{:,.0f}",
        "Online Payment": "{:.2f}",
    }).background_gradient(cmap="YlOrRd", axis=0), width="stretch")

    st.markdown("#### Order Value Distribution by Cluster")
    fig_box = go.Figure()
    for cl in sorted(order_df["cluster"].unique()):
        sub = order_df[order_df["cluster"] == cl]
        fig_box.add_trace(go.Box(
            y=sub["revenue"],
            name=f"Cluster {cl}",
            marker_color=cluster_colors[cl % len(cluster_colors)],
        ))
    fig_box.update_layout(title="Box Plot — Order Value by Cluster",
                          yaxis_title="Order Value (VND)")
    st.plotly_chart(_layout(fig_box, 400), width="stretch", key="order_boxplot")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filters")

    months = sorted([str(x) for x in df["order_month"].dropna().unique()])
    sel_months = st.sidebar.multiselect("Month", months, default=[])

    statuses = sorted([str(x) for x in df["Trạng Thái Đơn Hàng"].dropna().unique() if isinstance(x, str)])
    sel_status = st.sidebar.multiselect("Status", statuses, default=[])

    payments = sorted([str(x) for x in df["Phương thức thanh toán"].dropna().unique() if isinstance(x, str)])
    sel_pay = st.sidebar.multiselect("Payment", payments, default=[])

    regions = sorted([str(x) for x in df["region"].dropna().unique() if isinstance(x, str)])
    sel_region = st.sidebar.multiselect("Region", regions, default=[])

    combo_opt = st.sidebar.radio("Product Type", ["All", "Combo", "Single"], horizontal=True)

    st.sidebar.markdown("---")
    _val_col = pd.to_numeric(df["revenue"], errors="coerce")
    val_min = int(_val_col.min()) if _val_col.notna().any() else 0
    val_max = int(_val_col.max()) if _val_col.notna().any() else 1000000
    if val_min == val_max:
        val_max = val_min + 1
    val_range = st.sidebar.slider("Order Revenue (VND)", val_min, val_max, (val_min, val_max),
                                   format="%d", key="val_slider")

    m = pd.Series(True, index=df.index)
    if sel_months:
        m &= df["order_month"].isin(sel_months)
    if sel_status:
        m &= df["Trạng Thái Đơn Hàng"].isin(sel_status)
    if sel_pay:
        m &= df["Phương thức thanh toán"].isin(sel_pay)
    if sel_region:
        m &= df["region"].isin(sel_region)
    if combo_opt == "Combo":
        m &= df["is_combo"] == 1
    elif combo_opt == "Single":
        m &= df["is_combo"] == 0
    m &= pd.to_numeric(df["revenue"], errors="coerce").between(*val_range)

    return df[m].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.title("OTAKAMI — Shopee Business Analytics Dashboard")
    st.caption("Data: Oct 2025 – Mar 2026  |  EDA  |  K-Means Clustering  |  Business Analytics")

    df_raw = load_orders()
    if df_raw.empty:
        st.error("No order data found. Please check the data directory.")
        checked = st.session_state.get("_order_checked_paths", [])
        if checked:
            st.caption("Checked these file paths:")
            st.code("\n".join(checked))
        base = Path(__file__).parent
        available = sorted([str(p.name) for p in base.glob("*.xlsx")])
        st.caption("Available .xlsx files in app root:")
        st.code("\n".join(available) if available else "(none)")
        st.stop()

    stats = load_shop_stats()
    traffic = load_traffic_sources()

    df = apply_filters(df_raw)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{len(df):,}** / {len(df_raw):,} orders selected")
    if len(df) < 10:
        st.warning("Too few records after filtering. Please adjust the filters.")
        st.stop()

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Executive Overview",
        "Product Performance",
        "Customer & Operations",
        "Traffic & Conversion",
        "Order Clustering",
        "Data",
    ])

    with t1:
        page_executive(df, stats)
    with t2:
        page_product(df)
    with t3:
        page_customer_ops(df)
    with t4:
        page_traffic(df, stats, traffic)
    with t5:
        page_order_clustering(df)
    with t6:
        st.markdown("### Filtered Data")
        st.html("""
        <div class="concept-tip">
            Raw data after applying filters. Download as CSV for further analysis.
        </div>
        """)
        show_cols = ["Mã đơn hàng", "Ngày đặt hàng", "Trạng Thái Đơn Hàng",
                     "product_short", "Số lượng", "Giá gốc", "Tổng giá bán (sản phẩm)",
                     "revenue", "net_revenue", "Phương thức thanh toán", "Tỉnh/Thành phố", "region",
                     "is_combo", "is_cancelled", "cancel_reason_short",
                     "Đơn Vị Vận Chuyển", "order_month"]
        available_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[available_cols], width="stretch", hide_index=True)
        csv = df[available_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", csv, "otakami_filtered.csv", "text/csv")

    with st.expander("Chart Selection Rationale"):
        st.markdown("""
| # | Chart | Type | RQ | Rationale |
|---|-------|------|----|-----------|
| 1 | KPI Cards | Summary metrics | RQ1 | Quick overview of key business indicators |
| 2 | Revenue & orders trend | Dual-axis bar + line | RQ1 | Monthly growth trends |
| 3 | Order Funnel | Funnel chart | RQ1 | Conversion rate across order stages |
| 4 | Product Pareto | Bar + cumulative line | RQ2 | Identify 20% of products driving 80% of revenue |
| 5 | Combo vs Single | Grouped bar | RQ2 | Compare effectiveness of two product types |
| 6 | Product x KPI heatmap | Heatmap | RQ2 | Multi-dimensional product performance assessment |
| 7 | PCA + K-Means products | Scatter | RQ5 | Product segmentation by performance |
| 8 | Hour x Day heatmap | Heatmap | RQ3 | Identify peak buying times |
| 9 | Geography (Pie + Bar) | Pie + Horizontal bar | RQ3 | Order distribution by region |
| 10 | Cancellation analysis | Horizontal bar | RQ3 | Identify cancellation causes and patterns |
| 11 | Traffic radar | Radar chart | RQ4 | Compare revenue source structure |
| 12 | Traffic stacked | Stacked bar | RQ4 | Revenue trends by traffic source |
| 13 | CVR vs Traffic | Dual-axis | RQ4 | Traffic-conversion relationship |
| 14 | Order K-Means | Scatter + Box | RQ5 | Buying behavior segmentation |
        """)


if __name__ == "__main__":
    main()
