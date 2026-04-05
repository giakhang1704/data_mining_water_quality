import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Water Quality Dashboard",
    page_icon="🌊",
    layout="wide"
)

@st.cache_data
def load_data():
    df_land = pd.read_csv('dataset/WaterQuality_Clustered_Results.csv', low_memory=False)
    df_ocean = pd.read_csv('dataset/OceanQuality_Clustered_Results.csv', low_memory=False)
    df_meta = pd.read_excel('dataset/metadata.xlsx')

    for df in [df_land, df_ocean]:
        if 'Unnamed: 0' in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

    return df_land, df_ocean, df_meta


def prepare_data(df, zero_values=None):
    data = df.copy()
    if 'IndicatorsName' in data.columns and 'Value' in data.columns:
        if zero_values is not None:
            data['Value'] = data['Value'].replace(zero_values)
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
        data = (
            data
            .pivot_table(
                index=['MonitoringLocationIdentifier', 'MonitoringDate'],
                columns='IndicatorsName',
                values='Value',
                aggfunc='mean'
            )
            .reset_index()
        )
    data['MonitoringDate'] = pd.to_datetime(data['MonitoringDate'], format='mixed', dayfirst=True, errors='coerce')
    if 'MonitoringDate' in data.columns:
        data['month'] = data['MonitoringDate'].dt.month
        data['season'] = data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
    data = data.sort_values(['MonitoringLocationIdentifier', 'MonitoringDate']).reset_index(drop=True)
    return data


def get_indicator_columns(data):
    exclude_cols = {'MonitoringLocationIdentifier', 'MonitoringDate', 'month'}
    return [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col]) and not str(col).startswith('cluster_')]


def get_subplot_layout(n):
    """Calculate rows and cols for balanced subplot layout"""
    if n <= 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 6:
        rows = 2
        cols = (n + 1) // 2  # ceil(n/2): 3→2, 4→2, 5→3, 6→3
        return rows, cols
    # Cho n > 6
    nrows = int(np.ceil(np.sqrt(n)))
    ncols = int(np.ceil(n / nrows))
    return nrows, ncols


def make_trend_subplots(data, indicators):
    """Create subplots for time trends of each indicator"""
    if not indicators or data.empty:
        return None
    
    nrows, ncols = get_subplot_layout(len(indicators))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=indicators,
        specs=[[{'secondary_y': False}] * ncols for _ in range(nrows)]
    )
    
    row, col = 1, 1
    for idx, indicator in enumerate(indicators):
        subset = data.dropna(subset=[indicator])
        if not subset.empty:
            # Group by date and calculate mean
            trend = subset.groupby('MonitoringDate')[indicator].mean().reset_index()
            fig.add_trace(
                go.Scatter(
                    x=trend['MonitoringDate'],
                    y=trend[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(width=2)
                ),
                row=row,
                col=col
            )
        
        col += 1
        if col > ncols:
            col = 1
            row += 1
    
    fig.update_layout(height=300 * nrows, showlegend=False, title_text='Xu hướng theo thời gian')
    return fig


def make_histogram_grid(df, indicators):
    """Create histogram subplots with balanced layout"""
    if not indicators or df.empty:
        return None
    
    nrows, ncols = get_subplot_layout(len(indicators))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=indicators)
    
    row, col = 1, 1
    for idx, indicator in enumerate(indicators):
        fig.add_trace(
            go.Histogram(x=df[indicator], name=indicator, marker_color='#1f77b4', nbinsx=40),
            row=row,
            col=col
        )
        col += 1
        if col > ncols:
            col = 1
            row += 1
    
    fig.update_layout(height=300 * nrows, showlegend=False, title_text='Phân phối các chỉ số')
    return fig


def make_boxplot_grid(df, indicators):
    """Create boxplot subplots with balanced layout"""
    if not indicators or df.empty:
        return None
    
    nrows, ncols = get_subplot_layout(len(indicators))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=indicators)
    
    row, col = 1, 1
    for idx, indicator in enumerate(indicators):
        fig.add_trace(
            go.Box(y=df[indicator], name=indicator),
            row=row,
            col=col
        )
        col += 1
        if col > ncols:
            col = 1
            row += 1
    
    fig.update_layout(height=400 * nrows, showlegend=False, title_text='Boxplot các chỉ số')
    return fig


def get_cluster_columns(data):
    return [col for col in data.columns if str(col).startswith('cluster_') and pd.api.types.is_numeric_dtype(data[col])]


def show_seasonal_boxplot(data, indicator):
    if 'month' not in data.columns or 'season' not in data.columns:
        return

    subset = data.dropna(subset=[indicator, 'month', 'season'])
    if subset.empty:
        return

    cols = st.columns(2)
    month_fig = px.box(
        subset,
        x='month',
        y=indicator,
        points='outliers',
        title=f'{indicator} theo tháng',
        labels={'month': 'Tháng', indicator: 'Giá trị'}
    )
    month_fig.update_layout(height=420)
    cols[0].plotly_chart(month_fig, use_container_width=True)

    season_fig = px.box(
        subset,
        x='season',
        y=indicator,
        points='outliers',
        category_orders={'season': ['Winter', 'Spring', 'Summer', 'Autumn']},
        title=f'{indicator} theo mùa',
        labels={'season': 'Mùa', indicator: 'Giá trị'}
    )
    season_fig.update_layout(height=420)
    cols[1].plotly_chart(season_fig, use_container_width=True)


def show_overview(data, meta, title):
    st.header(title)
    st.write("**Tổng quan dữ liệu**")
    indicators = get_indicator_columns(data)
    st.markdown(f"- Số hàng: **{len(data):,}**")
    st.markdown(f"- Số trạm đo: **{data['MonitoringLocationIdentifier'].nunique():,}**")
    st.markdown(f"- Số chỉ số: **{len(indicators)}**")
    st.write("---")

    with st.expander("Xem bảng dữ liệu đầu tiên", expanded=False):
        st.dataframe(data.head(20))

    if len(indicators) == 0:
        st.warning("Không tìm thấy chỉ số số trong dữ liệu để vẽ biểu đồ.")
        return

    cols = st.columns(2)
    cols[0].metric("Chỉ số", len(indicators))
    cols[1].metric("Trạm", f"{data['MonitoringLocationIdentifier'].nunique():,}")

    st.write("### Ma trận tương quan")
    corr = data[indicators].corr()
    corr_fig = px.imshow(
        corr,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        title='Ma trận tương quan giữa các chỉ số'
    )
    st.plotly_chart(corr_fig, use_container_width=True)

    st.write("### Boxplot các chỉ số")
    box_fig = make_boxplot_grid(data[indicators].dropna(), indicators)
    if box_fig:
        st.plotly_chart(box_fig, use_container_width=True)

    st.write("### Histogram và phân phối")
    hist_fig = make_histogram_grid(data[indicators].dropna(), indicators)
    if hist_fig:
        st.plotly_chart(hist_fig, use_container_width=True)

    if 'month' in data.columns and len(indicators) > 0:
        seasonal_choice = st.selectbox("Chọn chỉ số seasonal", indicators, index=0, key=f"seasonal_{title}")
        show_seasonal_boxplot(data, seasonal_choice)

def show_station_details(data, meta, title):
    st.header(f"Chi tiết trạm: {title}")
    station_ids = data['MonitoringLocationIdentifier'].unique()
    station_id = st.selectbox("Chọn mã trạm", station_ids)
    station_data = data[data['MonitoringLocationIdentifier'] == station_id].copy()

    if station_data.empty:
        st.warning("Không có dữ liệu cho trạm này.")
        return

    station_meta = meta[meta['MonitoringLocationIdentifier'] == station_id]
    if not station_meta.empty:
        st.write("**Thông tin trạm**")
        # Chỉ hiển thị 4 cột chỉ đị
        cols_to_show = ['MonitoringLocationIdentifier', 'MonitoringLocationName', 'MonitoringLocationType', 'MonitoringLocationTypeName']
        display_meta = station_meta[[col for col in cols_to_show if col in station_meta.columns]].reset_index(drop=True)
        st.dataframe(display_meta)

    indicators = get_indicator_columns(station_data)
    st.write(f"**Số bản ghi trạm:** {len(station_data):,}")
    st.write("---")

    st.write("### Xu hướng theo thời gian")
    trend_fig = make_trend_subplots(station_data, indicators)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)

    st.write("### Boxplot và histogram trạm")
    if len(indicators) > 0:
        box_fig = make_boxplot_grid(station_data[indicators].dropna(), indicators)
        if box_fig:
            st.plotly_chart(box_fig, use_container_width=True)

        hist_fig = make_histogram_grid(station_data[indicators].dropna(), indicators)
        if hist_fig:
            st.plotly_chart(hist_fig, use_container_width=True)


def main():
    df_land, df_ocean, df_meta = load_data()
    st.title("Water Quality Dashboard")
    st.write("Dashboard này hiển thị phân tích tổng quan và chi tiết trên dữ liệu nước mặt và dữ liệu biển.")

    dataset_type = st.sidebar.radio(
        "Chọn bộ dữ liệu",
        ('Dữ liệu nước mặt', 'Dữ liệu biển')
    )

    if dataset_type == 'Dữ liệu nước mặt':
        data = prepare_data(df_land)
        show_overview(data, df_meta, "Tổng quan dữ liệu nước mặt")
        show_station_details(data, df_meta, "Dữ liệu nước mặt")
    else:
        data = prepare_data(df_ocean, zero_values={'< DL': 0})
        show_overview(data, df_meta, "Tổng quan dữ liệu biển")
        show_station_details(data, df_meta, "Dữ liệu biển")

    st.sidebar.markdown("---")


if __name__ == '__main__':
    main()
