import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="Water Quality Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching to prevent redundant I/O operations on app rerun
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
    
    # Pivot dataset from long to wide format to align indicators as features for modeling/visualization
    if 'IndicatorsName' in data.columns and 'Value' in data.columns:
        if zero_values is not None:
            data['Value'] = data['Value'].replace(zero_values)
        data['Value'] = pd.to_numeric(data['Value'], errors='coerce')
        data = (
            data.pivot_table(
                index=['MonitoringLocationIdentifier', 'MonitoringDate'],
                columns='IndicatorsName',
                values='Value',
                aggfunc='mean'
            ).reset_index()
        )
    
    data['MonitoringDate'] = pd.to_datetime(data['MonitoringDate'], format='mixed', dayfirst=True, errors='coerce')
    
    # Extract temporal features for seasonal trend analysis
    if 'MonitoringDate' in data.columns:
        data['YearMonth'] = data['MonitoringDate'].dt.strftime('%Y-%m') 
        data['month'] = data['MonitoringDate'].dt.month
        data['season'] = data['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
    return data.sort_values(['MonitoringLocationIdentifier', 'MonitoringDate']).reset_index(drop=True)

def get_indicator_columns(data):
    exclude_cols = {'MonitoringLocationIdentifier', 'MonitoringDate', 'month', 'YearMonth', 'season'}
    return [col for col in data.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(data[col]) and not str(col).startswith('cluster_')]

# Calculate optimal grid dimensions for subplots based on total items
def get_subplot_layout(n):
    if n <= 0: return 1, 1
    if n <= 2: return 1, n
    if n <= 6: return 2, math.ceil(n/2)
    nrows = int(np.ceil(np.sqrt(n)))
    return nrows, int(np.ceil(n / nrows))

def make_trend_subplots(data, indicators):
    if not indicators or data.empty: return None
    nrows, ncols = get_subplot_layout(len(indicators))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=indicators)
    
    row, col = 1, 1
    for indicator in indicators:
        subset = data.dropna(subset=[indicator])
        if not subset.empty:
            trend = subset.groupby('MonitoringDate')[indicator].mean().reset_index()
            fig.add_trace(
                go.Scatter(x=trend['MonitoringDate'], y=trend[indicator], mode='lines', name=indicator, line=dict(width=2)),
                row=row, col=col
            )
        col += 1
        if col > ncols:
            col, row = 1, row + 1
            
    fig.update_layout(height=300 * nrows, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    return fig

def make_distribution_plots(df, indicators, plot_type='histogram'):
    if not indicators or df.empty: return None
    nrows, ncols = get_subplot_layout(len(indicators))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=indicators)
    
    row, col = 1, 1
    for indicator in indicators:
        if plot_type == 'histogram':
            fig.add_trace(go.Histogram(x=df[indicator], marker_color='#1f77b4', nbinsx=50), row=row, col=col)
        else:
            fig.add_trace(go.Box(y=df[indicator], marker_color='#ff7f0e'), row=row, col=col)
        
        col += 1
        if col > ncols:
            col, row = 1, row + 1
            
    fig.update_layout(height=300 * nrows, showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    return fig

def make_correlation_heatmap(data, indicators):
    if len(indicators) < 2 or data.empty: 
        return None
    corr_matrix = data[indicators].corr()
    fig = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto", 
        color_continuous_scale="RdBu_r", 
        zmin=-1, zmax=1
    )
    fig.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
    return fig

def show_seasonal_boxplot(data, indicator):
    if 'month' not in data.columns or 'season' not in data.columns: return
    subset = data.dropna(subset=[indicator, 'month', 'season'])
    if subset.empty: return

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.box(subset, x='month', y=indicator, points='outliers', title=f'{indicator} Distribution by Month'), use_container_width=True)
    with col2:
        st.plotly_chart(px.box(subset, x='season', y=indicator, points='outliers', category_orders={'season': ['Winter', 'Spring', 'Summer', 'Autumn']}, title=f'{indicator} Distribution by Season'), use_container_width=True)

# Diagnostic module: Analyzes feature impact on a user-selected target variable using Pearson correlation and grouped scatter matrices
def show_target_analysis(data, indicators, cluster_col='cluster_kmeans_3'):
    if len(indicators) < 2: return
    
    st.markdown("Select a **Target Indicator** to deeply analyze how other variables influence it:")
    target_var = st.selectbox("Target Variable:", indicators, index=indicators.index('DO') if 'DO' in indicators else 0)

    corr_matrix = data[indicators].corr()
    target_corr = corr_matrix[[target_var]].drop(target_var).sort_values(by=target_var, ascending=True)

    col_bar, col_scatter = st.columns([1, 2])
    
    with col_bar:
        st.markdown(f"**Correlation Coefficient with {target_var}**")
        fig_corr = px.bar(
            target_corr, x=target_var, y=target_corr.index, orientation='h',
            color=target_var, color_continuous_scale='RdBu', range_color=[-1, 1],
            labels={target_var: 'Pearson Correlation', 'index': 'Indicator'}
        )
        fig_corr.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=350)
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_scatter:
        st.markdown(f"**Scatter Matrix Analysis (Impact on {target_var})**")
        other_vars = [v for v in indicators if v != target_var]
        nrows = math.ceil(len(other_vars) / 2)
        fig_scatter = make_subplots(rows=nrows, cols=2, subplot_titles=[f"{v} vs {target_var}" for v in other_vars])

        row, col = 1, 1
        color_map = {'0': '#28a745', '1': '#ffc107', '2': '#dc3545'}
        for var in other_vars:
            if cluster_col in data.columns:
                for c_val, c_color in color_map.items():
                    subset = data[data[cluster_col].astype(str) == c_val]
                    fig_scatter.add_trace(go.Scatter(x=subset[var], y=subset[target_var], mode='markers', marker=dict(color=c_color, opacity=0.5, size=4), name=f"Cluster {c_val}", showlegend=(row==1 and col==1)), row=row, col=col)
            else:
                fig_scatter.add_trace(go.Scatter(x=data[var], y=data[target_var], mode='markers', marker=dict(color='#1f77b4', opacity=0.5, size=4)), row=row, col=col)

            fig_scatter.update_xaxes(title_text=var, row=row, col=col)
            fig_scatter.update_yaxes(title_text=target_var, row=row, col=col)

            col += 1
            if col > 2: col, row = 1, row + 1

        fig_scatter.update_layout(height=350 * nrows, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

def show_threshold_alerts(data, indicators):
    """Automated Early Warning System based on predefined safety thresholds"""
    # Define safety limits (Can be adjusted based on actual environmental standards)
    thresholds = {}
    if 'DO' in indicators: thresholds['DO'] = {'limit': 4.0, 'type': 'min', 'name': 'DO (Oxygen Depletion)'}
    if 'COD' in indicators: thresholds['COD'] = {'limit': 15.0, 'type': 'max', 'name': 'COD (Upper Limit Exceeded)'}
    if 'NH4N' in indicators: thresholds['NH4N'] = {'limit': 0.5, 'type': 'max', 'name': 'NH4N (Ammonia Pollution)'}

    alerts = []
    
    for ind, rule in thresholds.items():
        if rule['type'] == 'max':
            violation = data[data[ind] > rule['limit']]
        else:
            violation = data[data[ind] < rule['limit']]
            
        if not violation.empty:
            alerts.append({
                'indicator': rule['name'],
                'count': len(violation),
                'worst_val': violation[ind].max() if rule['type'] == 'max' else violation[ind].min(),
                'limit': rule['limit']
            })

    if not alerts:
        st.success("System Status: Normal. No threshold violations detected in the current dataset.")
    else:
        st.error(f"AUTOMATED ALERT: {len(alerts)} INDICATORS EXCEEDED SAFE THRESHOLDS!")
        
        # Display alert cards
        alert_cols = st.columns(len(alerts))
        for idx, alert in enumerate(alerts):
            with alert_cols[idx]:
                st.warning(
                    f"**{alert['indicator']}**\n\n"
                    f"- Violation Count: **{alert['count']}** records\n"
                    f"- Most Extreme Value: **{alert['worst_val']:.2f}** (Threshold: {alert['limit']})"
                )

# Spatiotemporal mapping component with dynamic basemap and cross-filtering support
def render_interactive_map(df_data, df_meta, map_style):
    meta_coords = df_meta[[
        'MonitoringLocationIdentifier',
        'MonitoringLocationName',
        'LatitudeMeasure_WGS84',
        'LongitudeMeasure_WGS84'
    ]].drop_duplicates().rename(
        columns={'LatitudeMeasure_WGS84': 'lat', 'LongitudeMeasure_WGS84': 'lon', 'MonitoringLocationName': 'StationName'}
    )

    cluster_col = 'cluster_kmeans_3'
    if cluster_col not in df_data.columns:
        st.warning(f"Column '{cluster_col}' not found. Map will display without cluster colors.")
        cluster_col = None

    # Fetch latest known status per station to avoid map overlapping for historical data
    df_map = df_data.sort_values('MonitoringDate').drop_duplicates(subset=['MonitoringLocationIdentifier'], keep='last')
    df_map = df_map[['MonitoringLocationIdentifier'] + ([cluster_col] if cluster_col else [])].merge(meta_coords, on='MonitoringLocationIdentifier', how='inner')

    if df_map.empty or 'lat' not in df_map.columns or 'lon' not in df_map.columns:
        st.warning("Current dataset does not contain station coordinates.")
        return None

    if cluster_col:
        df_map[cluster_col] = df_map[cluster_col].astype(str)

    fig = px.scatter_mapbox(
        df_map, lat='lat', lon='lon', hover_name='StationName',
        hover_data={'MonitoringLocationIdentifier': True, cluster_col: True} if cluster_col else {'MonitoringLocationIdentifier': True},
        color=cluster_col if cluster_col else None,
        color_discrete_map={'0': '#28a745', '1': '#ffc107', '2': '#dc3545'},
        size=[12] * len(df_map),  # Increase point size for better visibility
        size_max=12,  # Maximum size for the points
        zoom=4, height=500,
    )

    # Dynamic raster tile configuration based on user selection
    if map_style == "esri":
        fig.update_layout(
            mapbox_style="white-bg",
            mapbox_layers=[{"below": 'traces', "sourcetype": "raster", "sourceattribution": "Esri", "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"]}],
            margin={'r':0,'t':0,'l':0,'b':0}
        )
    else:
        fig.update_layout(mapbox_style=map_style, margin={'r':0,'t':0,'l':0,'b':0})

    map_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    return map_event.selection['points'][0]['customdata'][0] if map_event and len(map_event.selection['points']) > 0 else None

def main():
    df_land, df_ocean, df_meta = load_data()
    
    st.sidebar.title("Dashboard Options")
    st.sidebar.markdown("---")
    
    dataset_type = st.sidebar.radio("Monitoring Environment:", ('Surface Water Data', 'Ocean Data'))
    data = prepare_data(df_land) if dataset_type == 'Surface Water Data' else prepare_data(df_ocean, zero_values={'< DL': 0})
    title_prefix = "Surface Water" if dataset_type == 'Surface Water Data' else "Ocean Water"

    st.sidebar.markdown("---")
    view_mode = st.sidebar.radio("Spatial Scope:", ('System-wide', 'By Region'))
    filtered_data = data.copy()
    current_meta = df_meta.copy()

    # Spatial hierarchy filtering
    if view_mode == 'By Region' and 'MonitoringLocationTypeName' in df_meta.columns:
        selected_types = st.sidebar.multiselect("Select Location Types:", options=sorted(df_meta['MonitoringLocationTypeName'].dropna().unique().tolist()))
        if selected_types:
            current_meta = df_meta[df_meta['MonitoringLocationTypeName'].isin(selected_types)]
            filtered_data = filtered_data[filtered_data['MonitoringLocationIdentifier'].isin(current_meta['MonitoringLocationIdentifier'].unique())].copy()
        else:
            filtered_data = pd.DataFrame()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Time Filter (Timeline)**")
    if not filtered_data.empty and 'YearMonth' in filtered_data.columns:
        selected_month = st.sidebar.select_slider(
            "Slide to view historical data:", 
            options=["All Time"] + sorted(filtered_data['YearMonth'].dropna().unique().tolist()),
            value="All Time"
        )
        if selected_month != "All Time":
            filtered_data = filtered_data[filtered_data['YearMonth'] == selected_month]

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Map Display Options**")
    map_style_option = st.sidebar.selectbox("Select Basemap Style:", ("Carto Positron (Light)", "OpenStreetMap (Standard)", "Esri Street (Map)"))
    selected_map_style = {"Carto Positron (Light)": "carto-positron", "OpenStreetMap (Standard)": "open-street-map", "Esri Street (Map)": "esri"}[map_style_option]

    st.title(f"Water Quality Analysis System - {title_prefix}")
    
    if filtered_data.empty:
        st.markdown("*No data available for the selected filters.*")
        return

    st.subheader("Spatial Clustering Map")
    st.markdown(f"*Map showing {'the **latest** known status' if selected_month == 'All Time' else f'data for **{selected_month}**'}.*")
        
    clicked_station = render_interactive_map(filtered_data, current_meta, map_style=selected_map_style)
    st.markdown("---")

    # Map cross-filtering logic: Update dashboard state based on clicked marker
    if clicked_station:
        st.markdown(f"**Display Status:** Detailed view for station **{clicked_station}**")
        display_data = data[data['MonitoringLocationIdentifier'] == clicked_station].copy()
        
        st.subheader("Station Information & Monitoring History")
        
        station_meta = df_meta[df_meta['MonitoringLocationIdentifier'] == clicked_station]
        cols_to_show = ['MonitoringLocationIdentifier', 'MonitoringLocationName', 'MonitoringLocationType', 'MonitoringLocationTypeName']
        st.markdown("**Administrative Information:**")
        st.dataframe(station_meta[[col for col in cols_to_show if col in station_meta.columns]].reset_index(drop=True), use_container_width=True, hide_index=True)

        indicators = get_indicator_columns(display_data)
        st.markdown("**Monitoring Records:**")
        if not display_data.empty:
            cluster_col = 'cluster_kmeans_3'
            hist_cols = ['MonitoringDate'] + ([cluster_col] if cluster_col in display_data.columns else []) + indicators
            hist_df = display_data[hist_cols].sort_values('MonitoringDate', ascending=False)
            hist_df['MonitoringDate'] = hist_df['MonitoringDate'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(hist_df, use_container_width=True, hide_index=True, height=250)
            
    else:
        st.markdown(f"**Display Status:** Aggregated data ({view_mode})")
        display_data = filtered_data.copy()

    if display_data.empty: return
    indicators = get_indicator_columns(display_data)

    st.markdown("---")
    st.subheader("1. Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(display_data):,}")
    col2.metric("Number of Stations", f"{display_data['MonitoringLocationIdentifier'].nunique():,}")
    col3.metric("Number of Chemical Indicators", len(indicators))

    
    st.markdown("---")
    st.subheader("2. Time Trends")
    trend_fig = make_trend_subplots(display_data, indicators)
    if trend_fig: st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Statistical Distribution")
    dist_type = st.radio("Choose plot type:", ("Boxplot", "Histogram"), horizontal=True)
    plot_func = 'boxplot' if dist_type == "Boxplot" else 'histogram'
    dist_fig = make_distribution_plots(display_data, indicators, plot_type=plot_func)
    if dist_fig: st.plotly_chart(dist_fig, use_container_width=True)

    if len(indicators) > 1:
        st.markdown("---")
        st.subheader("4. Correlation Heatmap")
        heatmap_fig = make_correlation_heatmap(display_data, indicators)
        if heatmap_fig: st.plotly_chart(heatmap_fig, use_container_width=True)

    if 'month' in display_data.columns and len(indicators) > 0:
        st.markdown("---")
        st.subheader("5. Seasonal Analysis")
        show_seasonal_boxplot(display_data, st.selectbox("Choose indicator:", indicators))

    st.markdown("---")
    st.subheader("6. Diagnostic Analysis")
    show_target_analysis(display_data, indicators, 'cluster_kmeans_3')

if __name__ == '__main__':
    main()