import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
DATA_FILE = "Airplane_Crashes_and_Fatalities_Since_1908_Cleaned.csv"
data = pd.read_csv(DATA_FILE)

# Calculate statistics
num_crashes = len(data)
total_fatalities = data['Fatalities'].sum()
total_ground_fatalities = data['Ground'].sum()

# Remove Streamlit title, add spacing at the top
st.set_page_config(page_title="SkySafe Regional Crash Dashboard", layout="wide")
st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

# Calculate survival outcome counts
survived_none = (data['SurvivalSeverity'] == 'None Survived').sum() if 'SurvivalSeverity' in data.columns else 0
survived_partial = (data['SurvivalSeverity'] == 'Partial Survived').sum() if 'SurvivalSeverity' in data.columns else 0
survived_all = (data['SurvivalSeverity'] == 'All Survived').sum() if 'SurvivalSeverity' in data.columns else 0

# Display all stats in a single row


# Mosaic layout: stats and plots left, pie chart right
left_col, right_col = st.columns([2.5, 1.2])
with left_col:
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown(
            f"""
            <div style='background: #1976d2; border-radius: 12px; padding: 1.2rem 0.5rem; text-align: center;'>
                <div style='font-size:1.3rem; font-weight:700; color:#fff; margin-bottom:0.5rem;'>Crashes</div>
                <div style='font-size:1.7rem; font-weight:800; color:#fff;'>{num_crashes:,}</div>
            </div>
            """, unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(
            f"""
            <div style='background: #1565c0; border-radius: 12px; padding: 1.2rem 0.5rem; text-align: center;'>
                <div style='font-size:1.3rem; font-weight:700; color:#fff; margin-bottom:0.5rem;'>Fatalities</div>
                <div style='font-size:1.7rem; font-weight:800; color:#fff;'>{total_fatalities:,}</div>
            </div>
            """, unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown(
            f"""
            <div style='background: #0d47a1; border-radius: 12px; padding: 1.2rem 0.5rem; text-align: center;'>
                <div style='font-size:1.3rem; font-weight:700; color:#fff; margin-bottom:0.5rem;'>Ground Fatalities</div>
                <div style='font-size:1.7rem; font-weight:800; color:#fff;'>{total_ground_fatalities:,}</div>
            </div>
            """, unsafe_allow_html=True)
    # Line plots directly below stat cards
    st.markdown("<div style='height: 0.3rem'></div>", unsafe_allow_html=True)
    if 'Year' in data.columns:
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            crashes_per_year = data.groupby('Year').size().reset_index(name='Crashes')
            fig1 = px.line(
                crashes_per_year,
                x='Year',
                y='Crashes',
                markers=True,
                title='Number of Crashes Per Year',
                labels={'Year': 'Year', 'Crashes': 'Number of Crashes'},
                hover_data={'Year': True, 'Crashes': True}
            )
            fig1.update_traces(marker=dict(size=8, color='#1976d2', line=dict(width=2, color='#fff')),
                              line=dict(color='#1976d2', width=3))
            fig1.update_layout(
                hovermode='x unified',
                plot_bgcolor='#f4f6fb',
                paper_bgcolor='#f4f6fb',
                font=dict(size=15),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig1, use_container_width=True, key="crashes_per_year_plot")
        with col_plot2:
            if 'SurvivalSeverity' in data.columns:
                severity_per_year = data.groupby(['Year', 'SurvivalSeverity']).size().reset_index(name='Count')
                severity_color_map = {
                    'None Survived': '#d32f2f',
                    'Partial Survived': '#43a047',
                    'All Survived': '#1976d2'
                }
                present_severities = severity_per_year['SurvivalSeverity'].unique()
                color_seq = [severity_color_map.get(sev, '#bdbdbd') for sev in present_severities]
                fig2 = px.line(
                    severity_per_year,
                    x='Year',
                    y='Count',
                    color='SurvivalSeverity',
                    markers=True,
                    title='Survival Severity Trends Over Years',
                    labels={'Year': 'Year', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
                    hover_data={'Year': True, 'Count': True, 'SurvivalSeverity': True},
                    category_orders={'SurvivalSeverity': list(severity_color_map.keys())},
                    color_discrete_map=severity_color_map
                )
                fig2.update_traces(marker=dict(size=7, line=dict(width=1.5, color='#fff')))
                fig2.update_layout(
                    hovermode='x unified',
                    plot_bgcolor='#f4f6fb',
                    paper_bgcolor='#f4f6fb',
                    font=dict(size=15),
                    margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False
                )
                st.plotly_chart(fig2, use_container_width=True, key="survival_severity_trend_plot")
            else:
                st.info("No 'SurvivalSeverity' column found in the dataset for plotting.")
        # --- Map plot directly under line charts ---
        st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
        map_data = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908_Cleaned_with_latlong.csv')
        map_data = map_data.dropna(subset=['latitude', 'longitude'])
        map_data = map_data[map_data['Country'].str.lower() != 'unknown']
        severity_color_map = {
            'None Survived': '#d32f2f',
            'Partial Survived': '#43a047',
            'All Survived': '#1976d2'
        }
        if not map_data.empty:
            fig_map = px.scatter_mapbox(
                map_data,
                lat='latitude',
                lon='longitude',
                color='SurvivalSeverity',
                color_discrete_map=severity_color_map,
                hover_name='Country',
                hover_data=['Year', 'Operator', 'Fatalities'],
                zoom=0.5,
                height=700,
                width=800,
                title='Crash Locations by Survival Severity'
            )
            fig_map.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0,"t":40,"l":0,"b":0},
                showlegend=False,
                font=dict(size=15),
                paper_bgcolor='#f4f6fb',
                plot_bgcolor='#f4f6fb'
            )
            st.plotly_chart(fig_map, use_container_width=False, key='crash_map')
            # Place weather and dayperiod plots directly under map, side by side
            st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
            col_weather, col_dayperiod = st.columns([1, 1])
            with col_weather:
                if 'WeatherCondition' in data.columns and 'SurvivalSeverity' in data.columns:
                    filtered_weather = data[data['WeatherCondition'].str.lower() != 'unknown']
                    weather_severity = filtered_weather.groupby(['WeatherCondition', 'SurvivalSeverity']).size().reset_index(name='Count')
                    weather_types = sorted(filtered_weather['WeatherCondition'].dropna().unique())
                    fig_weather_grouped = px.bar(
                        weather_severity,
                        x='WeatherCondition',
                        y='Count',
                        color='SurvivalSeverity',
                        barmode='group',
                        color_discrete_map=severity_color_map,
                        category_orders={
                            'SurvivalSeverity': list(severity_color_map.keys()),
                            'WeatherCondition': weather_types
                        },
                        labels={'WeatherCondition': 'Weather Condition', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
                        title='Survival Severity by Weather Condition'
                    )
                    fig_weather_grouped.update_layout(
                        font=dict(size=13),
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='#f4f6fb',
                        paper_bgcolor='#f4f6fb',
                        showlegend=False,
                        height=350,
                        width=350
                    )
                    st.plotly_chart(fig_weather_grouped, use_container_width=False, key='weathercondition_severity_groupedbar')
                else:
                    st.info("WeatherCondition or SurvivalSeverity column not found in the dataset.")
            with col_dayperiod:
                if 'DayPeriod' in data.columns and 'SurvivalSeverity' in data.columns:
                    filtered_dayperiod = data[data['DayPeriod'].str.lower() != 'unknown']
                    dayperiod_severity = filtered_dayperiod.groupby(['DayPeriod', 'SurvivalSeverity']).size().reset_index(name='Count')
                    fig_dayperiod_line = px.line(
                        dayperiod_severity,
                        x='DayPeriod',
                        y='Count',
                        color='SurvivalSeverity',
                        markers=True,
                        color_discrete_map=severity_color_map,
                        category_orders={
                            'SurvivalSeverity': list(severity_color_map.keys())
                        },
                        labels={'DayPeriod': 'Day Period', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
                        title='Survival Severity by Day Period'
                    )
                    fig_dayperiod_line.update_traces(marker=dict(size=8, line=dict(width=1.5, color='#fff')))
                    fig_dayperiod_line.update_layout(
                        font=dict(size=13),
                        margin=dict(l=10, r=10, t=30, b=10),
                        plot_bgcolor='#f4f6fb',
                        paper_bgcolor='#f4f6fb',
                        showlegend=False,
                        height=350,
                        width=350
                    )
                    st.plotly_chart(fig_dayperiod_line, use_container_width=False, key='dayperiod_severity_linechart')
                else:
                    st.info("DayPeriod or SurvivalSeverity column not found in the dataset.")
        else:
            st.info("No valid crash locations found to plot on the map.")
    else:
        st.info("No 'Year' column found in the dataset for plotting.")

with right_col:
    import plotly.graph_objects as go
    pie_labels = ['None Survived', 'Partial Survived', 'All Survived']
    pie_values = [survived_none, survived_partial, survived_all]
    pie_colors = ['#d32f2f', '#43a047', '#1976d2']
    fig_pie = go.Figure(data=[go.Pie(
        labels=pie_labels,
        values=pie_values,
        marker=dict(colors=pie_colors),
        textinfo='label+percent',
        insidetextorientation='auto'
    )])
    fig_pie.update_layout(
        title_text='Survival Outcomes Distribution',
        showlegend=True,
        legend=dict(orientation='h', y=-0.15, x=0.1),
        margin=dict(t=40, b=0, l=0, r=0),
        font=dict(size=15),
        height=420,
        width=420
    )
    st.plotly_chart(fig_pie, use_container_width=False, key="survival_outcomes_pie_chart")

# Plots: Number of crashes per year and Survival Severity trends
# --- SurvivalSeverity vs FlightPhase Bar Chart ---

# --- Mosaic layout for map and bar chart ---
# Prepare map data and color map for mosaic layout
map_data = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908_Cleaned_with_latlong.csv')
map_data = map_data.dropna(subset=['latitude', 'longitude'])
# Remove crashes where country is unknown
map_data = map_data[map_data['Country'].str.lower() != 'unknown']
severity_color_map = {
    'None Survived': '#d32f2f',   # red
    'Partial Survived': '#43a047', # green
    'All Survived': '#1976d2'      # blue
}
st.markdown("<div style='height: 2.2rem'></div>", unsafe_allow_html=True)
map_col, bar_col = st.columns([2, 1])
with map_col:
    # ...existing code...
    pass

    # ...existing code...
with right_col:
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    if 'FlightPhase' in data.columns and 'SurvivalSeverity' in data.columns:
        # Remove 'Unknown' category from FlightPhase
        filtered_data = data[data['FlightPhase'].str.lower() != 'unknown']
        phase_severity = filtered_data.groupby(['FlightPhase', 'SurvivalSeverity']).size().reset_index(name='Count')
        severity_color_map = {
            'None Survived': '#d32f2f',
            'Partial Survived': '#43a047',
            'All Survived': '#1976d2'
        }
        flightphases = sorted(filtered_data['FlightPhase'].dropna().unique())
        fig_bar = px.bar(
            phase_severity,
            x='FlightPhase',
            y='Count',
            color='SurvivalSeverity',
            barmode='group',
            color_discrete_map=severity_color_map,
            category_orders={
                'SurvivalSeverity': list(severity_color_map.keys()),
                'FlightPhase': flightphases
            },
            labels={'FlightPhase': 'Flight Phase', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
            title='Survival Severity by Flight Phase'
        )
        fig_bar.update_layout(
            font=dict(size=15),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='#f4f6fb',
            paper_bgcolor='#f4f6fb',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True, key='flightphase_severity_bar')
    else:
        st.info("FlightPhase or SurvivalSeverity column not found in the dataset.")

    # SurvivalSeverity vs LocationType stacked bar chart below FlightPhase plot
    st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
    if 'LocationType' in data.columns and 'SurvivalSeverity' in data.columns:
        filtered_loc = data[data['LocationType'].str.lower() != 'unknown']
        loc_severity = filtered_loc.groupby(['LocationType', 'SurvivalSeverity']).size().reset_index(name='Count')
        location_types = sorted(filtered_loc['LocationType'].dropna().unique())
        fig_stacked = px.bar(
            loc_severity,
            x='LocationType',
            y='Count',
            color='SurvivalSeverity',
            barmode='stack',
            color_discrete_map=severity_color_map,
            category_orders={
                'SurvivalSeverity': list(severity_color_map.keys()),
                'LocationType': location_types
            },
            labels={'LocationType': 'Location Type', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
            title='Survival Severity by Location Type'
        )
        fig_stacked.update_layout(
            font=dict(size=15),
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='#f4f6fb',
            paper_bgcolor='#f4f6fb',
            showlegend=False
        )
        st.plotly_chart(fig_stacked, use_container_width=True, key='locationtype_severity_stackedbar')
    else:
        st.info("LocationType or SurvivalSeverity column not found in the dataset.")

    # SurvivalSeverity vs CauseCategory grouped bar chart directly under LocationType
    st.markdown("<div style='height: 1.2rem'></div>", unsafe_allow_html=True)
    if 'CauseCategory' in data.columns and 'SurvivalSeverity' in data.columns:
        filtered_cause = data[data['CauseCategory'].str.lower() != 'unknown']
        cause_severity = filtered_cause.groupby(['CauseCategory', 'SurvivalSeverity']).size().reset_index(name='Count')
        cause_categories = sorted(filtered_cause['CauseCategory'].dropna().unique())
        fig_cause_grouped = px.bar(
            cause_severity,
            x='CauseCategory',
            y='Count',
            color='SurvivalSeverity',
            barmode='group',
            color_discrete_map=severity_color_map,
            category_orders={
                'SurvivalSeverity': list(severity_color_map.keys()),
                'CauseCategory': cause_categories
            },
            labels={'CauseCategory': 'Cause Category', 'Count': 'Number of Crashes', 'SurvivalSeverity': 'Survival Severity'},
            title='Survival Severity by Cause Category'
        )
        fig_cause_grouped.update_layout(
            font=dict(size=13),
            margin=dict(l=10, r=10, t=30, b=10),
            plot_bgcolor='#f4f6fb',
            paper_bgcolor='#f4f6fb',
            showlegend=False,
            height=350,
            width=700
        )
        st.plotly_chart(fig_cause_grouped, use_container_width=True, key='causecategory_severity_groupedbar_rightcol')
    else:
        st.info("CauseCategory or SurvivalSeverity column not found in the dataset.")

    # ...existing code...
# --- Interactive Global Map of Crash Locations ---


