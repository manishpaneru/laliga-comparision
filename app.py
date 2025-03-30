import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt
from datetime import datetime
import time
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="La Liga Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for luxury design
st.markdown("""
<style>
    /* Main background color */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #3A86FF !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #3A86FF;
    }
    
    .metric-label {
        font-size: 14px;
        color: #A3ABB2;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1E2130;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        color: #A3ABB2;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3A86FF !important;
        color: white !important;
    }
    
    /* Card container */
    .card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
    }
    
    /* Loader */
    .stSpinner > div > div {
        border-color: #3A86FF !important;
    }
    
    /* Selector boxes */
    div[data-baseweb="select"] {
        background-color: #1E2130;
        border-radius: 6px;
    }
    
    /* Slider */
    .stSlider > div {
        color: #3A86FF;
    }
    
    /* Dividers */
    hr {
        border-color: #3A86FF;
        margin: 30px 0;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #252A37 !important;
        border-radius: 6px;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        padding: 10px 15px !important;
        border-left: 4px solid #3A86FF !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: #2C3344 !important;
    }

    /* Improve overall text readability */
    p, label, div {
        color: #E0E0E0 !important;
    }

    /* Make links more visible */
    a {
        color: #3A86FF !important;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }

    /* Improve code and preformatted text */
    code, pre {
        background-color: #252A37 !important;
        color: #E0E0E0 !important;
    }

    /* Improve selectbox and multiselect readability */
    .stSelectbox label, .stMultiSelect label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Improve table readability */
    .dataframe {
        color: #E0E0E0 !important;
    }

    .dataframe th {
        background-color: #252A37 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    .dataframe td {
        background-color: #1E2130 !important;
        color: #E0E0E0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("LaLiga_Matches.csv")
    
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Extract year and month from date
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # Create season start year
    df['Season_Start_Year'] = df['Season'].apply(lambda x: int(x.split('-')[0]))
    
    return df

# Loading animation
with st.spinner('Loading La Liga Dashboard...'):
    data = load_data()
    time.sleep(1)  # Simulate loading time

# Get unique seasons and teams
seasons = sorted(data['Season'].unique())
teams = sorted(data['HomeTeam'].unique())

# Main header with luxury design
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #0E1117, #1E2130, #0E1117); border-radius: 15px; margin-bottom: 30px; box-shadow: 0 8px 16px rgba(0,0,0,0.3);">
    <h1 style="color: white !important; font-size: 48px; margin-bottom: 10px; font-weight: 700;">⚽ LA LIGA DASHBOARD</h1>
    <p style="color: #A3ABB2; font-size: 20px; font-weight: 300;">Interactive Analytics & Insights | 1995-2024</p>
</div>
""", unsafe_allow_html=True)

# Dashboard description
col1, col2, col3 = st.columns([1,4,1])
with col2:
    st.markdown("""
    <div style="background-color: #252A37; padding: 15px; border-radius: 10px; margin-bottom: 30px; border-left: 5px solid #3A86FF;">
        <p style="color: #FFFFFF; font-size: 16px; margin: 0; line-height: 1.5;">
            Explore 29 seasons of La Liga data through interactive visualizations. Analyze team performances, 
            match outcomes, goal trends, and more. Use the tabs below to navigate between different analysis sections.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Creator information in an accordion
with st.expander("👨‍💻 About the Creator", expanded=False):
    # Profile section with two columns
    profile_col1, profile_col2 = st.columns([1, 3])
    
    with profile_col1:
        # You can replace this with an actual image if you have one
        st.markdown("### 👤")
    
    with profile_col2:
        st.markdown("<h3 style='color: #FFFFFF; margin-bottom: 5px;'>Manish Paneru</h3>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; font-size: 16px; font-weight: bold; margin-bottom: 5px;'>Data Analyst & Visualization Specialist</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #E0E0E0; margin-bottom: 15px;'>Experienced in transforming complex data into actionable insights through interactive dashboards and visualizations.</p>", unsafe_allow_html=True)
    
    # Skills section
    st.markdown("<hr style='margin: 15px 0; border-color: #3A86FF;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFFFFF; margin: 10px 0;'>Skills & Expertise</h3>", unsafe_allow_html=True)
    
    skill_col1, skill_col2, skill_col3 = st.columns(3)
    
    with skill_col1:
        st.markdown("<div style='background-color: #252A37; padding: 15px; border-radius: 8px; border-left: 3px solid #3A86FF;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #3A86FF; font-weight: bold; margin-bottom: 8px;'>Data Analysis</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Python/Pandas</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• SQL & Database</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Statistical Analysis</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with skill_col2:
        st.markdown("<div style='background-color: #252A37; padding: 15px; border-radius: 8px; border-left: 3px solid #FF9F1C;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FF9F1C; font-weight: bold; margin-bottom: 8px;'>Visualization</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Streamlit</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Plotly/Matplotlib</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Tableau</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with skill_col3:
        st.markdown("<div style='background-color: #252A37; padding: 15px; border-radius: 8px; border-left: 3px solid #43AA8B;'>", unsafe_allow_html=True)
        st.markdown("<p style='color: #43AA8B; font-weight: bold; margin-bottom: 8px;'>Machine Learning</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Predictive Models</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Natural Language Processing</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #FFFFFF; margin: 5px 0;'>• Data Mining</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Contact & Links with improved visibility
    st.markdown("<hr style='margin: 15px 0; border-color: #3A86FF;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #FFFFFF; margin: 10px 0;'>Connect With Me</h3>", unsafe_allow_html=True)
    
    link_col1, link_col2, link_col3 = st.columns(3)
    
    with link_col1:
        st.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/manish.paneru1)")
    
    with link_col2:
        st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/manishpaneru)")
    
    with link_col3:
        st.markdown("[![Portfolio](https://img.shields.io/badge/Portfolio-1DA1F2?style=for-the-badge&logo=website&logoColor=white)](https://www.analystpaneru.xyz)")

# Global filters in an expander
with st.expander("📊 **Dashboard Filters**", expanded=False):
    st.markdown("<div style='background-color: #252A37; padding: 10px; border-radius: 8px; margin-bottom: 15px;'>", unsafe_allow_html=True)
    st.markdown("<p style='color: #FFFFFF; margin: 0;'>Select filters below to customize the dashboard view. Changes will apply to all visualization sections.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        selected_seasons = st.multiselect(
            "Select Seasons", 
            options=seasons,
            default=seasons[-5:],  # Default to last 5 seasons
            key="global_seasons"
        )
    
    with filter_col2:
        selected_teams = st.multiselect(
            "Select Teams",
            options=teams,
            default=["Barcelona", "Real Madrid"] if "Barcelona" in teams and "Real Madrid" in teams else teams[:2],
            key="global_teams"
        )
    
    with filter_col3:
        date_range = st.slider(
            "Year Range",
            min_value=int(data['Year'].min()),
            max_value=int(data['Year'].max()),
            value=(int(data['Year'].max()-10), int(data['Year'].max())),
            key="global_years"
        )
    
    # Apply filters button
    st.button("Apply Filters", type="primary", key="apply_filters")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Season Overview", 
    "🔍 Team Analysis", 
    "⚔️ Match Statistics", 
    "🔮 Predictions"
])

# Filter data based on global filters
if selected_seasons:
    filtered_data = data[data['Season'].isin(selected_seasons)]
else:
    filtered_data = data

if selected_teams:
    team_filter = (filtered_data['HomeTeam'].isin(selected_teams)) | (filtered_data['AwayTeam'].isin(selected_teams))
    filtered_data = filtered_data[team_filter]

filtered_data = filtered_data[(filtered_data['Year'] >= date_range[0]) & (filtered_data['Year'] <= date_range[1])]

#--------------------------
# TAB 1: SEASON OVERVIEW
#--------------------------
with tab1:
    st.markdown("""
    <h2 style="text-align: center; margin-bottom: 20px;">Season Overview</h2>
    """, unsafe_allow_html=True)
    
    # Create helper functions for this tab
    def calculate_season_standings(df, season):
        """Calculate standings for a specific season"""
        season_data = df[df['Season'] == season]
        teams = set(season_data['HomeTeam'].unique()) | set(season_data['AwayTeam'].unique())
        
        standings = []
        for team in teams:
            # Home games
            home_games = season_data[season_data['HomeTeam'] == team]
            home_points = sum(home_games['FTR'].map({'H': 3, 'D': 1, 'A': 0}))
            home_wins = sum(home_games['FTR'] == 'H')
            home_draws = sum(home_games['FTR'] == 'D')
            home_losses = sum(home_games['FTR'] == 'A')
            home_goals_for = home_games['FTHG'].sum()
            home_goals_against = home_games['FTAG'].sum()
            
            # Away games
            away_games = season_data[season_data['AwayTeam'] == team]
            away_points = sum(away_games['FTR'].map({'A': 3, 'D': 1, 'H': 0}))
            away_wins = sum(away_games['FTR'] == 'A')
            away_draws = sum(away_games['FTR'] == 'D')
            away_losses = sum(away_games['FTR'] == 'H')
            away_goals_for = away_games['FTAG'].sum()
            away_goals_against = away_games['HTHG'].sum()
            
            # Totals
            total_points = home_points + away_points
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            total_losses = home_losses + away_losses
            total_goals_for = home_goals_for + away_goals_for
            total_goals_against = home_goals_against + away_goals_against
            goal_difference = total_goals_for - total_goals_against
            
            standings.append({
                'Team': team,
                'Points': total_points,
                'Wins': total_wins,
                'Draws': total_draws,
                'Losses': total_losses,
                'GF': total_goals_for,
                'GA': total_goals_against,
                'GD': goal_difference,
                'Home_Points': home_points,
                'Away_Points': away_points
            })
        
        standings_df = pd.DataFrame(standings)
        standings_df = standings_df.sort_values(['Points', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        standings_df['Position'] = standings_df.index + 1
        
        return standings_df
    
    # Season selection for this tab
    latest_seasons = seasons[-5:] if len(seasons) >= 5 else seasons
    selected_season = st.selectbox("Select Season", options=latest_seasons, index=len(latest_seasons)-1, key="tab1_season")
    
    # Calculate standings for selected season
    standings_df = calculate_season_standings(data, selected_season)
    
    # Top metrics row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    # Season data for metrics
    season_data = data[data['Season'] == selected_season]
    total_matches = len(season_data)
    home_wins = sum(season_data['FTR'] == 'H')
    away_wins = sum(season_data['FTR'] == 'A')
    draws = sum(season_data['FTR'] == 'D')
    total_goals = season_data['FTHG'].sum() + season_data['FTAG'].sum()
    avg_goals_per_match = total_goals / total_matches if total_matches > 0 else 0
    
    # Championship team
    champion_team = standings_df.iloc[0]['Team'] if not standings_df.empty else "Unknown"
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Champion</div>
            <div class="metric-value">{champion_team}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Matches</div>
            <div class="metric-value">{total_matches}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Goals</div>
            <div class="metric-value">{total_goals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Goals per Match</div>
            <div class="metric-value">{avg_goals_per_match:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Season standings table and visualization
    st.markdown("""
    <h3 style="margin-top: 30px; margin-bottom: 15px;">Season Standings</h3>
    """, unsafe_allow_html=True)
    
    table_col, viz_col = st.columns([1, 2])
    
    with table_col:
        # Show the standings table
        display_columns = ['Position', 'Team', 'Points', 'Wins', 'Draws', 'Losses', 'GF', 'GA', 'GD']
        st.dataframe(
            standings_df[display_columns],
            column_config={
                "Position": st.column_config.NumberColumn(format="%d"),
                "Team": "Team",
                "Points": st.column_config.NumberColumn(format="%d"),
                "GD": "Goal Diff"
            },
            use_container_width=True,
            hide_index=True
        )
    
    with viz_col:
        # Create points visualization (top 10 teams)
        top_teams = standings_df.head(10)
        
        # Color mapping based on position
        def get_position_color(pos):
            if pos <= 4:  # Champions League
                return '#3A86FF'
            elif pos <= 6:  # Europa League
                return '#5CC9F5'
            elif pos >= len(standings_df) - 3:  # Relegation
                return '#FF5A5F'
            else:  # Mid-table
                return '#43AA8B'
        
        top_teams['Color'] = top_teams['Position'].apply(get_position_color)
        
        # Points breakdown chart (Home vs Away)
        fig = go.Figure()
        
        # Home points
        fig.add_trace(go.Bar(
            y=top_teams['Team'],
            x=top_teams['Home_Points'],
            name='Home Points',
            orientation='h',
            marker=dict(color='rgba(58, 134, 255, 0.8)'),
            hovertemplate='Home Points: %{x}<extra></extra>'
        ))
        
        # Away points
        fig.add_trace(go.Bar(
            y=top_teams['Team'],
            x=top_teams['Away_Points'],
            name='Away Points',
            orientation='h',
            marker=dict(color='rgba(66, 190, 165, 0.8)'),
            hovertemplate='Away Points: %{x}<extra></extra>'
        ))
        
        # Total points labels
        annotations = []
        for i, row in top_teams.iterrows():
            annotations.append(dict(
                x=row['Home_Points'] + row['Away_Points'] + 1,
                y=row['Team'],
                text=str(row['Points']),
                font=dict(family='Arial', size=12, color='white'),
                showarrow=False
            ))
        
        fig.update_layout(
            title=f"Points Breakdown - {selected_season}",
            barmode='stack',
            yaxis=dict(
                title='',
                categoryorder='total ascending',
                showgrid=False
            ),
            xaxis=dict(
                title='Points',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            annotations=annotations,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(l=10, r=10, t=50, b=10),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Goal trends throughout the season
    st.markdown("""
    <h3 style="margin-top: 30px; margin-bottom: 15px;">Goal Trends</h3>
    """, unsafe_allow_html=True)
    
    # Prepare data for goal trends
    season_data = season_data.sort_values('Date')
    season_data['Match_Number'] = range(1, len(season_data) + 1)
    season_data['Total_Goals'] = season_data['FTHG'] + season_data['FTAG']
    season_data['Cumulative_Goals'] = season_data['Total_Goals'].cumsum()
    season_data['Rolling_Avg_Goals'] = season_data['Total_Goals'].rolling(window=5, min_periods=1).mean()
    
    # Goal trend visualization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add goals per match
    fig.add_trace(
        go.Bar(
            x=season_data['Match_Number'],
            y=season_data['Total_Goals'],
            name='Goals per Match',
            marker_color='rgba(58, 134, 255, 0.6)',
            hovertemplate='Match #%{x}<br>Goals: %{y}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add rolling average
    fig.add_trace(
        go.Scatter(
            x=season_data['Match_Number'],
            y=season_data['Rolling_Avg_Goals'],
            name='5-Match Rolling Average',
            line=dict(color='#FF9F1C', width=3),
            hovertemplate='Match #%{x}<br>5-Match Avg: %{y:.2f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add cumulative goals
    fig.add_trace(
        go.Scatter(
            x=season_data['Match_Number'],
            y=season_data['Cumulative_Goals'],
            name='Cumulative Goals',
            line=dict(color='#43AA8B', width=2, dash='dot'),
            hovertemplate='Match #%{x}<br>Cumulative Goals: %{y}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f'Goal Trends Throughout {selected_season} Season',
        xaxis=dict(
            title='Match Number',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis=dict(
            title='Goals per Match',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis2=dict(
            title='Cumulative Goals',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Home vs Away goal production
    st.markdown("""
    <h3 style="margin-top: 30px; margin-bottom: 15px;">Home vs Away Goal Production</h3>
    """, unsafe_allow_html=True)
    
    # Prepare data for home vs away comparison
    home_goals_by_team = season_data.groupby('HomeTeam')['FTHG'].sum().reset_index()
    home_goals_by_team.columns = ['Team', 'Home_Goals']
    
    away_goals_by_team = season_data.groupby('AwayTeam')['FTAG'].sum().reset_index()
    away_goals_by_team.columns = ['Team', 'Away_Goals']
    
    goals_by_team = pd.merge(home_goals_by_team, away_goals_by_team, on='Team', how='outer').fillna(0)
    goals_by_team['Total_Goals'] = goals_by_team['Home_Goals'] + goals_by_team['Away_Goals']
    goals_by_team = goals_by_team.sort_values('Total_Goals', ascending=False)
    
    # Only show top 10 teams by total goals
    top_goal_teams = goals_by_team.head(10)
    
    # Calculate home goal percentage
    top_goal_teams['Home_Goal_Pct'] = top_goal_teams['Home_Goals'] / top_goal_teams['Total_Goals'] * 100
    
    # Create the visualization
    fig = go.Figure()
    
    # Home goals
    fig.add_trace(go.Bar(
        y=top_goal_teams['Team'],
        x=top_goal_teams['Home_Goals'],
        name='Home Goals',
        orientation='h',
        marker=dict(color='rgba(58, 134, 255, 0.8)'),
        hovertemplate='Home Goals: %{x}<extra></extra>'
    ))
    
    # Away goals
    fig.add_trace(go.Bar(
        y=top_goal_teams['Team'],
        x=top_goal_teams['Away_Goals'],
        name='Away Goals',
        orientation='h',
        marker=dict(color='rgba(255, 90, 95, 0.8)'),
        hovertemplate='Away Goals: %{x}<extra></extra>'
    ))
    
    # Add percentage annotations
    annotations = []
    for i, row in top_goal_teams.iterrows():
        annotations.append(dict(
            x=row['Total_Goals'] + 2,
            y=row['Team'],
            text=f"{row['Home_Goal_Pct']:.1f}% home",
            font=dict(family='Arial', size=11, color='white'),
            showarrow=False
        ))
    
    fig.update_layout(
        title=f'Goal Production by Team (Home vs Away) - {selected_season}',
        barmode='stack',
        yaxis=dict(
            title='',
            categoryorder='total ascending',
            showgrid=False
        ),
        xaxis=dict(
            title='Goals Scored',
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=annotations,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

#--------------------------
# TAB 2: TEAM ANALYSIS
#--------------------------
with tab2:
    st.markdown("""
    <h2 style="text-align: center; margin-bottom: 20px;">Team Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Team selection
    team_analysis_col1, team_analysis_col2 = st.columns([1,1])
    
    with team_analysis_col1:
        selected_team = st.selectbox("Select Team to Analyze", options=teams, key="team_analysis_team")
    
    with team_analysis_col2:
        # Season range for performance over time
        team_seasons = data[
            (data['HomeTeam'] == selected_team) | 
            (data['AwayTeam'] == selected_team)
        ]['Season'].unique()
        
        selected_team_seasons = st.multiselect(
            "Select Seasons to Compare", 
            options=team_seasons,
            default=team_seasons[-5:] if len(team_seasons) >= 5 else team_seasons,
            key="team_analysis_seasons"
        )
    
    # Calculate team metrics across seasons
    def calculate_team_season_metrics(df, team, seasons):
        """Calculate metrics for a team across multiple seasons"""
        results = []
        
        for season in seasons:
            season_data = df[df['Season'] == season]
            
            # Home games
            home_games = season_data[season_data['HomeTeam'] == team]
            home_matches = len(home_games)
            home_wins = sum(home_games['FTR'] == 'H')
            home_draws = sum(home_games['FTR'] == 'D')
            home_losses = sum(home_games['FTR'] == 'A')
            home_goals_for = home_games['FTHG'].sum()
            home_goals_against = home_games['FTAG'].sum()
            
            # Away games
            away_games = season_data[season_data['AwayTeam'] == team]
            away_matches = len(away_games)
            away_wins = sum(away_games['FTR'] == 'A')
            away_draws = sum(away_games['FTR'] == 'D')
            away_losses = sum(away_games['FTR'] == 'H')
            away_goals_for = away_games['FTAG'].sum()
            away_goals_against = away_games['FTHG'].sum()
            
            # Total
            total_matches = home_matches + away_matches
            total_wins = home_wins + away_wins
            total_draws = home_draws + away_draws
            total_losses = home_losses + away_losses
            total_goals_for = home_goals_for + away_goals_for
            total_goals_against = home_goals_against + away_goals_against
            
            # Calculate points and win rate
            total_points = (total_wins * 3) + total_draws
            win_rate = (total_wins / total_matches) * 100 if total_matches > 0 else 0
            
            # Calculate average goals for and against
            avg_goals_for = total_goals_for / total_matches if total_matches > 0 else 0
            avg_goals_against = total_goals_against / total_matches if total_matches > 0 else 0
            
            # Calculate final position (requires standings calculation)
            standings = calculate_season_standings(df, season)
            try:
                team_row = standings[standings['Team'] == team]
                final_position = team_row['Position'].values[0] if not team_row.empty else None
            except:
                final_position = None
            
            results.append({
                'Season': season,
                'Final_Position': final_position,
                'Matches': total_matches,
                'Wins': total_wins,
                'Draws': total_draws,
                'Losses': total_losses,
                'Goals_For': total_goals_for,
                'Goals_Against': total_goals_against,
                'Points': total_points,
                'Win_Rate': win_rate,
                'Avg_Goals_For': avg_goals_for,
                'Avg_Goals_Against': avg_goals_against,
                'Home_Wins': home_wins,
                'Away_Wins': away_wins,
                'Home_Points': (home_wins * 3) + home_draws,
                'Away_Points': (away_wins * 3) + away_draws
            })
        
        return pd.DataFrame(results)
    
    # Get team metrics
    if selected_team_seasons:
        team_metrics = calculate_team_season_metrics(data, selected_team, selected_team_seasons)
        
        # Performance timeline
        st.markdown("""
        <h3 style="margin-top: 30px; margin-bottom: 15px;">Performance Timeline</h3>
        """, unsafe_allow_html=True)
        
        # Team performance over seasons
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add points line
        fig.add_trace(
            go.Scatter(
                x=team_metrics['Season'],
                y=team_metrics['Points'],
                name='Points',
                line=dict(color='#3A86FF', width=3),
                hovertemplate='%{x}<br>Points: %{y}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add position line (inverted so higher is better)
        if not team_metrics['Final_Position'].isnull().all():
            max_position = team_metrics['Final_Position'].max()
            fig.add_trace(
                go.Scatter(
                    x=team_metrics['Season'],
                    y=[(max_position - pos + 1) if not pd.isna(pos) else None for pos in team_metrics['Final_Position']],
                    name='Position',
                    line=dict(color='#FF9F1C', width=3, dash='dash'),
                    hovertemplate='%{x}<br>Position: %{text}<extra></extra>',
                    text=[str(int(pos)) if not pd.isna(pos) else 'N/A' for pos in team_metrics['Final_Position']]
                ),
                secondary_y=True
            )
            
            # Add position labels
            fig.update_yaxes(
                title_text="Position",
                tickvals=list(range(1, max_position + 1)),
                ticktext=[str(max_position - i + 1) for i in range(1, max_position + 1)],
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title=f'{selected_team} Performance Timeline',
            xaxis=dict(
                title='Season',
                showgrid=False
            ),
            yaxis=dict(
                title='Points',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Win-Draw-Loss distribution across seasons
        st.markdown("""
        <h3 style="margin-top: 30px; margin-bottom: 15px;">Win-Draw-Loss Distribution</h3>
        """, unsafe_allow_html=True)
        
        # Prepare data for stacked bar chart
        fig = go.Figure()
        
        # Add win bars
        fig.add_trace(go.Bar(
            x=team_metrics['Season'],
            y=team_metrics['Wins'],
            name='Wins',
            marker_color='rgba(67, 170, 139, 0.8)',
            hovertemplate='%{x}<br>Wins: %{y}<extra></extra>'
        ))
        
        # Add draw bars
        fig.add_trace(go.Bar(
            x=team_metrics['Season'],
            y=team_metrics['Draws'],
            name='Draws',
            marker_color='rgba(92, 201, 245, 0.8)',
            hovertemplate='%{x}<br>Draws: %{y}<extra></extra>'
        ))
        
        # Add loss bars
        fig.add_trace(go.Bar(
            x=team_metrics['Season'],
            y=team_metrics['Losses'],
            name='Losses',
            marker_color='rgba(255, 90, 95, 0.8)',
            hovertemplate='%{x}<br>Losses: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{selected_team} Match Results by Season',
            barmode='stack',
            xaxis=dict(
                title='Season',
                showgrid=False
            ),
            yaxis=dict(
                title='Number of Matches',
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart comparison across seasons
        st.markdown("""
        <h3 style="margin-top: 30px; margin-bottom: 15px;">Performance Radar Chart</h3>
        """, unsafe_allow_html=True)
        
        # Prepare data for radar chart
        radar_metrics = ['Win_Rate', 'Avg_Goals_For', 'Avg_Goals_Against', 'Home_Wins', 'Away_Wins']
        radar_display_names = ['Win Rate (%)', 'Avg Goals Scored', 'Avg Goals Conceded', 'Home Wins', 'Away Wins']
        
        # Normalize metrics for radar chart (0-1 scale)
        radar_df = team_metrics[['Season'] + radar_metrics].copy()
        
        for metric in radar_metrics:
            max_val = radar_df[metric].max()
            min_val = radar_df[metric].min()
            if max_val > min_val:
                radar_df[f'{metric}_Normalized'] = (radar_df[metric] - min_val) / (max_val - min_val)
            else:
                radar_df[f'{metric}_Normalized'] = 0.5  # Default if all values are the same
        
        # Create radar chart
        fig = go.Figure()
        
        # Add a trace for each season
        colors = ['#3A86FF', '#FF9F1C', '#43AA8B', '#F94144', '#9D4EDD', '#FFD166', '#06D6A0']
        
        for i, season in enumerate(radar_df['Season']):
            season_row = radar_df[radar_df['Season'] == season]
            
            # Get normalized values
            values = [season_row[f'{metric}_Normalized'].values[0] for metric in radar_metrics]
            # Add the first value again to close the loop
            values = values + [values[0]]
            
            # Add original values for display in hover text
            display_values = [season_row[metric].values[0] for metric in radar_metrics]
            hover_text = [f"{name}: {value:.2f}" for name, value in zip(radar_display_names, display_values)]
            hover_text = hover_text + [hover_text[0]]  # Close the loop
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_display_names + [radar_display_names[0]],  # Close the loop
                name=season,
                fill='toself',
                line=dict(color=colors[i % len(colors)]),
                hovertemplate='%{text}<extra>' + season + '</extra>',
                text=hover_text
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            title=f"{selected_team} Performance Comparison Across Seasons",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Home vs Away performance
        st.markdown("""
        <h3 style="margin-top: 30px; margin-bottom: 15px;">Home vs Away Performance</h3>
        """, unsafe_allow_html=True)
        
        # Prepare data for home vs away comparison
        home_away_col1, home_away_col2 = st.columns(2)
        
        with home_away_col1:
            # Home vs Away Points
            fig = go.Figure()
            
            # Home points
            fig.add_trace(go.Bar(
                x=team_metrics['Season'],
                y=team_metrics['Home_Points'],
                name='Home Points',
                marker_color='rgba(58, 134, 255, 0.8)',
                hovertemplate='%{x}<br>Home Points: %{y}<extra></extra>'
            ))
            
            # Away points
            fig.add_trace(go.Bar(
                x=team_metrics['Season'],
                y=team_metrics['Away_Points'],
                name='Away Points',
                marker_color='rgba(255, 90, 95, 0.8)',
                hovertemplate='%{x}<br>Away Points: %{y}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{selected_team} Home vs Away Points',
                barmode='group',
                xaxis=dict(
                    title='Season',
                    showgrid=False
                ),
                yaxis=dict(
                    title='Points',
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with home_away_col2:
            # Home vs Away Wins
            fig = go.Figure()
            
            # Home wins
            fig.add_trace(go.Bar(
                x=team_metrics['Season'],
                y=team_metrics['Home_Wins'],
                name='Home Wins',
                marker_color='rgba(58, 134, 255, 0.8)',
                hovertemplate='%{x}<br>Home Wins: %{y}<extra></extra>'
            ))
            
            # Away wins
            fig.add_trace(go.Bar(
                x=team_metrics['Season'],
                y=team_metrics['Away_Wins'],
                name='Away Wins',
                marker_color='rgba(255, 90, 95, 0.8)',
                hovertemplate='%{x}<br>Away Wins: %{y}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title=f'{selected_team} Home vs Away Wins',
                barmode='group',
                xaxis=dict(
                    title='Season',
                    showgrid=False
                ),
                yaxis=dict(
                    title='Wins',
                    showgrid=True,
                    gridcolor='rgba(255, 255, 255, 0.1)'
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Head-to-head analysis
        st.markdown("""
        <h3 style="margin-top: 30px; margin-bottom: 15px;">Head-to-Head Analysis</h3>
        """, unsafe_allow_html=True)
        
        opponent_team = st.selectbox("Select Opponent Team", 
                                    options=[t for t in teams if t != selected_team], 
                                    key="head_to_head_opponent")
        
        # Get head-to-head matches
        h2h_matches = data[
            ((data['HomeTeam'] == selected_team) & (data['AwayTeam'] == opponent_team)) |
            ((data['HomeTeam'] == opponent_team) & (data['AwayTeam'] == selected_team))
        ].sort_values('Date')
        
        if not h2h_matches.empty:
            # Calculate head-to-head stats
            total_matches = len(h2h_matches)
            selected_team_wins = 0
            opponent_team_wins = 0
            draws = 0
            
            for _, match in h2h_matches.iterrows():
                if match['HomeTeam'] == selected_team and match['FTR'] == 'H':
                    selected_team_wins += 1
                elif match['AwayTeam'] == selected_team and match['FTR'] == 'A':
                    selected_team_wins += 1
                elif match['HomeTeam'] == opponent_team and match['FTR'] == 'H':
                    opponent_team_wins += 1
                elif match['AwayTeam'] == opponent_team and match['FTR'] == 'A':
                    opponent_team_wins += 1
                else:
                    draws += 1
            
            # Display head-to-head stats
            h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
            
            with h2h_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{selected_team} Wins</div>
                    <div class="metric-value">{selected_team_wins}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with h2h_col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Draws</div>
                    <div class="metric-value">{draws}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with h2h_col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{opponent_team} Wins</div>
                    <div class="metric-value">{opponent_team_wins}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display head-to-head matches
            st.markdown("""
            <h4 style="margin-top: 20px; margin-bottom: 10px;">Recent Head-to-Head Matches</h4>
            """, unsafe_allow_html=True)
            
            # Display the most recent 10 matches
            recent_h2h = h2h_matches.tail(10)
            
            # Format the dataframe for display
            display_h2h = recent_h2h[['Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']].copy()
            display_h2h['Score'] = display_h2h['FTHG'].astype(str) + ' - ' + display_h2h['FTAG'].astype(str)
            display_h2h['Date'] = pd.to_datetime(display_h2h['Date']).dt.strftime('%d %b %Y')
            display_h2h['Match'] = display_h2h['HomeTeam'] + ' vs ' + display_h2h['AwayTeam']
            
            st.dataframe(
                display_h2h[['Season', 'Date', 'Match', 'Score']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"No head-to-head matches found between {selected_team} and {opponent_team}.")
    else:
        st.warning("Please select at least one season to analyze.")