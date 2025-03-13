import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import io
import base64
import webbrowser
import os

# Set style elements
plt.style.use('fivethirtyeight')
sns.set_palette("viridis")

# Load the data from Excel
def load_fantasy_data(file_path):
    """Load all sheets from the Excel file into a dictionary of dataframes."""
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
    return sheets

# 1. VISUALIZATION: Team Offensive Stats Chart
def create_offensive_stat_chart(data, stat_column):
    """Create a horizontal bar chart for any offensive stat."""
    teams_df = data['Team Stats']
    sorted_teams = teams_df.sort_values(stat_column, ascending=False)
    
    # Calculate league average
    league_avg = teams_df[stat_column].mean()
    
    # Create color mapping for teams (consistent colors)
    colors = px.colors.qualitative.Bold + px.colors.qualitative.Safe + px.colors.qualitative.Vivid
    
    # Determine title and axis labels
    if 'Yards' in stat_column:
        x_title = 'Yards'
    elif 'TDs' in stat_column:
        x_title = 'Touchdowns'
    else:
        x_title = 'Count'
    
    # Create figure
    fig = px.bar(
        sorted_teams,
        x=stat_column,
        y='Team Name',
        orientation='h',
        title=stat_column,
        labels={stat_column: x_title, 'Team Name': ''},
        text=stat_column,
        height=700,
        color='Team Name',  # Use team name for coloring
        color_discrete_sequence=colors  # Use the defined color palette
    )
    
    # Add league average line
    fig.add_vline(
        x=league_avg,
        line_dash='dash',
        line_color='grey',
        line_width=2
    )
    
    # Add league average annotation above the chart
    fig.add_annotation(
        x=league_avg,
        y=-0.5,  # Position above the chart
        yref="paper",
        text=f"League Avg: {league_avg:.0f}",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=14, color="grey"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="grey",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=100, b=20),
        bargap=0.3,
        yaxis=dict(autorange="reversed"),
        xaxis=dict(title=dict(text=x_title, font=dict(size=14))),
        plot_bgcolor='rgba(248,248,250,0.5)',
        showlegend=False  # Hide the legend since colors are intuitive
    )
    
    fig.update_traces(
        texttemplate='%{x:.0f}',
        textposition='outside'
    )
    
    return fig

# We now have a single function that can create any offensive stat chart

# 2. VISUALIZATION: Weekly Performance Heatmap
def create_weekly_performance_heatmap(data):
    """Create a heatmap of weekly team performances."""
    weekly_scores = data['Weekly Scores']
    
    # Pivot the data to create a team x week matrix of scores
    heatmap_data = weekly_scores.pivot(index='Team Name', columns='Week', values='Points')
    
    # Create a heatmap using plotly
    fig = px.imshow(
        heatmap_data,
        text_auto='.1f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        labels=dict(x="Week", y="Team", color="Fantasy Points")
    )
    
    fig.update_layout(
        title='Team Weekly Performance Heatmap',
        height=600
    )
    
    return fig

# 3. VISUALIZATION: Team Consistency Analysis
def create_consistency_analysis(data):
    """Create a scatter plot showing team consistency vs average score."""
    standings = data['Standings']
    
    # Create figure
    fig = px.scatter(
        standings,
        x='Score Std Dev',
        y='Average Score',
        color='Playoff Seed',
        size='Points For',
        hover_name='Team Name',
        text='Team Name',
        title='Team Consistency vs. Average Score',
        labels={
            'Score Std Dev': 'Standard Deviation (Lower = More Consistent)',
            'Average Score': 'Average Weekly Score',
            'Playoff Seed': 'Playoff Seed'
        },
        height=600
    )
    
    # Add quadrant lines based on median values
    median_std = standings['Score Std Dev'].median()
    median_avg = standings['Average Score'].median()
    
    fig.add_vline(x=median_std, line_dash="dash", line_color="gray")
    fig.add_hline(y=median_avg, line_dash="dash", line_color="gray")
    
    # Add quadrant labels
    fig.add_annotation(x=standings['Score Std Dev'].max() * 0.9, 
                      y=standings['Average Score'].max() * 0.9, 
                      text="High Scoring, Inconsistent", 
                      showarrow=False)
    
    fig.add_annotation(x=standings['Score Std Dev'].min() * 1.1, 
                      y=standings['Average Score'].max() * 0.9, 
                      text="High Scoring, Consistent", 
                      showarrow=False)
    
    fig.add_annotation(x=standings['Score Std Dev'].max() * 0.9, 
                      y=standings['Average Score'].min() * 1.1, 
                      text="Low Scoring, Inconsistent", 
                      showarrow=False)
    
    fig.add_annotation(x=standings['Score Std Dev'].min() * 1.1, 
                      y=standings['Average Score'].min() * 1.1, 
                      text="Low Scoring, Consistent", 
                      showarrow=False)
    
    fig.update_traces(textposition='top center')
    
    return fig

# 4. VISUALIZATION: Offensive Efficiency Analysis
# Update the create_offensive_efficiency function to fix the legend overlap
def create_offensive_efficiency(data):
    """Create visualization analyzing offensive efficiency."""
    teams = data['Teams']
    team_stats = data['Team Stats']
    
    # Merge datasets
    merged_df = pd.merge(teams, team_stats, on=['Team ID', 'Team Name'])
    
    # Calculate offensive efficiency (Points per 100 yards)
    merged_df['Total Yards'] = merged_df['Passing Yards'] + merged_df['Rushing Yards'] + merged_df['Receiving Yards']
    merged_df['Points per 100 Yards'] = (merged_df['Points For'] / merged_df['Total Yards']) * 100
    
    # Calculate TD Efficiency (TDs per 100 yards)
    merged_df['Total TDs'] = merged_df['Passing TDs'] + merged_df['Rushing TDs'] + merged_df['Receiving TDs']
    merged_df['TDs per 100 Yards'] = (merged_df['Total TDs'] / merged_df['Total Yards']) * 100
    
    # Create a scatter plot
    fig = px.scatter(
        merged_df,
        x='Total Yards',
        y='Points For',
        size='Total TDs',
        color='Points per 100 Yards',
        hover_name='Team Name',
        text='Team Name',
        title='Offensive Efficiency Analysis',
        labels={
            'Total Yards': 'Total Offensive Yards',
            'Points For': 'Fantasy Points',
            'Total TDs': 'Total Touchdowns',
            'Points per 100 Yards': 'Efficiency (Pts/100 Yds)'  # Shortened label
        },
        height=600
    )
    
    # Update color bar title formatting
    fig.update_coloraxes(
        colorbar=dict(
            title=dict(
                text="Efficiency<br>(Pts/100 Yds)",  # Line break for better formatting
                font=dict(size=12)
            ),
            len=0.8,  # Shorter color bar
            thickness=15  # Slightly thicker
        )
    )
    
    # Add trendline and improve layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(r=80)  # Add more right margin for the colorbar
    )
    
    # Add efficiency reference line
    x_range = np.linspace(merged_df['Total Yards'].min(), merged_df['Total Yards'].max(), 100)
    avg_efficiency = merged_df['Points per 100 Yards'].mean()
    y_range = avg_efficiency * x_range / 100
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=f'Avg Efficiency ({avg_efficiency:.2f})',  # Shortened name
            line=dict(dash='dash', color='gray')
        )
    )
    
    # Improve text positioning to avoid overlap
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=10)  # Smaller text size
    )
    
    return fig

def create_points_analysis(data):
    """Create improved visualization showing points for/against with better separation."""
    import plotly.graph_objects as go
    
    teams = data['Teams']
    
    # Calculate point differential and win percentage for sorting
    teams['Point Differential'] = teams['Points For'] - teams['Points Against']
    
    # Sort by point differential
    teams_sorted = teams.sort_values('Point Differential', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Calculate minimum value to start y-axis (75% of minimum value)
    y_min = min(teams['Points For'].min(), teams['Points Against'].min()) * 0.75
    
    # Add Points For vs Points Against
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Points For'],
            name='Points For',
            marker_color='#3498db',
            text=teams_sorted['Points For'].round(0).astype(int),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Points For: %{y}<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Points Against'],
            name='Points Against',
            marker_color='#f39c12',
            text=teams_sorted['Points Against'].round(0).astype(int),
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Points Against: %{y}<extra></extra>'
        )
    )
    
    # Add point differential as a third element
    fig.add_trace(
        go.Scatter(
            x=teams_sorted['Team Name'],
            y=[(teams_sorted['Points For'][i] + teams_sorted['Points Against'][i])/2 for i in teams_sorted.index],
            mode='markers+text',
            marker=dict(
                symbol='diamond',
                size=12,
                color=['green' if diff > 0 else 'red' for diff in teams_sorted['Point Differential']],
                line=dict(color='black', width=1)
            ),
            text=teams_sorted['Point Differential'].round(0).astype(int).apply(lambda x: f"+{x}" if x > 0 else str(x)),
            textposition='middle right',
            name='Point Diff',
            hovertemplate='<b>%{x}</b><br>Point Differential: %{text}<extra></extra>'
        )
    )
    
    # Update layout with zoomed y-axis
    fig.update_layout(
        title='Points For vs Points Against Analysis',
        barmode='group',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        yaxis=dict(
            range=[y_min, max(teams['Points For'].max(), teams['Points Against'].max()) * 1.1],
            title='Fantasy Points'
        ),
        annotations=[
            dict(
                x=0.5,
                y=-0.15,
                xref='paper',
                yref='paper',
                text='Note: Y-axis does not start at zero to highlight differences between teams',
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )
    
    # Add win-loss record on x-axis
    team_labels = []
    for idx, row in teams_sorted.iterrows():
        team_labels.append(f"{row['Team Name']}<br>({row['Wins']}-{row['Losses']})")
    
    fig.update_xaxes(
        tickangle=45,
        tickmode='array',
        tickvals=teams_sorted['Team Name'],
        ticktext=team_labels
    )
    
    return fig

# 6. VISUALIZATION: Weekly Scoring Trends
def create_weekly_scoring_trends(data):
    """Create a line chart showing weekly scoring trends."""
    weekly_scores = data['Weekly Scores']
    
    # Calculate weekly averages, max, min
    weekly_stats = weekly_scores.groupby('Week').agg(
        avg_score=('Points', 'mean'),
        max_score=('Points', 'max'),
        min_score=('Points', 'min')
    ).reset_index()
    
    # Create figure
    fig = go.Figure()
    
    # Add average line
    fig.add_trace(
        go.Scatter(
            x=weekly_stats['Week'],
            y=weekly_stats['avg_score'],
            mode='lines+markers',
            name='Average Score',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        )
    )
    
    # Add range area
    fig.add_trace(
        go.Scatter(
            x=weekly_stats['Week'],
            y=weekly_stats['max_score'],
            mode='lines',
            name='Max Score',
            line=dict(width=0),
            marker=dict(color="#444"),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=weekly_stats['Week'],
            y=weekly_stats['min_score'],
            mode='lines',
            name='Min Score',
            line=dict(width=0),
            marker=dict(color="#444"),
            fillcolor='rgba(52, 152, 219, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    )
    
    # Add highest and lowest scores
    for i, row in weekly_stats.iterrows():
        # Get team with highest score for the week
        max_team = weekly_scores[(weekly_scores['Week'] == row['Week']) & 
                                (weekly_scores['Points'] == row['max_score'])]['Team Name'].values[0]
        
        min_team = weekly_scores[(weekly_scores['Week'] == row['Week']) & 
                                (weekly_scores['Points'] == row['min_score'])]['Team Name'].values[0]
        
        # Add annotations for highest score
        if i % 2 == 0:  # Only annotate every other week to reduce clutter
            fig.add_annotation(
                x=row['Week'],
                y=row['max_score'],
                text=f"{max_team}: {row['max_score']:.1f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-30
            )
    
    # Update layout
    fig.update_layout(
        title='Weekly Scoring Trends',
        xaxis=dict(title='Week', tickmode='linear'),
        yaxis=dict(title='Fantasy Points'),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    
    return fig

# 7. VISUALIZATION: Home vs Away Performance
def create_home_away_analysis(data):
    """Create analysis of home vs away performance."""
    close_games = data['Close Game Analysis']
    
    # Calculate home and away win percentages
    close_games['Home Win %'] = close_games['Home Wins'] / (close_games['Home Wins'] + close_games['Home Losses'])
    close_games['Away Win %'] = close_games['Away Wins'] / (close_games['Away Wins'] + close_games['Away Losses'])
    close_games['Home Advantage'] = close_games['Home Win %'] - close_games['Away Win %']
    
    # Sort by home advantage
    sorted_teams = close_games.sort_values('Home Advantage', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add home win % bars
    fig.add_trace(
        go.Bar(
            y=sorted_teams['Team Name'],
            x=sorted_teams['Home Win %'],
            name='Home Win %',
            orientation='h',
            marker_color='#2ecc71'
        )
    )
    
    # Add away win % bars
    fig.add_trace(
        go.Bar(
            y=sorted_teams['Team Name'],
            x=sorted_teams['Away Win %'],
            name='Away Win %',
            orientation='h',
            marker_color='#e74c3c'
        )
    )
    
    # Add home advantage annotations
    for i, team in enumerate(sorted_teams['Team Name']):
        home_adv = sorted_teams.loc[sorted_teams['Team Name'] == team, 'Home Advantage'].values[0]
        
        # Only add text if there's a significant advantage
        if abs(home_adv) > 0.05:
            fig.add_annotation(
                y=team,
                x=1.0,  # Position at the end
                text=f"Home Adv: {home_adv:.2f}",
                showarrow=False,
                xanchor='left',
                xshift=10
            )
    
    # Update layout
    fig.update_layout(
        title='Home vs Away Performance Analysis',
        xaxis=dict(title='Win Percentage', tickformat='.0%'),
        barmode='group',
        height=600
    )
    
    return fig

# 8. VISUALIZATION: Points Distribution Analysis
def create_points_overall_distribution(data):
    """Create visualization showing overall distribution of points."""
    weekly_scores = data['Weekly Scores']
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=weekly_scores['Points'],
            nbinsx=20,
            marker_color='#3498db',
            name='All Scores'
        )
    )
    
    # Add mean line
    mean_score = weekly_scores['Points'].mean()
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.1f}",
        annotation_position="top right"
    )
    
    # Add median line
    median_score = weekly_scores['Points'].median()
    fig.add_vline(
        x=median_score,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Median: {median_score:.1f}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title='Overall Fantasy Points Distribution',
        xaxis_title='Fantasy Points',
        yaxis_title='Frequency',
        height=500,
        margin=dict(t=50, b=50),
        plot_bgcolor='rgba(240, 240, 245, 0.8)',
        bargap=0.1
    )
    
    return fig

def create_team_points_distribution(data):
    """Create visualization showing team-specific point distributions."""
    weekly_scores = data['Weekly Scores']
    teams_df = data['Teams']
    
    # Get all team names
    all_teams = teams_df['Team Name'].unique().tolist()
    
    # Calculate team stats for sorting
    team_stats = []
    for team in all_teams:
        team_data = weekly_scores[weekly_scores['Team Name'] == team]['Points']
        if len(team_data) > 0:  # Only add if there are scores
            team_stats.append({
                'team': team,
                'mean': team_data.mean(),
                'max': team_data.max(),
                'min': team_data.min(),
                'data': team_data
            })
    
    # Sort teams by mean score (highest to lowest)
    sorted_teams = sorted(team_stats, key=lambda x: x['mean'], reverse=True)
    sorted_team_names = [team['team'] for team in sorted_teams]
    
    # Create the figure for team distributions
    fig = go.Figure()
    
    # Add box plots for all teams
    for team_stat in sorted_teams:
        team = team_stat['team']
        team_data = team_stat['data']
        
        fig.add_trace(
            go.Box(
                y=[team] * len(team_data),
                x=team_data,
                name=team,
                orientation='h',
                boxpoints='all',  # Show all points
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(
                    size=4,
                    opacity=0.7
                ),
                line=dict(width=1.5),
                showlegend=False
            )
        )
    
    # Add league average line
    mean_score = weekly_scores['Points'].mean()
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"League Average: {mean_score:.1f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title='Team-specific Points Distributions (Sorted by Average Score)',
        xaxis_title='Fantasy Points',
        yaxis_title='Team',
        height=700,  # Taller to accommodate all teams
        margin=dict(l=150, r=50, t=50, b=50),  # More left margin for team names
        plot_bgcolor='rgba(240, 240, 245, 0.8)',
        yaxis=dict(
            categoryorder='array',
            categoryarray=sorted_team_names
        )
    )
    
    # Add annotations for key metrics
    for i, team_stat in enumerate(sorted_teams):
        # Add max annotation for every other team to reduce clutter
        if i % 2 == 0:
            fig.add_annotation(
                x=team_stat['max'],
                y=team_stat['team'],
                text=f"Max: {team_stat['max']:.1f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="#636363",
                ax=30,
                ay=0,
                font=dict(size=9, color='darkblue')
            )
    
    return fig

# 9. VISUALIZATION: Transaction Analysis
def create_transaction_scatter(data):
    """Create scatter plot for transaction activity vs team success."""
    transactions = data['Transactions']
    standings = data['Standings']
    
    # Merge transaction data with standings
    merged_df = pd.merge(transactions, standings[['Team ID', 'Team Name', 'Wins', 'Losses', 'Win %']], 
                         on=['Team ID', 'Team Name'])
    
    # Create the figure
    fig = go.Figure()
    
    # Add scatter plot for transactions vs win %
    fig.add_trace(
        go.Scatter(
            x=merged_df['Total Acquisitions'],
            y=merged_df['Win %'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=merged_df['Total Trades'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title='Trades Made',
                    thickness=15,
                    len=0.8
                )
            ),
            text=merged_df['Team Name'],
            textposition='top center',
            name='Teams'
        )
    )
    
    # Calculate correlation
    correlation = merged_df['Total Acquisitions'].corr(merged_df['Win %'])
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.95,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.2f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )
    
    # Update layout
    fig.update_layout(
        title='Transaction Activity vs. Team Success',
        xaxis=dict(title='Total Player Acquisitions'),
        yaxis=dict(title='Win Percentage', tickformat='.0%'),
        height=600,
        margin=dict(t=50, r=50, b=75, l=50)
    )
    
    return fig

def create_weekly_transactions(data):
    """Create bar chart for weekly transaction activity."""
    transactions = data['Transactions']
    
    # Sum transactions by week
    weeks = [col for col in transactions.columns if 'Week' in col and 'Adds' in col]
    weekly_adds = {}
    
    for week in weeks:
        if 'Week' in week and 'Adds' in week:
            week_num = int(''.join(filter(str.isdigit, week)))
            # Convert to numeric, treating empty strings as 0
            transactions[week] = pd.to_numeric(transactions[week], errors='coerce').fillna(0)
            weekly_adds[week_num] = transactions[week].sum()
    
    # Convert to dataframe
    weekly_adds_df = pd.DataFrame({
        'Week': list(weekly_adds.keys()),
        'Transactions': list(weekly_adds.values())
    }).sort_values('Week')
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart for weekly transactions
    fig.add_trace(
        go.Bar(
            x=weekly_adds_df['Week'],
            y=weekly_adds_df['Transactions'],
            marker_color='#3498db',
            name='Weekly Transactions'
        )
    )
    
    # Add line for average
    avg_transactions = weekly_adds_df['Transactions'].mean()
    fig.add_hline(
        y=avg_transactions,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_transactions:.1f}",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        title='Weekly Transaction Activity',
        xaxis=dict(title='Week', tickmode='linear'),
        yaxis=dict(title='Number of Transactions'),
        height=500,
        margin=dict(t=50, r=50, b=50, l=50)
    )
    
    return fig

# 10. VISUALIZATION: Close Game Analysis
def create_close_game_analysis(data):
    """Create a simplified, clear visualization for close games analysis."""
    # Extract the data we need
    close_games = data['Close Game Analysis'] 
    
    # Check data structure 
    print("Original columns:", close_games.columns.tolist())
    
    # Create basic derived columns if needed
    if 'Close Wins' in close_games.columns and 'Blowout Wins' in close_games.columns:
        # Calculate total wins
        if 'Total Wins' not in close_games.columns:
            close_games['Total Wins'] = close_games['Close Wins'] + close_games['Blowout Wins']
        
        # Make sure we have the right columns for our analysis
        needed_columns = ['Team Name', 'Close Wins', 'Blowout Wins', 'Total Wins']
        if all(col in close_games.columns for col in needed_columns):
            # Calculate percentages of each win type
            close_games['Close Win %'] = close_games['Close Wins'] / close_games['Total Wins']
            close_games['Blowout Win %'] = close_games['Blowout Wins'] / close_games['Total Wins']
            
            # Sort by total wins for the visualization
            sorted_teams = close_games.sort_values('Total Wins', ascending=False)
            
            # Create the visualization
            fig = go.Figure()
            
            # Add stacked bars for win types
            fig.add_trace(go.Bar(
                y=sorted_teams['Team Name'],
                x=sorted_teams['Close Wins'],
                name='Close Wins',
                orientation='h',
                marker_color='#2ecc71',
                hovertemplate='<b>%{y}</b><br>Close Wins: %{x}<extra></extra>'
            ))
            
            fig.add_trace(go.Bar(
                y=sorted_teams['Team Name'],
                x=sorted_teams['Blowout Wins'],
                name='Blowout Wins',
                orientation='h',
                marker_color='#3498db',
                hovertemplate='<b>%{y}</b><br>Blowout Wins: %{x}<extra></extra>'
            ))
            
            # Add a line for close game percentage
            if 'Close Wins' in close_games.columns and 'Close Losses' in close_games.columns:
                # Calculate close game percentage
                close_games['Total Close Games'] = close_games['Close Wins'] + close_games['Close Losses']
                
                if 'Total Games' in close_games.columns:
                    close_games['Close Game %'] = close_games['Total Close Games'] / close_games['Total Games']
                    
                    # Add scatter points showing percentage of games that are close
                    fig.add_trace(go.Scatter(
                        y=sorted_teams['Team Name'],
                        x=sorted_teams['Total Wins'] + 1,  # Offset to the right of the bars
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            size=15,
                            color=sorted_teams['Close Game %'],
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(
                                title="% of Games<br>That Are Close",
                                thickness=15,
                                x=1.1  # Position colorbar further right
                            )
                        ),
                        name='% of Games That Are Close',
                        hovertemplate='<b>%{y}</b><br>Close Game %: %{marker.color:.0%}<extra></extra>'
                    ))
            
            # Update layout
            fig.update_layout(
                title="Close Game Analysis: Win Distribution",
                barmode='stack',
                height=600,
                margin=dict(t=50, r=150, b=50, l=150),  # Increased right margin for colorbar
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                yaxis=dict(
                    title="",
                    autorange="reversed",  # This puts the team with most wins at the top
                ),
                xaxis=dict(
                    title="Number of Wins",
                ),
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref="paper",
                        yref="paper",
                        text="Close games are defined as games with a margin < 10 points",
                        showarrow=False,
                        font=dict(size=12),
                        align="center"
                    )
                ]
            )
            
            # Add annotations showing win split percentages
            for i, team in enumerate(sorted_teams['Team Name']):
                team_data = sorted_teams[sorted_teams['Team Name'] == team]
                close_pct = team_data['Close Win %'].values[0]
                
                # Add percentage annotation
                fig.add_annotation(
                    y=team,
                    x=team_data['Total Wins'].values[0] / 2,  # Middle of the stacked bar
                    text=f"{close_pct:.0%} close",
                    showarrow=False,
                    font=dict(color='white', size=12)
                )
            
            return fig
        
    # Fallback to a super simple visualization if we don't have the right data structure
    print("Falling back to simple visualization due to data structure")
    
    # We need at least team name and total wins
    if 'Team Name' in close_games.columns and 'Total Wins' in close_games.columns:
        sorted_teams = close_games.sort_values('Total Wins', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=sorted_teams['Team Name'],
            x=sorted_teams['Total Wins'],
            orientation='h',
            marker_color='#3498db',
            name='Total Wins'
        ))
        
        fig.update_layout(
            title="Team Wins",
            height=600,
            margin=dict(t=50, r=50, b=50, l=150),
            yaxis=dict(
                title="",
                autorange="reversed"
            ),
            xaxis=dict(
                title="Number of Wins"
            ),
            annotations=[
                dict(
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    text="Note: Full close game analysis not available due to data structure",
                    showarrow=False,
                    font=dict(size=12),
                    align="center"
                )
            ]
        )
        
        return fig
    
    # If all else fails, return a message
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text="Could not create Close Game Analysis due to missing data",
        showarrow=False,
        font=dict(size=16)
    )
    
    fig.update_layout(
        height=500,
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False)
    )
    
    return fig

# 11. VISUALIZATION: Custom Metric - Luck Factor Analysis
def create_luck_factor_analysis(data):
    """Create visualization analyzing team 'luck' based on points vs record."""
    standings = data['Standings']
    
    # Calculate expected wins based on points ratio
    total_points_for = standings['Points For'].sum()
    total_points_against = standings['Points Against'].sum()
    
    # Expected win % based on points ratio
    standings['Expected Win %'] = standings['Points For'] / (standings['Points For'] + standings['Points Against'])
    
    # Calculate luck factor (actual wins - expected wins)
    standings['Expected Wins'] = standings['Expected Win %'] * (standings['Wins'] + standings['Losses'])
    standings['Luck Factor'] = standings['Wins'] - standings['Expected Wins']
    
    # Sort by luck factor
    sorted_teams = standings.sort_values('Luck Factor', ascending=False)
    
    # Create the figure
    fig = go.Figure()
    
    # Add actual wins bar
    fig.add_trace(
        go.Bar(
            x=sorted_teams['Team Name'],
            y=sorted_teams['Wins'],
            name='Actual Wins',
            marker_color='#3498db'
        )
    )
    
    # Add expected wins bar
    fig.add_trace(
        go.Bar(
            x=sorted_teams['Team Name'],
            y=sorted_teams['Expected Wins'],
            name='Expected Wins',
            marker_color='#f39c12'
        )
    )
    
    # Add luck factor annotations
    for i, team in enumerate(sorted_teams['Team Name']):
        luck = sorted_teams.loc[sorted_teams['Team Name'] == team, 'Luck Factor'].values[0]
        
        color = "#2ecc71" if luck > 0 else "#e74c3c"
        
        fig.add_annotation(
            x=team,
            y=sorted_teams.loc[sorted_teams['Team Name'] == team, 'Wins'].values[0] + 0.5,
            text=f"{luck:+.1f}",
            showarrow=False,
            font=dict(color=color, size=14)
        )
    
    # Update layout
    fig.update_layout(
        title='Luck Factor Analysis: Actual vs Expected Wins',
        xaxis=dict(title='Team'),
        yaxis=dict(title='Wins'),
        barmode='group',
        height=600
    )
    
    return fig

# 12. VISUALIZATION: Schedule Strength Analysis
def create_schedule_strength_analysis(data):
    """Create a matrix showing how teams would perform with other teams' schedules."""
    matchups = data['Matchups']
    teams = data['Teams']
    
    # Get list of all teams
    team_names = teams['Team Name'].tolist()
    
    # Filter to regular season matchups
    regular_matchups = matchups[~matchups['Is Playoff']]
    
    # Create a matrix to store the win differential
    win_matrix = pd.DataFrame(0.0, index=team_names, columns=team_names)
    
    # For each team (row)
    for team_x in team_names:
        # Get team X's actual win count
        actual_wins = teams[teams['Team Name'] == team_x]['Wins'].values[0]
        
        # For each team's schedule (column)
        for team_y in team_names:
            if team_x == team_y:
                # Same team - differential is 0
                win_matrix.loc[team_x, team_y] = 0
                continue
            
            simulated_wins = 0
            
            # For each week
            for week in regular_matchups['Week'].unique():
                # Get team X's score for this week
                x_score = None
                
                # Check if team X was home that week
                x_home = regular_matchups[(regular_matchups['Week'] == week) & 
                                         (regular_matchups['Home Team'] == team_x)]
                if len(x_home) > 0:
                    x_score = x_home['Home Points'].values[0]
                
                # Check if team X was away that week
                x_away = regular_matchups[(regular_matchups['Week'] == week) & 
                                         (regular_matchups['Away Team'] == team_x)]
                if len(x_away) > 0:
                    x_score = x_away['Away Points'].values[0]
                
                # Skip if team X doesn't have a score for this week
                if x_score is None:
                    continue
                
                # Get team Y's opponent for this week
                y_opponent = None
                y_opponent_score = None
                
                # Check if team Y was home
                y_home = regular_matchups[(regular_matchups['Week'] == week) & 
                                         (regular_matchups['Home Team'] == team_y)]
                if len(y_home) > 0:
                    y_opponent = y_home['Away Team'].values[0]
                    y_opponent_score = y_home['Away Points'].values[0]
                
                # Check if team Y was away
                y_away = regular_matchups[(regular_matchups['Week'] == week) & 
                                         (regular_matchups['Away Team'] == team_y)]
                if len(y_away) > 0:
                    y_opponent = y_away['Home Team'].values[0]
                    y_opponent_score = y_away['Home Points'].values[0]
                
                # Skip if opponent info is not available
                if y_opponent is None or y_opponent_score is None:
                    continue
                
                # Skip if the opponent is team X (can't play against yourself)
                if y_opponent == team_x:
                    continue
                
                # Simulate the matchup
                if x_score > y_opponent_score:
                    simulated_wins += 1
            
            # Calculate win differential
            win_differential = simulated_wins - actual_wins
            win_matrix.loc[team_x, team_y] = win_differential
    
    # Create a discrete color scale centered at 0
    max_abs_value = max(abs(win_matrix.min().min()), abs(win_matrix.max().max()))
    
    # Create the custom discrete color scale
    # Colors for negative values (blues)
    neg_colors = ['#08306b', '#2171b5', '#4292c6', '#9ecae1']  
    # Colors for positive values (reds)
    pos_colors = ['#fee0d2', '#fc9272', '#de2d26', '#a50f15']  
    
    # Create a 9-level discrete colorscale with white at center (0)
    colors = neg_colors + ['#f7f7f7'] + pos_colors
    
    # Create discrete colorscale with values mapped to the specific numbers in the data
    # Find unique values in the win matrix
    unique_values = sorted(np.unique(win_matrix.values))
    
    # Create a custom colorscale that maps specific values to specific colors
    colorscale = []
    
    # Function to normalize values between 0 and 1 for colorscale
    def normalize_value(val, min_val, max_val):
        return (val - min_val) / (max_val - min_val)
    
    # Create a discrete colorscale with specific steps
    steps = [-7, -6, -4, -2, 0, 1, 2, 3, 4]  # Adjust these based on your actual data
    min_val = min(steps)
    max_val = max(steps)
    
    # Map each step to a color
    for i, step in enumerate(steps):
        normalized = normalize_value(step, min_val, max_val)
        colorscale.append([normalized, colors[i]])
        
        # Add another entry with the same color unless we're at the last step
        if i < len(steps) - 1:
            next_normalized = normalize_value(steps[i+1], min_val, max_val)
            mid_point = (normalized + next_normalized) / 2
            colorscale.append([mid_point - 0.0001, colors[i]])
    
    # Create heatmap with the discrete color scale
    fig = go.Figure(data=go.Heatmap(
        z=win_matrix.values,
        x=win_matrix.columns,
        y=win_matrix.index,
        text=win_matrix.values,
        texttemplate="%{text:.0f}",
        colorscale=colorscale,
        zmid=0,  # Center the color scale at 0
        zmin=min_val,
        zmax=max_val,
        showscale=True,
        colorbar=dict(
            title="Win Differential",
            tickvals=steps,
            ticktext=[str(i) for i in steps]
        ),
        hovertemplate="Row Team: %{y}<br>Column Team: %{x}<br>Win Differential: %{text:.0f}<extra></extra>"
    ))
    
    # Create the title with explanation above the chart
    title_with_explanation = (
        "Schedule Strength Analysis: Win Differential with Other Teams' Schedules<br>" +
        "<span style='font-size:12px;font-weight:normal;'>" +
        "Values show how many more wins (+) or fewer wins (-) a team would have if they played against the opponents on another team's schedule" +
        "</span>"
    )
    
    # Update layout with adjusted margins and title
    fig.update_layout(
        title=title_with_explanation,
        title_x=0.5,  # Center the title
        title_font=dict(size=16),
        height=750,
        width=1100,
        xaxis_title="<b>Team Whose Schedule Is Used</b>",  # Bold title below x-axis
        yaxis_title="Team",
        margin=dict(l=20, r=20, t=80, b=150),  # Increased top margin for title, more bottom for labels
        font=dict(size=12)
    )
    
    # Set x-axis properties for better readability
    fig.update_xaxes(
        tickangle=45,  # 45 degree angle as requested
        tickfont=dict(size=12),
        tickmode='array',
        tickvals=list(range(len(team_names))),
        ticktext=team_names,
        title_standoff=30  # Give more space between the axis and its title
    )
    
    fig.update_yaxes(
        tickfont=dict(size=12)
    )
    
    return fig

def create_consistency_wins_analysis(data):
    """
    Create visualization analyzing the relationship between team consistency and wins.
    This helps determine if more consistent teams (lower standard deviation) actually win more games.
    """
    # Extract the data we need
    standings = data['Standings']
    weekly_scores = data['Weekly Scores']
    teams = data['Teams']
    
    # Make sure we have the necessary data - calculate if needed
    if 'Score Std Dev' not in standings.columns or 'Average Score' not in standings.columns:
        # Calculate team stats from weekly scores
        team_stats = []
        for team in teams['Team Name'].unique():
            team_weekly_scores = weekly_scores[weekly_scores['Team Name'] == team]['Points']
            if len(team_weekly_scores) > 0:
                std_dev = team_weekly_scores.std()
                avg_score = team_weekly_scores.mean()
                team_stats.append({
                    'Team Name': team,
                    'Score Std Dev': std_dev,
                    'Average Score': avg_score
                })
        
        # Create or merge with standings DataFrame
        if len(team_stats) > 0:
            team_stats_df = pd.DataFrame(team_stats)
            if standings is not None:
                standings = pd.merge(standings, team_stats_df, on='Team Name', how='left')
            else:
                standings = team_stats_df
    
    # Make sure we have win data
    if 'Win %' not in standings.columns and 'Wins' in standings.columns and 'Losses' in standings.columns:
        standings['Win %'] = standings['Wins'] / (standings['Wins'] + standings['Losses'])
    
    # Create the main scatter plot
    fig = go.Figure()
    
    # Add a scatter plot of consistency (std dev) vs. win percentage
    fig.add_trace(
        go.Scatter(
            x=standings['Score Std Dev'],
            y=standings['Win %'],
            mode='markers+text',
            marker=dict(
                size=standings['Average Score'] / 3,  # Size of bubble represents average score
                color=standings['Average Score'],
                colorscale='Viridis',
                colorbar=dict(
                    title='Avg Score',
                    thickness=15,
                    len=0.7,  # Make colorbar shorter
                    y=0.5     # Center it vertically
                ),
                showscale=True
            ),
            text=standings['Team Name'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>' +
                          'Win %: %{y:.1%}<br>' +
                          'Std Dev: %{x:.2f}<br>' +
                          'Avg Score: %{marker.color:.1f}<br>' +
                          'Wins: %{customdata[0]}-Losses: %{customdata[1]}<extra></extra>',
            customdata=np.stack((standings['Wins'], standings['Losses']), axis=1),
            name='Teams'  # Give this trace a name to avoid "trace 0" in legend
        )
    )
    
    # Calculate correlation
    correlation = standings['Score Std Dev'].corr(standings['Win %'])
    
    # Calculate regression line for better visualization
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        standings['Score Std Dev'], standings['Win %']
    )
    
    # Add trend line
    x_range = np.linspace(standings['Score Std Dev'].min() * 0.9, 
                         standings['Score Std Dev'].max() * 1.1, 100)
    y_range = slope * x_range + intercept
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name=f'Trend Line (r={r_value:.2f})',
            hoverinfo='skip'
        )
    )
    
    # Add quadrant lines based on median values
    median_std = standings['Score Std Dev'].median()
    median_win = standings['Win %'].median()
    
    fig.add_vline(x=median_std, line_dash="dash", line_color="gray")
    fig.add_hline(y=median_win, line_dash="dash", line_color="gray")
    
    # Add quadrant labels - repositioned for better spacing
    fig.add_annotation(
        x=standings['Score Std Dev'].max(), 
        y=standings['Win %'].max() * 0.95, 
        text="Inconsistent,<br>High Win %", 
        showarrow=False,
        font=dict(size=12),
        align="right",
        xanchor="right"
    )
    
    fig.add_annotation(
        x=standings['Score Std Dev'].min(), 
        y=standings['Win %'].max() * 0.95, 
        text="Consistent,<br>High Win %", 
        showarrow=False,
        font=dict(size=12),
        align="left",
        xanchor="left"
    )
    
    fig.add_annotation(
        x=standings['Score Std Dev'].max(), 
        y=standings['Win %'].min() * 1.2, 
        text="Inconsistent,<br>Low Win %", 
        showarrow=False,
        font=dict(size=12),
        align="right",
        xanchor="right"
    )
    
    fig.add_annotation(
        x=standings['Score Std Dev'].min(), 
        y=standings['Win %'].min() * 1.2, 
        text="Consistent,<br>Low Win %", 
        showarrow=False,
        font=dict(size=12),
        align="left",
        xanchor="left"
    )
    
    # Update layout for main figure
    fig.update_layout(
        title='Team Consistency vs. Win Percentage Analysis',
        xaxis=dict(
            title='Standard Deviation (Lower = More Consistent)',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Win Percentage',
            tickformat='.0%',
            tickfont=dict(size=12),
            range=[0, 1]
        ),
        height=700,
        margin=dict(t=50, r=50, b=50, l=50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Create a unified dashboard with all fantasy football visualizations
def create_fantasy_dashboard():
    """Create a unified dashboard with all fantasy football visualizations."""
    file_path = 'fantasy_football.xlsx'
    print("Loading fantasy football data...")
    data = load_fantasy_data(file_path)
    
    # Define visualization functions and titles
    vis_functions = [
        # Existing visualizations
        (lambda d: create_offensive_stat_chart(d, 'Passing Yards'), "Passing Yards"),
        (lambda d: create_offensive_stat_chart(d, 'Passing TDs'), "Passing TDs"),
        (lambda d: create_offensive_stat_chart(d, 'Rushing Yards'), "Rushing Yards"),
        (lambda d: create_offensive_stat_chart(d, 'Rushing TDs'), "Rushing TDs"),
        (lambda d: create_offensive_stat_chart(d, 'Receiving Yards'), "Receiving Yards"),
        (lambda d: create_offensive_stat_chart(d, 'Receiving TDs'), "Receiving TDs"),
        (lambda d: create_offensive_stat_chart(d, 'Receptions'), "Receptions"),
        (create_weekly_performance_heatmap, "Weekly Performance Heatmap"),
        (create_consistency_analysis, "Team Consistency Analysis"),
        (create_offensive_efficiency, "Offensive Efficiency Analysis"),
        (create_points_analysis, "Win-Loss Record Analysis"),
        (create_weekly_scoring_trends, "Weekly Scoring Trends"),
        (create_home_away_analysis, "Home vs Away Performance"),
        (create_points_overall_distribution, "Overall Points Distribution"),
        (create_team_points_distribution, "Team Points Distributions"),
        (create_transaction_scatter, "Transaction Analysis"),
        (create_weekly_transactions, "Weekly Transaction Activity"),
        (create_close_game_analysis, "Close Game Analysis"),
        (create_luck_factor_analysis, "Luck Factor Analysis"),
        (create_schedule_strength_analysis, "Schedule Strength Analysis"),
        (create_consistency_wins_analysis, "Consistency vs Wins Analysis")
    ]
    
    # Create HTML content
    html_content = [
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fantasy Football Dashboard</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }
                .dashboard-container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px 5px 0 0;
                    margin-bottom: 20px;
                }
                .visualization-card {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    overflow: hidden;
                }
                .card-header {
                    background-color: #3498db;
                    color: white;
                    padding: 15px;
                    font-size: 18px;
                    font-weight: bold;
                }
                .card-content {
                    padding: 15px;
                }
                .footer {
                    text-align: center;
                    margin-top: 20px;
                    padding: 10px;
                    color: #7f8c8d;
                    font-size: 14px;
                }
                .table-of-contents {
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 30px;
                    padding: 15px;
                }
                .table-of-contents h2 {
                    margin-top: 0;
                    color: #2c3e50;
                }
                .dropdown-container {
                    margin-top: 15px;
                }
                .visualization-select {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    background-color: white;
                    font-size: 16px;
                    color: #3498db;
                    cursor: pointer;
                }
                .visualization-select:focus {
                    outline: none;
                    border-color: #3498db;
                    box-shadow: 0 0 5px rgba(52, 152, 219, 0.5);
                }
            </style>
            <script>
                function scrollToVisualization(visId) {
                    if (visId) {
                        document.getElementById(visId).scrollIntoView({ 
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            </script>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header">
                    <h1>Rylytics Dashboard</h1>
                    <p>A comprehensive analysis of Rick & Morty</p>
                </div>
                
                <div class="table-of-contents">
                    <h2>Visualizations</h2>
                    <div class="dropdown-container">
                        <select id="visualization-dropdown" class="visualization-select" onchange="scrollToVisualization(this.value)">
                            <option value="">Select a visualization...</option>
        """
    ]
    
    # Add dropdown options
    for i, (_, vis_title) in enumerate(vis_functions):
        html_content.append(f'                            <option value="vis-{i+1}">{i+1}. {vis_title}</option>')
    
    html_content.append("""
                        </select>
                    </div>
                </div>
    """)
    
    # Create each visualization and add to HTML
    for i, (vis_func, vis_title) in enumerate(vis_functions):
        print(f"Creating visualization {i+1}/{len(vis_functions)}: {vis_title}")
        
        try:
            # Create the visualization
            fig = vis_func(data)
            
            # Convert to HTML
            vis_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn' if i == 0 else False)
            
            # Add to dashboard
            html_content.append(f"""
                <div class="visualization-card" id="vis-{i+1}">
                    <div class="card-header">{i+1}. {vis_title}</div>
                    <div class="card-content">
                        {vis_html}
                    </div>
                </div>
            """)
            
        except Exception as e:
            print(f"Error creating {vis_title}: {e}")
            html_content.append(f"""
                <div class="visualization-card" id="vis-{i+1}">
                    <div class="card-header">{i+1}. {vis_title}</div>
                    <div class="card-content">
                        <p>Error creating visualization: {e}</p>
                    </div>
                </div>
            """)
    
    # Close HTML
    html_content.append("""
                <div class="footer">
                    <p>Ryltics</p>
                </div>
            </div>
        </body>
        </html>
    """)
    
    # Join all HTML content
    full_html = "\n".join(html_content)
    
    # Write to file
    output_file = "fantasy_football_dashboard.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_html)
    
    print(f"Dashboard created successfully! Open {output_file} in your browser to view.")
    
    # Get absolute path
    abs_path = os.path.abspath(output_file)
    print(f"Dashboard saved to {abs_path}")
    
    return abs_path

# Run the dashboard creator
if __name__ == "__main__":
    try:
        # Create the dashboard
        dashboard_path = create_fantasy_dashboard()
        
        # Open browser with the dashboard
        if dashboard_path:
            file_url = f"file:///{dashboard_path.replace(os.sep, '/')}"
            print(f"Opening {file_url}")
            webbrowser.open(file_url)
            print("✅ Dashboard opened in browser")
        else:
            print("❌ Error: Dashboard path is None")
            
    except Exception as e:
        print(f"❌ Error creating dashboard: {e}")
        import traceback
        traceback.print_exc()