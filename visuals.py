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
            'Points per 100 Yards': 'Efficiency (Points per 100 Yards)'
        },
        height=600
    )
    
    # Add trendline
    fig.update_layout(
        showlegend=True
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
            name=f'Average Efficiency ({avg_efficiency:.2f} pts/100yds)',
            line=dict(dash='dash', color='gray')
        )
    )
    
    return fig

# 5. VISUALIZATION: Win-Loss Record Analysis
def create_win_loss_analysis(data):
    """Create grouped bar charts showing win-loss records and points for/against."""
    teams = data['Teams']
    
    # Sort by win percentage
    teams_sorted = teams.sort_values('Win Percentage', ascending=False)
    
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Win-Loss Records', 'Points For vs. Points Against'),
                        vertical_spacing=0.12,
                        row_heights=[0.4, 0.6])
    
    # Add win-loss bars
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Wins'],
            name='Wins',
            marker_color='#2ecc71'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Losses'],
            name='Losses',
            marker_color='#e74c3c'
        ),
        row=1, col=1
    )
    
    # Add Points For vs Points Against
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Points For'],
            name='Points For',
            marker_color='#3498db'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=teams_sorted['Team Name'],
            y=teams_sorted['Points Against'],
            name='Points Against',
            marker_color='#f39c12'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Team Performance Analysis',
        barmode='group',
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
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
def create_points_distribution(data):
    """Create visualizations showing distribution of points."""
    weekly_scores = data['Weekly Scores']
    
    # Create figure with 1x2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Overall Points Distribution', 'Team-specific Distributions'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Overall distribution (left plot)
    fig.add_trace(
        go.Histogram(
            x=weekly_scores['Points'],
            nbinsx=20,
            name='All Scores',
            marker_color='#3498db'
        ),
        row=1, col=1
    )
    
    # Add mean line
    mean_score = weekly_scores['Points'].mean()
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.1f}",
        annotation_position="top right",
        row=1, col=1
    )
    
    # Team-specific distributions (right plot)
    # Get top 5 teams
    teams_df = data['Teams']
    top5_teams = teams_df.nlargest(5, 'Points For')['Team Name'].tolist()
    
    # Plot violin for each team
    for team in top5_teams:
        team_scores = weekly_scores[weekly_scores['Team Name'] == team]['Points']
        
        fig.add_trace(
            go.Violin(
                y=team_scores,
                name=team,
                box_visible=True,
                meanline_visible=True
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title='Fantasy Points Distribution Analysis',
        showlegend=True,
        height=600
    )
    
    return fig

# 9. VISUALIZATION: Transaction Analysis
def create_transaction_analysis(data):
    """Create visualizations analyzing transaction patterns."""
    transactions = data['Transactions']
    standings = data['Standings']
    
    # Merge transaction data with standings
    merged_df = pd.merge(transactions, standings[['Team ID', 'Team Name', 'Wins', 'Losses', 'Win %']], 
                         on=['Team ID', 'Team Name'])
    
    # Create figure with 1x2 subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Transaction Activity vs. Team Success', 'Weekly Transaction Activity'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Add scatter plot for transactions vs win %
    fig.add_trace(
        go.Scatter(
            x=merged_df['Total Acquisitions'],
            y=merged_df['Win %'],
            mode='markers+text',
            marker=dict(
                size=merged_df['Moves to Active']/5,
                color=merged_df['Total Trades'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Trades Made')
            ),
            text=merged_df['Team Name'],
            textposition='top center',
            name='Teams'
        ),
        row=1, col=1
    )
    
    # Calculate correlation
    correlation = merged_df['Total Acquisitions'].corr(merged_df['Win %'])
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.95,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.2f}",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        row=1, col=1
    )
    
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
    
    # Add bar chart for weekly transactions
    fig.add_trace(
        go.Bar(
            x=weekly_adds_df['Week'],
            y=weekly_adds_df['Transactions'],
            marker_color='#3498db',
            name='Weekly Transactions'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False
    )
    
    fig.update_xaxes(title_text='Total Player Acquisitions', row=1, col=1)
    fig.update_yaxes(title_text='Win Percentage', tickformat='.0%', row=1, col=1)
    
    fig.update_xaxes(title_text='Week', tickmode='linear', row=2, col=1)
    fig.update_yaxes(title_text='Number of Transactions', row=2, col=1)
    
    return fig

# 10. VISUALIZATION: Close Game Analysis
def create_close_game_analysis(data):
    """Create visualizations analyzing close games."""
    close_games = data['Close Game Analysis']
    matchups = data['Matchups']
    
    # Define what constitutes a "close game" (e.g., margin < 10 points)
    close_threshold = 10
    matchups['Is Close'] = matchups['Margin'] < close_threshold
    
    # Count close games by week
    close_by_week = matchups.groupby('Week')['Is Close'].sum().reset_index()
    close_by_week['Close Game %'] = close_by_week['Is Close'] / matchups.groupby('Week').size().values
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Team Close Game Performance', 'Weekly Close Game Trends'),
        specs=[[{'type': 'xy'}, {'type': 'xy'}]]
    )
    
    # Team close game analysis (left plot)
    # Sort teams by close win percentage
    close_games['Close Win %'] = close_games['Close Wins'] / (close_games['Close Wins'] + close_games['Blowout Wins'])
    sorted_teams = close_games.sort_values('Close Win %', ascending=False)
    
    # Create stacked bar chart
    fig.add_trace(
        go.Bar(
            x=sorted_teams['Team Name'],
            y=sorted_teams['Close Wins'],
            name='Close Wins',
            marker_color='#2ecc71'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=sorted_teams['Team Name'],
            y=sorted_teams['Blowout Wins'],
            name='Blowout Wins',
            marker_color='#3498db'
        ),
        row=1, col=1
    )
    
    # Weekly close game trends (right plot)
    fig.add_trace(
        go.Scatter(
            x=close_by_week['Week'],
            y=close_by_week['Close Game %'],
            mode='lines+markers',
            name='% Close Games',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=8)
        ),
        row=1, col=2
    )
    
    # Add average line
    avg_close = close_by_week['Close Game %'].mean()
    fig.add_hline(
        y=avg_close,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Avg: {avg_close:.0%}",
        annotation_position="bottom right",
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Close Game Analysis',
        barmode='stack',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.update_xaxes(title_text='Team', row=1, col=1)
    fig.update_yaxes(title_text='Number of Wins', row=1, col=1)
    
    fig.update_xaxes(title_text='Week', tickmode='linear', row=1, col=2)
    fig.update_yaxes(title_text='% of Close Games', tickformat='.0%', row=1, col=2)
    
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
    
    # Create heatmap
    fig = px.imshow(
        win_matrix,
        text_auto='.0f',
        color_continuous_scale='RdBu_r',  # Reversed so red is positive, blue is negative
        labels=dict(x="Schedule Used", y="Team", color="Win Differential"),
        title="Schedule Strength Analysis: Win Differential with Other Teams' Schedules"
    )
    
    # Update layout
    fig.update_layout(
        height=750,
        width=1100,
        xaxis_title="",  # Remove title from axis
        yaxis_title="Team",
        margin=dict(l=20, r=20, t=50, b=200)  # Significantly increase bottom margin
    )
    
    # Set x-axis and y-axis properties for better readability
    fig.update_xaxes(
        tickangle=90,  # Vertical text
        tickfont=dict(size=12),
        tickmode='array',
        tickvals=list(range(len(team_names))),
        ticktext=team_names
    )
    
    fig.update_yaxes(
        tickfont=dict(size=12)
    )
    
    # Add title below the chart instead of on the axis
    fig.add_annotation(
        x=0.5,
        y=-0.10,
        xref="paper",
        yref="paper",
        text="<b>Team Whose Schedule Is Used</b>",
        showarrow=False,
        align="center",
        font=dict(size=14)
    )
    
    # Add annotation explaining the matrix
    fig.add_annotation(
        x=0.5,
        y=-0.20,  # Moved lower for more space
        xref="paper",
        yref="paper",
        text="Values show how many more wins (+) or fewer wins (-) a team would have<br>if they played against the opponents on another team's schedule",
        showarrow=False,
        align="center"
    )
    
    return fig



# Update the create_fantasy_dashboard function to include these new visualizations
def create_fantasy_dashboard():
    """Create a unified dashboard with all fantasy football visualizations."""
    file_path = 'fantasy_football.xlsx'
    print("Loading fantasy football data...")
    data = load_fantasy_data(file_path)





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
        (create_win_loss_analysis, "Win-Loss Record Analysis"),
        (create_weekly_scoring_trends, "Weekly Scoring Trends"),
        (create_home_away_analysis, "Home vs Away Performance"),
        (create_points_distribution, "Points Distribution Analysis"),
        (create_transaction_analysis, "Transaction Analysis"),
        (create_close_game_analysis, "Close Game Analysis"),
        (create_luck_factor_analysis, "Luck Factor Analysis"),
        (create_schedule_strength_analysis, "Schedule Strength Analysis"),

        # New visualizations
    ]
    
    # Rest of the function remains the same...
    
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
                .table-of-contents ul {
                    list-style-type: none;
                    padding-left: 0;
                }
                .table-of-contents li {
                    margin-bottom: 10px;
                }
                .table-of-contents a {
                    color: #3498db;
                    text-decoration: none;
                    display: block;
                    padding: 8px;
                    border-radius: 3px;
                    transition: background-color 0.3s;
                }
                .table-of-contents a:hover {
                    background-color: #f0f0f0;
                }
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header">
                    <h1>Fantasy Football Analytics Dashboard</h1>
                    <p>A comprehensive analysis of your fantasy football league</p>
                </div>
                
                <div class="table-of-contents">
                    <h2>Visualizations</h2>
                    <ul>
        """
    ]
    
    # Add table of contents
    for i, (_, vis_title) in enumerate(vis_functions):
        html_content.append(f'<li><a href="#vis-{i+1}">{i+1}. {vis_title}</a></li>')
    
    html_content.append("""
                    </ul>
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
                    <p>Fantasy Football Dashboard | Created with Python, Plotly, and Pandas</p>
                </div>
            </div>
        </body>
        </html>
    """)
    
    # Join all HTML content
    full_html = "\n".join(html_content)
    
    # Write to file
    output_file = "index.html"
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

