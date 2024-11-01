import streamlit as st
import numpy as np
import pandas as pd

# Elo rating functions
def initialize_elo(team_names):
    return {team: 1500 for team in team_names}

def calculate_elo(current_elo, opponent_elo, result, k=32):
    expected_score = 1 / (1 + 10 ** ((opponent_elo - current_elo) / 400))
    new_elo = current_elo + k * (result - expected_score)
    return new_elo

def update_elo(elo_ratings, match_results):
    for match in match_results:
        home_team, away_team, result = match
        if result == '胜':
            home_result, away_result = 1, 0
        elif result == '平':
            home_result, away_result = 0.5, 0.5
        else:
            home_result, away_result = 0, 1
        
        elo_ratings[home_team] = calculate_elo(elo_ratings[home_team], elo_ratings[away_team], home_result)
        elo_ratings[away_team] = calculate_elo(elo_ratings[away_team], elo_ratings[home_team], away_result)
    
    return elo_ratings

# Function to simulate handicap data
def simulate_handicap_data(n_samples=1000):
    np.random.seed(42)
    features = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    }
    target = np.random.choice([0, 1], size=n_samples)
    data = pd.DataFrame(features)
    data['result'] = target
    return data

# Function to estimate average goals
def estimate_avg_goals(goals, conceded, home_avg_goals, away_avg_goals, home_avg_conceded, away_avg_conceded, correlation):
    avg_goals = (np.mean(goals) + home_avg_goals) * correlation
    avg_conceded = (np.mean(conceded) + away_avg_conceded) * correlation
    return avg_goals, avg_conceded

# Function to generate scores
def generate_scores(avg_goals, size, factors):
    adjusted_goals = avg_goals * factors
    return np.random.poisson(adjusted_goals, size)

# Function to analyze data
def analyze_data(data, column_name):
    counts = data.value_counts()
    percentages = data.value_counts(normalize=True) * 100
    return pd.DataFrame({column_name: counts.index, '次数': counts.values, '百分比': percentages.values})

# Function to calculate odds
def calculate_odds(stats):
    stats['赔率'] = 100 / stats['百分比']
    return stats

# Function to calculate handicap results
def calculate_handicap(home_goals, away_goals, handicap):
    if home_goals + handicap > away_goals:
        return '胜'
    elif home_goals + handicap < away_goals:
        return '负'
    else:
        return '平'

# Calculate handicap results and analyze
def analyze_handicaps(handicaps, home_goals_list, away_goals_list):
    handicap_results = {}
    
    for handicap in handicaps:
        results = [calculate_handicap(home_goals, away_goals, handicap) for home_goals, away_goals in zip(home_goals_list, away_goals_list)]
        results_analysis = analyze_data(pd.Series(results), f'盘口为{handicap}的结果')
        results_odds = calculate_odds(results_analysis)
        handicap_results[handicap] = results_odds
    
    return handicap_results

# Streamlit application
st.title('足球比赛模拟器')

# 在侧边栏添加用户输入
st.sidebar.title("输入参数")
home_avg_goals = st.sidebar.number_input('主队场均进球', value=1.0, format="%.1f")
away_avg_goals = st.sidebar.number_input('客队场均进球', value=1.0, format="%.1f")
home_avg_conceded = st.sidebar.number_input('主队场均失球', value=1.0, format="%.1f")
away_avg_conceded = st.sidebar.number_input('客队场均失球', value=1.0, format="%.1f")
correlation = st.sidebar.slider('进球相关性', 0.0, 1.0, 0.8)
n_simulations = st.sidebar.number_input('模拟次数', value=500000, step=10000)
selected_handicap = st.sidebar.select_slider('选择盘口', options=np.arange(-5, 5.5, 0.25), value=0.0)

# 模拟数据
weather_factors = np.random.normal(1.0, 0.1, n_simulations)
team_factors = np.random.normal(1.0, 0.1, n_simulations)
home_away_factors = np.random.normal(1.0, 0.05, n_simulations)
card_factors = np.random.normal(1.0, 0.05, n_simulations)

home_goals_list = generate_scores(home_avg_goals, n_simulations, weather_factors * team_factors * home_away_factors * card_factors)
away_goals_list = generate_scores(away_avg_goals, n_simulations, weather_factors * team_factors / home_away_factors * card_factors)

# 计算比赛结果
match_results = np.where(home_goals_list > away_goals_list, '胜', np.where(home_goals_list < away_goals_list, '负', '平'))

# 分析比赛结果
results_analysis = analyze_data(pd.Series(match_results), '比赛结果')

# 计算总进球数
total_goals_list = home_goals_list + away_goals_list

# 计算比分
match_scores_list = [f"{hg}-{ag}" for hg, ag in zip(home_goals_list, away_goals_list)]

# 分析结果
home_goals_analysis = analyze_data(pd.Series(home_goals_list), '进球数')
away_goals_analysis = analyze_data(pd.Series(away_goals_list), '进球数')
total_goals_analysis = analyze_data(pd.Series(total_goals_list), '总进球数')
match_scores_analysis = analyze_data(pd.Series(match_scores_list), '比分')

# 计算赔率
results_odds = calculate_odds(results_analysis)
home_goals_odds = calculate_odds(home_goals_analysis)
away_goals_odds = calculate_odds(away_goals_analysis)
total_goals_odds = calculate_odds(total_goals_analysis)
match_scores_odds = calculate_odds(match_scores_analysis).nlargest(10, '百分比')

# 输出表格
st.header("比赛结果统计")
st.write(results_odds)

st.header("主队进球数统计")
st.write(home_goals_odds)

st.header("客队进球数统计")
st.write(away_goals_odds)

st.header("总进球数统计")
st.write(total_goals_odds)

st.header("比分统计（前十）")
st.write(match_scores_odds)

# Handicap analysis table
st.header(f"盘口为 {selected_handicap} 的结果统计")
handicap_results = analyze_handicaps([selected_handicap], home_goals_list, away_goals_list)[selected_handicap]
st.write(handicap_results)
