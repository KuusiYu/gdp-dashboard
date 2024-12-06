import os
import logging
import joblib
import streamlit as st
import requests
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.api import GLM, families
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

# 固定 API 密钥
API_KEY = '0c2379b28acb446bb97bd417f2666f81'  # 请替换为你的实际 API 密钥

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_data_with_retries(self, url, headers, retries=5, delay=2):
        for attempt in range(retries):
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logging.warning("API 429: Too Many Requests. Retrying...")
                time.sleep(delay)
            else:
                logging.error(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")
                st.error(f"请求失败，状态码: {response.status_code}")
                return None
        return None

    def get_leagues(self):
        url = 'https://api.football-data.org/v4/competitions/'
        headers = {'X-Auth-Token': self.api_key}
        data = self.get_data_with_retries(url, headers)
        if data:
            competitions = data.get('competitions', [])
            return {competition['name']: competition['id'] for competition in competitions}
        return {}

    def get_teams_in_league(self, league_id):
        url = f'https://api.football-data.org/v4/competitions/{league_id}/teams'
        headers = {'X-Auth-Token': self.api_key}
        data = self.get_data_with_retries(url, headers)
        if data:
            teams = data.get('teams', [])
            return {team['name']: team['id'] for team in teams}
        return {}

    def get_team_history(self, team_id, limit=5):
        cache_file = f"cache/team_{team_id}_history.joblib"
        if os.path.exists(cache_file):
            logging.info(f"Loading team history from cache: {cache_file}")
            return joblib.load(cache_file)[:limit]
        else:
            uri = f'https://api.football-data.org/v4/teams/{team_id}/matches'
            headers = {'X-Auth-Token': self.api_key}
            data = self.get_data_with_retries(uri, headers)
            if data:
                matches = data.get('matches', [])
                history = [
                    (match['homeTeam']['id'], match['score']['fullTime']['home'], match['score']['fullTime']['away']) if match['homeTeam']['id'] == team_id else 
                    (match['awayTeam']['id'], match['score']['fullTime']['away'], match['score']['fullTime']['home'])
                    for match in matches
                    if match['score']['fullTime']['home'] is not None and match['score']['fullTime']['away'] is not None
                ]
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                joblib.dump(history, cache_file)
                logging.info(f"Saved team history to cache: {cache_file}")
                return history[:limit]
            else:
                logging.error("无法获取历史比赛数据")
                st.error("无法获取历史比赛数据")
                return []

def poisson_prediction(avg_goals, max_goals=5):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

def calculate_weighted_average_goals(history, n=5):
    if len(history) == 0:
        return 0
    recent_performances = history[-n:]
    return np.mean([match[1] for match in recent_performances])

def calculate_average_goals(home_history, away_history):
    avg_home_goals = calculate_weighted_average_goals(home_history)
    avg_away_goals = calculate_weighted_average_goals(away_history)
    return avg_home_goals, avg_away_goals

def adjust_goals_for_injuries(avg_goals, injury_factor):
    return avg_goals * injury_factor

def calculate_total_goals_prob(home_goals_prob, away_goals_prob):
    max_goals = len(home_goals_prob) + len(away_goals_prob) - 2
    total_goals_prob = np.zeros(max_goals + 1)
    
    for i in range(len(home_goals_prob)):
        for j in range(len(away_goals_prob)):
            total_goals = i + j
            total_goals_prob[total_goals] += home_goals_prob[i] * away_goals_prob[j]

    return total_goals_prob

def score_probability(home_goals_prob, away_goals_prob):
    score_probs = np.zeros((len(home_goals_prob), len(away_goals_prob)))
    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            score_probs[i, j] = home_prob * away_prob
    return score_probs

def calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob):
    home_win_prob = sum(home_goals_prob[i] * sum(away_goals_prob[j] for j in range(i)) for i in range(len(home_goals_prob)))  
    draw_prob = sum(home_goals_prob[i] * away_goals_prob[i] for i in range(len(home_goals_prob)))  
    away_win_prob = sum(away_goals_prob[j] * sum(home_goals_prob[i] for i in range(j)) for j in range(len(away_goals_prob)))  
    return home_win_prob, draw_prob, away_win_prob

def calculate_odds(home_win_prob, draw_prob, away_win_prob):
    home_odds = 1 / home_win_prob if home_win_prob > 0 else float('inf')
    draw_odds = 1 / draw_prob if draw_prob > 0 else float('inf')
    away_odds = 1 / away_win_prob if away_win_prob > 0 else float('inf')
    
    return home_odds, draw_odds, away_odds

def train_models(home_history, away_history, home_degree, away_degree, home_betweenness, away_betweenness, home_closeness, away_closeness):
    home_goals = [goals for _, goals, _ in home_history]
    away_goals = [goals for _, _, goals in away_history]
    
    X_home = np.column_stack((
        np.arange(len(home_goals)),
        [home_degree] * len(home_goals),
        [home_betweenness] * len(home_goals),
        [home_closeness] * len(home_goals)
    ))
    y_home = np.array(home_goals)

    X_away = np.column_stack((
        np.arange(len(away_goals)),
        [away_degree] * len(away_goals),
        [away_betweenness] * len(away_goals),
        [away_closeness] * len(away_goals)
    ))
    y_away = np.array(away_goals)

    scaler = StandardScaler()
    X_home = scaler.fit_transform(X_home)
    X_away = scaler.transform(X_away)

    rf_home = RandomForestRegressor(n_estimators=100)
    rf_home.fit(X_home, y_home)

    rf_away = RandomForestRegressor(n_estimators=100)
    rf_away.fit(X_away, y_away)

    xgb_home = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
    xgb_home.fit(X_home, y_home)

    xgb_away = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror')
    xgb_away.fit(X_away, y_away)

    lgb_home = lgb.LGBMRegressor(n_estimators=100)
    lgb_home.fit(X_home, y_home)

    lgb_away = lgb.LGBMRegressor(n_estimators=100)
    lgb_away.fit(X_away, y_away)

    model_home = GLM(y_home, X_home, family=families.NegativeBinomial()).fit()
    model_away = GLM(y_away, X_away, family=families.NegativeBinomial()).fit()

    gbr_home = GradientBoostingRegressor(n_estimators=100)
    gbr_home.fit(X_home, y_home)

    gbr_away = GradientBoostingRegressor(n_estimators=100)
    gbr_away.fit(X_away, y_away)

    return (rf_home, xgb_home, lgb_home, model_home, gbr_home), (rf_away, xgb_away, lgb_away, model_away, gbr_away)

def build_team_graph(home_history, away_history):
    G = nx.Graph()
    home_teams = set([match[0] for match in home_history])
    away_teams = set([match[0] for match in away_history])
    all_teams = home_teams.union(away_teams)
    
    for team in all_teams:
        G.add_node(team)
    
    for match in home_history:
        if match[0] in G and match[2] > 0:  
            G.add_edge(match[0], match[1], weight=match[2])
    
    for match in away_history:
        if match[1] in G and match[2] > 0:  
            G.add_edge(match[1], match[0], weight=match[2])
    
    return G

def analyze_graph(G):
    degrees = dict(nx.degree(G))
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    return degrees, betweenness, closeness

def build_and_train_model(X_train, y_train):
    X_train = np.array(X_train, ndmin=2)  # 确保 X_train 是二维数组
    y_train = np.array(y_train, ndmin=1)  # 确保 y_train 是一维数组

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['mae'])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    return model

def prepare_data(home_history, away_history, home_degree, away_degree, home_betweenness, away_betweenness, home_closeness, away_closeness):
    avg_home_goals, avg_away_goals = calculate_average_goals(home_history, away_history)
    
    injury_factor_home = 1.0
    injury_factor_away = 1.0

    avg_home_goals = adjust_goals_for_injuries(avg_home_goals, injury_factor_home)
    avg_away_goals = adjust_goals_for_injuries(avg_away_goals, injury_factor_away)

    # 输入特征应是二维数组
    feature_vector = np.array([
        [1, home_degree, home_betweenness, home_closeness]
    ])

    return feature_vector

st.title('⚽ 足球比赛进球数预测')

st.sidebar.title("输入参数设置")

fetcher = DataFetcher(API_KEY)

leagues = fetcher.get_leagues()
if leagues:
    selected_league_name = st.sidebar.selectbox('选择联赛', list(leagues.keys()))
    league_id = leagues[selected_league_name]

    teams = fetcher.get_teams_in_league(league_id)
    if teams:
        selected_home_team_name = st.sidebar.selectbox('选择主队', list(teams.keys()))
        selected_away_team_name = st.sidebar.selectbox('选择客队', list(teams.keys()))

        confirm_button = st.sidebar.button("确认选择")

        if confirm_button:
            with st.spinner("正在加载数据..."):
                home_team_id = teams[selected_home_team_name]
                away_team_id = teams[selected_away_team_name]

                home_history = fetcher.get_team_history(home_team_id, limit=5)
                away_history = fetcher.get_team_history(away_team_id, limit=5)

                G = build_team_graph(home_history, away_history)
                degrees, betweenness, closeness = analyze_graph(G)

                home_degree = degrees.get(home_team_id, 0)
                away_degree = degrees.get(away_team_id, 0)
                home_betweenness = betweenness.get(home_team_id, 0)
                away_betweenness = betweenness.get(away_team_id, 0)
                home_closeness = closeness.get(home_team_id, 0)
                away_closeness = closeness.get(away_team_id, 0)

                features = prepare_data(home_history, away_history, 
                                        home_degree, away_degree, 
                                        home_betweenness, away_betweenness, 
                                        home_closeness, away_closeness)

                home_models, away_models = train_models(home_history, away_history, 
                                                        home_degree, away_degree, 
                                                        home_betweenness, away_betweenness, 
                                                        home_closeness, away_closeness)

                predictions = []
                for model in home_models:
                    predictions.append(model.predict(features)[0])
                
                nn_model = build_and_train_model(features, [np.mean([match[1] for match in home_history])])
                predictions.append(nn_model.predict(features)[0][0])

                combined_prediction = np.mean(predictions)

                st.header("⚽ 预测结果")
                st.markdown(f"<h3 style='color: green;'>预测主队进球数: {combined_prediction:.2f}</h3>", unsafe_allow_html=True)

                home_goals_prob = poisson_prediction(np.mean([match[1] for match in home_history]))
                away_goals_prob = poisson_prediction(np.mean([match[2] for match in away_history]))

                total_goals_prob = calculate_total_goals_prob(home_goals_prob, away_goals_prob)

                home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob)

                home_odds, draw_odds, away_odds = calculate_odds(home_win_prob, draw_prob, away_win_prob)

                st.header("⚽ 中心性指标")
                st.write(f"**主队度数中心性:** {home_degree:.2f}")
                st.write(f"**客队度数中心性:** {away_degree:.2f}")
                st.write(f"**主队介数中心性:** {home_betweenness:.2f}")
                st.write(f"**客队介数中心性:** {away_betweenness:.2f}")
                st.write(f"**主队接近中心性:** {home_closeness:.2f}")
                st.write(f"**客队接近中心性:** {away_closeness:.2f}")

                columns = [f'客队进球数 {i}' for i in range(len(away_goals_prob))]
                index = [f'主队进球数 {i}' for i in range(len(home_goals_prob))]
                score_probs_df = pd.DataFrame(score_probability(home_goals_prob, away_goals_prob), 
                                               columns=columns, index=index)

                # 将概率乘以100并保留两位小数
                score_probs_df *= 100
                score_probs_df = score_probs_df.round(2)
                score_probs_df = score_probs_df.applymap(lambda x: f"{x:.2f}%").applymap(lambda x: x.rstrip('0').rstrip('%') if '.' in x else x)

                st.write("#### 进球数概率统计表 (%):")
                styled_df = score_probs_df.style.background_gradient(cmap='Blues', low=0, high=1)
                st.dataframe(styled_df)

                st.header("⚽ 各队进球数概率")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"{selected_home_team_name} 进球数概率:")
                    for i, prob in enumerate(home_goals_prob):
                        st.write(f"进球数 {i}: 概率 {prob * 100:.2f}%")

                with col2:
                    st.subheader(f"{selected_away_team_name} 进球数概率:")
                    for i, prob in enumerate(away_goals_prob):
                        st.write(f"进球数 {i}: 概率 {prob * 100:.2f}%")

                st.header("⚽ 总进球数概率")
                for total_goals, prob in enumerate(total_goals_prob):
                    if prob > 0:
                        st.write(f"总进球数: {total_goals}, 概率: {prob * 100:.2f}%")

                st.write("#### 胜平负概率:")
                st.write(f"主队胜的概率: {home_win_prob:.2%}")
                st.write(f"平局的概率: {draw_prob:.2%}")
                st.write(f"客队胜的概率: {away_win_prob:.2%}")

                st.header("📈 博彩建议")
                
                total_goals_avg = np.argmax(total_goals_prob)
                if total_goals_avg >= 2.5:
                    st.write("建议：投注总进球数大于2.5")
                else:
                    st.write("建议：投注总进球数小于2.5")
                # 让球建议
                if abs(home_win_prob - away_win_prob) < 0.1:  # 如果两队实力接近
                    st.write("建议：选择平局或小额投注")
                elif home_win_prob > away_win_prob:
                    st.write(f"建议：主队让球，推荐投注主队胜，赔率为 {home_odds:.2f}")
                else:
                    st.write(f"建议：客队让球，推荐投注客队胜，赔率为 {away_odds:.2f}")

    else:
        st.error("未能加载该联赛的球队，请检查 API。")
else:
    st.error("没有可用的联赛数据。")
