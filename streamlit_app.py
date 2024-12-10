import os
import logging
import joblib
import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import lru_cache

# 设置中文字体，确保可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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
        return self.get_data_with_retries(url, headers)

    def get_teams_in_league(self, league_id):
        url = f'https://api.football-data.org/v4/competitions/{league_id}/teams'
        headers = {'X-Auth-Token': self.api_key}
        return self.get_data_with_retries(url, headers)

    def get_team_history(self, team_id, limit=6):
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
                    for match in matches[:limit]  
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

# 初始化 DataFetcher 实例
fetcher = DataFetcher(API_KEY)

@st.cache_data
def cache_get_leagues(api_key):
    fetcher = DataFetcher(api_key)
    return fetcher.get_leagues()

@st.cache_data
def cache_get_teams_in_league(api_key, league_id):
    fetcher = DataFetcher(api_key)
    return fetcher.get_teams_in_league(league_id)

@st.cache_data
def cache_get_team_history(api_key, team_id, limit=6):
    fetcher = DataFetcher(api_key)
    return fetcher.get_team_history(team_id, limit)

def poisson_prediction(avg_goals, max_goals=6):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

def calculate_weighted_average_goals(history, n=10):
    if len(history) == 0:
        return 0
    recent_performances = history[-n:]
    return np.mean([match[1] for match in recent_performances])

def calculate_average_goals(home_history, away_history):
    avg_home_goals = calculate_weighted_average_goals(home_history)
    avg_away_goals = calculate_weighted_average_goals(away_history)
    return avg_home_goals, avg_away_goals

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

def train_models(home_history, away_history):
    home_goals = [goals for _, goals, _ in home_history]
    away_goals = [goals for _, _, goals in away_history]
    
    X_home = np.arange(len(home_goals)).reshape(-1, 1)
    y_home = np.array(home_goals)

    X_away = np.arange(len(away_goals)).reshape(-1, 1)
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

    lgb_home = lgb.LGBMRegressor(n_estimators=100, min_data_in_leaf=5)  
    lgb_home.fit(X_home, y_home)

    lgb_away = lgb.LGBMRegressor(n_estimators=100, min_data_in_leaf=5)  
    lgb_away.fit(X_away, y_away)

    return (rf_home, xgb_home, lgb_home), (rf_away, xgb_away, lgb_away)

st.title('⚽ 足球比赛进球数预测')

st.sidebar.title("输入参数设置")

# 获取联赛数据
leagues_data = cache_get_leagues(API_KEY)
if leagues_data:
    leagues = {league['name']: league['id'] for league in leagues_data['competitions']}
    selected_league_name = st.sidebar.selectbox('选择联赛', list(leagues.keys()))
    league_id = leagues[selected_league_name]

    teams_data = cache_get_teams_in_league(API_KEY, league_id)
    if teams_data:
        teams = {team['name']: team['id'] for team in teams_data['teams']}
        selected_home_team_name = st.sidebar.selectbox('选择主队', list(teams.keys()))
        selected_away_team_name = st.sidebar.selectbox('选择客队', list(teams.keys()))

        confirm_button = st.sidebar.button("确认选择")
        point_handicap = st.sidebar.number_input('输入受让/让球盘口', min_value=-5.0, max_value=5.0, value=0.0)
        total_goals_line = st.sidebar.number_input('输入大小球盘口', min_value=0.0, max_value=10.0, value=0.0)

        if confirm_button:
            with st.spinner("正在加载数据..."):
                home_team_id = teams[selected_home_team_name]
                away_team_id = teams[selected_away_team_name]

                home_history = cache_get_team_history(API_KEY, home_team_id, limit=6)
                away_history = cache_get_team_history(API_KEY, away_team_id, limit=6)
                avg_home_goals, avg_away_goals = calculate_average_goals(home_history, away_history)

                home_models, away_models = train_models(home_history, away_history)

                predictions_home = []
                predictions_away = []
                for model in home_models:
                    predictions_home.append(model.predict([[0]])[0])
                for model in away_models:
                    predictions_away.append(model.predict([[0]])[0])

                predicted_home_goals = np.mean(predictions_home)
                predicted_away_goals = np.mean(predictions_away)

                st.header("⚽ 预测结果")
                st.markdown(f"<h3 style='color: green;'>预测主队进球数: {predicted_home_goals:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: green;'>预测客队进球数: {predicted_away_goals:.2f}</h3>", unsafe_allow_html=True)

                home_goals_prob = poisson_prediction(predicted_home_goals)
                away_goals_prob = poisson_prediction(predicted_away_goals)
                total_goals_prob = calculate_total_goals_prob(home_goals_prob, away_goals_prob)

                home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob)

                home_odds, draw_odds, away_odds = calculate_odds(home_win_prob, draw_prob, away_win_prob)

                st.header("⚽ 比赛结果概率")
                st.write(f"主队胜的概率: {home_win_prob:.2%}")
                st.write(f"平局的概率: {draw_prob:.2%}")
                st.write(f"客队胜的概率: {away_win_prob:.2%}")

                st.header("📈 博彩建议")
                total_goals_line_int = int(total_goals_line)
                if np.sum(total_goals_prob[total_goals_line_int:]) > 0.5:
                    st.write("建议：投注总进球数大于或等于盘口")
                else:
                    st.write("建议：投注总进球数小于盘口")

                if predicted_home_goals > predicted_away_goals:
                    st.write(f"建议：投注主队让{point_handicap}球胜")
                elif predicted_home_goals < predicted_away_goals:
                    st.write(f"建议：投注客队受{point_handicap}球胜")
                else:
                    st.write("建议：投注平局")

                st.write(f"主队胜的赔率: {home_odds:.2f}")
                st.write(f"平局的赔率: {draw_odds:.2f}")
                st.write(f"客队胜的赔率: {away_odds:.2f}")

                columns = [f'客队进球数 {i}' for i in range(len(away_goals_prob))]
                index = [f'主队进球数 {i}' for i in range(len(home_goals_prob))]
                score_probs_df = pd.DataFrame(score_probability(home_goals_prob, away_goals_prob), 
                                               columns=columns, index=index)

                # 将概率乘以100并保留两位小数
                score_probs_df *= 100
                score_probs_df = score_probs_df.round(2)

                # 将比分表格转换为热力图
                st.header("📈 比分概率热力图")
                fig, ax = plt.subplots(figsize=(10, 8))
                cmap = cm.viridis  # 使用渐变色
                im = ax.imshow(score_probs_df, cmap=cmap, interpolation='nearest')
                fig.colorbar(im, ax=ax)

                # 设置 x 和 y 轴标签
                ax.set_xticks(np.arange(len(columns)))
                ax.set_yticks(np.arange(len(index)))
                ax.set_xticklabels(columns,fontsize=10)
                ax.set_yticklabels(index,fontsize=10)

                # 在热力图上显示数值
                for i in range(score_probs_df.shape[0]):
                    for j in range(score_probs_df.shape[1]):
                        ax.text(j, i, f"{score_probs_df.iloc[i, j]:.2f}", ha="center", va="center", color="r",fontsize=20)

                # 显示图形
                st.pyplot(fig)

                # 显示各队进球数概率
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

                # 显示总进球数概率
                st.header("⚽ 总进球数概率")
                for total_goals, prob in enumerate(total_goals_prob):
                    if prob > 0:
                        st.write(f"总进球数: {total_goals}, 概率: {prob * 100:.2f}%")

        else:
            st.error("未能加载该联赛的球队，请检查 API。")
    else:
        st.error("没有可用的联赛数据。")
else:
    st.error("无法连接到足球数据 API。")
