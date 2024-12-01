import os
import logging
import pickle
import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from scipy.stats import poisson

# 固定 API 密钥
API_KEY = '0c2379b28acb446bb97bd417f2666f81'  # 请替换为你的实际 API 密钥

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_leagues(self):
        url = 'https://api.football-data.org/v4/competitions/'
        headers = {'X-Auth-Token': self.api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            competitions = response.json().get('competitions', [])
            return {competition['name']: competition['id'] for competition in competitions}
        else:
            logging.error(f"无法获取联赛数据，状态码: {response.status_code}, 响应内容: {response.text}")
            st.error(f"无法获取联赛数据，状态码: {response.status_code}")
            return {}

    def get_upcoming_matches(self, league_id):
        url = f'https://api.football-data.org/v4/competitions/{league_id}/matches?status=SCHEDULED'
        headers = {'X-Auth-Token': self.api_key}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json().get('matches', [])
        else:
            logging.error(f"无法获取即将进行的比赛，状态码: {response.status_code}, 响应内容: {response.text}")
            st.error(f"无法获取即将进行的比赛，状态码: {response.status_code}")
            return []

    def get_team_history(self, team_id):
        cache_file = f"cache/team_{team_id}_history.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            uri = f'https://api.football-data.org/v4/teams/{team_id}/matches'
            headers = {'X-Auth-Token': self.api_key}
            response = requests.get(uri, headers=headers)
            if response.status_code == 200:
                matches = response.json().get('matches', [])
                history = [
                    (match['homeTeam']['id'], match['score']['fullTime']['home'], match['score']['fullTime']['away']) if match['homeTeam']['id'] == team_id else 
                    (match['awayTeam']['id'], match['score']['fullTime']['away'], match['score']['fullTime']['home'])
                    for match in matches
                    if match['score']['fullTime']['home'] is not None and match['score']['fullTime']['away'] is not None
                ]
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(history, f)
                return history
            else:
                logging.error(f"无法获取历史比赛数据，状态码: {response.status_code}, 响应内容: {response.text}")
                st.error(f"无法获取历史比赛数据，状态码: {response.status_code}")
                return []

def poisson_prediction(avg_goals, max_goals=5):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

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

st.title('预测结果仅供参考')

st.sidebar.title("输入参数设置")

fetcher = DataFetcher(API_KEY)

leagues = fetcher.get_leagues()
if leagues:
    selected_league_name = st.sidebar.selectbox('选择联赛', list(leagues.keys()))
    league_id = leagues[selected_league_name]

    matches = fetcher.get_upcoming_matches(league_id)
    if matches:
        match_options = {f"{match['homeTeam']['name']} vs {match['awayTeam']['name']} - {match['utcDate']}": match for match in matches}
        selected_match_key = st.sidebar.selectbox('选择即将进行的比赛', list(match_options.keys()))
        confirm_button = st.sidebar.button("确认选择")

        if confirm_button:
            selected_match_data = match_options[selected_match_key]
            home_team_id = selected_match_data['homeTeam']['id']
            away_team_id = selected_match_data['awayTeam']['id']

            home_history = fetcher.get_team_history(home_team_id)
            away_history = fetcher.get_team_history(away_team_id)

            home_avg_goals = np.mean([goals for _, goals, _ in home_history]) if home_history else 0
            away_avg_goals = np.mean([goals for _, _, goals in away_history]) if away_history else 0

            home_goals_prob = poisson_prediction(home_avg_goals, max_goals=5)
            away_goals_prob = poisson_prediction(away_avg_goals, max_goals=5)

            score_probs = score_probability(home_goals_prob, away_goals_prob)

            # 计算胜平负概率
            home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob)

            st.header("⚽ 预测结果")
            st.write(f"主队的平均进球数: {home_avg_goals:.2f}")
            st.write(f"客队的平均进球数: {away_avg_goals:.2f}")

            # 创建表格数据
            table_data = []
            for i, home_prob in enumerate(home_goals_prob):
                for j, away_prob in enumerate(away_goals_prob):
                    row = [i, j, i + j, score_probs[i, j]]
                    table_data.append(row)

            # 将表格数据转换为DataFrame
            df = pd.DataFrame(table_data, columns=["主队进球数", "客队进球数", "总进球数", "概率"])
            st.write("进球概率表格:")
            st.write(df)

            # 显示胜平负概率
            st.write(f"主队胜的概率: {home_win_prob:.2%}")
            st.write(f"平局的概率: {draw_prob:.2%}")
            st.write(f"客队胜的概率: {away_win_prob:.2%}")

        else:
            st.error("请确认选择即将进行的比赛。")
    else:
        st.error("未能加载即将进行的比赛，请检查 API。")
else:
    st.error("没有可用的联赛数据。")
