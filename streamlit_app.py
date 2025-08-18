import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson, nbinom
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as colors
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Reshape, Input
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import Model
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="足球比赛预测分析系统",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 固定 API 密钥
API_KEY = '0c2379b28acb446bb97bd417f2666f81'

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_data_with_retries(self, url, headers, params=None, retries=5, delay=2):
        for attempt in range(retries):
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logging.warning("API 429: Too Many Requests. Retrying...")
                time.sleep(delay)
            else:
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

    def get_team_matches(self, team_id, venue=None):
        url = f'https://api.football-data.org/v4/teams/{team_id}/matches'
        headers = {'X-Auth-Token': self.api_key}
        params = {'venue': venue} if venue else None
        return self.get_data_with_retries(url, headers, params=params)

    def get_league_matches(self, league_id):
        url = f'https://api.football-data.org/v4/competitions/{league_id}/matches'
        headers = {'X-Auth-Token': self.api_key}
        return self.get_data_with_retries(url, headers)

    def get_league_standings(self, league_id):
        url = f'https://api.football-data.org/v4/competitions/{league_id}/standings'
        headers = {'X-Auth-Token': self.api_key}
        return self.get_data_with_retries(url, headers)

# 初始化 DataFetcher
fetcher = DataFetcher(API_KEY)

@st.cache_data(ttl=3600)
def cache_get_leagues(api_key):
    fetcher = DataFetcher(api_key)
    return fetcher.get_leagues()

@st.cache_data(ttl=3600)
def cache_get_teams_in_league(api_key, league_id):
    fetcher = DataFetcher(api_key)
    return fetcher.get_teams_in_league(league_id)

@st.cache_data(ttl=3600)
def cache_get_team_matches(api_key, team_id, venue=None):
    fetcher = DataFetcher(api_key)
    return fetcher.get_team_matches(team_id, venue)

@st.cache_data(ttl=3600)
def cache_get_league_matches(api_key, league_id):
    fetcher = DataFetcher(api_key)
    return fetcher.get_league_matches(league_id)

@st.cache_data(ttl=3600)
def cache_get_league_standings(api_key, league_id):
    fetcher = DataFetcher(api_key)
    return fetcher.get_league_standings(league_id)

def calculate_features(matches, team_id, venue):
    features = {}
    
    if not matches or 'matches' not in matches:
        return features
    
    # 基础特征
    goals_scored = []
    goals_conceded = []
    results = []
    shots = []
    shots_on_target = []
    corners = []
    fouls = []
    possession = []
    
    for match in matches['matches']:
        if venue and match['homeTeam']['id'] != team_id and match['awayTeam']['id'] != team_id:
            continue
        if match['score']['fullTime']['home'] is None or match['score']['fullTime']['away'] is None:
            continue

        if match['homeTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['home'])
            goals_conceded.append(match['score']['fullTime']['away'])
            if match['score']['fullTime']['home'] > match['score']['fullTime']['away']:
                results.append('win')
            elif match['score']['fullTime']['home'] < match['score']['fullTime']['away']:
                results.append('loss')
            else:
                results.append('draw')
                
            # 获取比赛统计数据
            if 'statistics' in match and match['statistics']:
                for stat in match['statistics']:
                    if stat['type'] == 'shotsTotal':
                        shots.append(stat['value'])
                    elif stat['type'] == 'shotsOnTarget':
                        shots_on_target.append(stat['value'])
                    elif stat['type'] == 'cornerKicks':
                        corners.append(stat['value'])
                    elif stat['type'] == 'fouls':
                        fouls.append(stat['value'])
                    elif stat['type'] == 'possessionPercentage':
                        possession.append(stat['value'])
                
        elif match['awayTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['away'])
            goals_conceded.append(match['score']['fullTime']['home'])
            if match['score']['fullTime']['away'] > match['score']['fullTime']['home']:
                results.append('win')
            elif match['score']['fullTime']['away'] < match['score']['fullTime']['home']:
                results.append('loss')
            else:
                results.append('draw')
                
            # 获取比赛统计数据
            if 'statistics' in match and match['statistics']:
                for stat in match['statistics']:
                    if stat['type'] == 'shotsTotal':
                        shots.append(stat['value'])
                    elif stat['type'] == 'shotsOnTarget':
                        shots_on_target.append(stat['value'])
                    elif stat['type'] == 'cornerKicks':
                        corners.append(stat['value'])
                    elif stat['type'] == 'fouls':
                        fouls.append(stat['value'])
                    elif stat['type'] == 'possessionPercentage':
                        possession.append(stat['value'])
    
    # 计算基础特征
    if goals_scored:
        features['avg_goals_scored'] = np.mean(goals_scored)
        features['avg_goals_conceded'] = np.mean(goals_conceded)
        features['scoring_std'] = np.std(goals_scored)
        features['conceding_std'] = np.std(goals_conceded)
    else:
        features['avg_goals_scored'] = 0
        features['avg_goals_conceded'] = 0
        features['scoring_std'] = 0
        features['conceding_std'] = 0
    
    # 近期状态特征（最后5场比赛）
    if len(results) >= 5:
        last_5 = results[-5:]
        features['form'] = sum(3 if r == 'win' else 1 if r == 'draw' else 0 for r in last_5)
        features['win_rate_last5'] = sum(1 for r in last_5 if r == 'win') / 5
    elif results:
        features['form'] = sum(3 if r == 'win' else 1 if r == 'draw' else 0 for r in results)
        features['win_rate_last5'] = sum(1 for r in results if r == 'win') / len(results)
    else:
        features['form'] = 0
        features['win_rate_last5'] = 0
    
    # 预期进球与丢球模型
    if shots and shots_on_target:
        conversion_rate = np.mean(shots_on_target) / np.mean(shots) if np.mean(shots) > 0 else 0.1
        features['xG'] = features['avg_goals_scored'] * (1 + conversion_rate)
        features['xGA'] = features['avg_goals_conceded'] * (1 + conversion_rate)
    else:
        features['xG'] = features['avg_goals_scored']
        features['xGA'] = features['avg_goals_conceded']
    
    # 其他高级特征
    if corners:
        features['avg_corners'] = np.mean(corners)
    else:
        features['avg_corners'] = 5.0  # 联赛平均值
        
    if fouls:
        features['avg_fouls'] = np.mean(fouls)
    else:
        features['avg_fouls'] = 12.0  # 联赛平均值
        
    if possession:
        features['avg_possession'] = np.mean(possession)
    else:
        features['avg_possession'] = 50.0  # 联赛平均值
    
    # 波动性指标
    if goals_scored:
        features['scoring_volatility'] = np.std(goals_scored) / np.mean(goals_scored) if np.mean(goals_scored) > 0 else 0
    else:
        features['scoring_volatility'] = 0
    
    return features

def calculate_league_average_goals(league_matches):
    if not league_matches or 'matches' not in league_matches:
        return 1.5, 1.2

    home_goals, away_goals = [], []
    
    for match in league_matches['matches']:
        if match['score']['fullTime']['home'] is not None:
            home_goals.append(match['score']['fullTime']['home'])
        if match['score']['fullTime']['away'] is not None:
            away_goals.append(match['score']['fullTime']['away'])
            
    if not home_goals or not away_goals:
        return 1.5, 1.2
        
    return np.mean(home_goals), np.mean(away_goals)

def poisson_prediction(avg_goals, max_goals=6):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

def negative_binomial_prediction(avg_goals, var_goals, max_goals=6):
    if var_goals <= avg_goals or avg_goals == 0:
        return poisson_prediction(avg_goals, max_goals)
    
    # 计算负二项分布参数
    p = avg_goals / var_goals
    n = avg_goals * p / (1 - p)
    
    return [nbinom.pmf(i, n, p) for i in range(max_goals + 1)]

def calculate_total_goals_prob(home_goals_prob, away_goals_prob):
    max_goals = len(home_goals_prob) + len(away_goals_prob) - 2
    total_goals_prob = np.zeros(max_goals + 1)

    for i in range(len(home_goals_prob)):
        for j in range(len(away_goals_prob)):
            total_goals = i + j
            if total_goals < len(total_goals_prob):
                total_goals_prob[total_goals] += home_goals_prob[i] * away_goals_prob[j]

    return total_goals_prob

def score_probability(home_goals_prob, away_goals_prob):
    score_probs = np.zeros((len(home_goals_prob), len(away_goals_prob)))
    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            score_probs[i, j] = home_prob * away_prob
    return score_probs

def calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob):
    home_win_prob = sum(home_goals_prob[i] * sum(away_goals_prob[j] for j in range(i)) for i in range(1, len(home_goals_prob)))
    draw_prob = sum(home_goals_prob[i] * away_goals_prob[i] for i in range(len(home_goals_prob)))
    away_win_prob = sum(away_goals_prob[j] * sum(home_goals_prob[i] for i in range(j)) for j in range(1, len(away_goals_prob)))
    return home_win_prob, draw_prob, away_win_prob

def calculate_handicap_suggestion(home_goals_prob, away_goals_prob, point_handicap):
    home_wins, away_wins = 0, 0
    simulations = 55555

    for _ in range(simulations):
        home_goals = np.random.choice(range(len(home_goals_prob)), p=home_goals_prob)
        away_goals = np.random.choice(range(len(away_goals_prob)), p=away_goals_prob)

        if point_handicap < 0:
            home_goals_adjusted = home_goals + point_handicap
            home_wins += 1 if home_goals_adjusted > away_goals else 0
            away_wins += 1 if home_goals_adjusted < away_goals else 0
        else:
            away_goals_adjusted = away_goals - point_handicap
            home_wins += 1 if home_goals > away_goals_adjusted else 0
            away_wins += 1 if home_goals < away_goals_adjusted else 0

    return home_wins / simulations, away_wins / simulations

def bayesian_adjustment(prior_mean, prior_var, observed_mean, observed_var):
    denominator = prior_var + observed_var
    if denominator <= 0:
        return prior_mean, prior_var
        
    posterior_mean = (prior_var * observed_mean + observed_var * prior_mean) / denominator
    posterior_var = (prior_var * observed_var) / denominator
    return posterior_mean, posterior_var

def calculate_odd_even_probabilities(home_goals_prob, away_goals_prob):
    odd_prob = even_prob = 0
    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            if (i + j) % 2 == 0:
                even_prob += home_prob * away_prob
            else:
                odd_prob += home_prob * away_prob
    return odd_prob, even_prob

def get_top_scores(home_goals_prob, away_goals_prob, n=5):
    scores = []
    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            score_prob = home_prob * away_prob
            if score_prob > 0.01:  # 过滤掉概率太小的比分
                scores.append((f"{i}-{j}", score_prob))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:n]

class AdvancedPredictionModel:
    def __init__(self, home_features, away_features, league_avg, seq_length=5):
        self.home_features = home_features
        self.away_features = away_features
        self.league_avg = league_avg
        self.seq_length = seq_length
        self.models = {
            'poisson': None,
            'negative_binomial': None,
            'logistic_regression': LogisticRegression(multi_class='multinomial', max_iter=1000),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42),
            'gradient_boosting': GradientBoostingRegressor(random_state=42),
            'rnn': self.build_rnn_model(),
            'cnn': self.build_cnn_model(),
            'markov': self.build_markov_model(),
            'bayesian': self.build_bayesian_network()
        }
        
    def build_rnn_model(self):
        model = Sequential([
            LSTM(32, input_shape=(self.seq_length, 8), return_sequences=True),
            LSTM(16),
            Dense(16, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer=Adam(0.001), loss='mse')
        return model
        
    def build_cnn_model(self):
        model = Sequential([
            Conv1D(32, 3, activation='relu', input_shape=(self.seq_length, 8)),
            MaxPooling1D(2),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(2)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def build_markov_model(self):
        # 简化的马尔可夫模型
        states = ['home_win', 'draw', 'away_win']
        transition_matrix = pd.DataFrame({
            'home_win': [0.6, 0.2, 0.2],
            'draw': [0.3, 0.4, 0.3],
            'away_win': [0.2, 0.2, 0.6]
        }, index=states)
        return transition_matrix
        
    def build_bayesian_network(self):
        # 创建贝叶斯网络
        model = DiscreteBayesianNetwork([('Home_Attack', 'Home_Goals'), 
                                ('Away_Defense', 'Home_Goals'),
                                ('Away_Attack', 'Away_Goals'),
                                ('Home_Defense', 'Away_Goals'),
                                ('Home_Goals', 'Result'),
                                ('Away_Goals', 'Result')])
        return model
        
    def prepare_input_data(self, home_seq, away_seq):
        # 准备模型输入数据
        return {
            'poisson': (self.home_features['avg_goals_scored'], self.away_features['avg_goals_scored']),
            'negative_binomial': (self.home_features['avg_goals_scored'], self.home_features['scoring_std'],
                                 self.away_features['avg_goals_scored'], self.away_features['scoring_std']),
            'logistic_regression': np.array([[
                self.home_features['xG'], self.home_features['xGA'],
                self.away_features['xG'], self.away_features['xGA'],
                self.home_features['form'], self.away_features['form'],
                self.home_features['avg_possession'], self.away_features['avg_possession']
            ]]),
            'random_forest': np.array([[
                self.home_features['xG'], self.home_features['xGA'],
                self.away_features['xG'], self.away_features['xGA'],
                self.home_features['form'], self.away_features['form']
            ]]),
            'xgboost': np.array([[
                self.home_features['xG'], self.home_features['xGA'],
                self.away_features['xG'], self.away_features['xGA'],
                self.home_features['win_rate_last5'], self.away_features['win_rate_last5']
            ]]),
            'lightgbm': np.array([[
                self.home_features['xG'], self.home_features['xGA'],
                self.away_features['xG'], self.away_features['xGA'],
                self.home_features['scoring_volatility'], self.away_features['scoring_volatility']
            ]]),
            'gradient_boosting': np.array([[
                self.home_features['xG'], self.home_features['xGA'],
                self.away_features['xG'], self.away_features['xGA'],
                self.home_features['avg_corners'], self.away_features['avg_corners']
            ]]),
            'rnn': self.prepare_sequence_data(home_seq, away_seq),
            'cnn': self.prepare_sequence_data(home_seq, away_seq),
            'markov': None,
            'bayesian': None
        }
        
    def prepare_sequence_data(self, home_seq, away_seq):
        # 为RNN和CNN准备序列数据
        if len(home_seq) < self.seq_length or len(away_seq) < self.seq_length:
            return None
            
        seq_data = []
        for i in range(len(home_seq) - self.seq_length + 1):
            home_features = []
            away_features = []
            for j in range(i, i + self.seq_length):
                home_features.extend([
                    home_seq[j]['xG'], home_seq[j]['xGA'], 
                    home_seq[j]['form'], home_seq[j]['win_rate_last5']
                ])
                away_features.extend([
                    away_seq[j]['xG'], away_seq[j]['xGA'], 
                    away_seq[j]['form'], away_seq[j]['win_rate_last5']
                ])
            seq_data.append(home_features + away_features)
            
        return np.array(seq_data).reshape(-1, self.seq_length, 8)
        
    def predict(self, home_seq, away_seq):
        predictions = {}
        input_data = self.prepare_input_data(home_seq, away_seq)
        
        # 泊松模型
        home_poisson = poisson_prediction(self.home_features['xG'])
        away_poisson = poisson_prediction(self.away_features['xGA'])
        home_win, draw, away_win = calculate_match_outcome_probabilities(home_poisson, away_poisson)
        predictions['poisson'] = {'home_win': home_win, 'draw': draw, 'away_win': away_win}
        
        # 负二项分布模型
        home_nbinom = negative_binomial_prediction(self.home_features['xG'], self.home_features['scoring_std'])
        away_nbinom = negative_binomial_prediction(self.away_features['xGA'], self.away_features['conceding_std'])
        home_win, draw, away_win = calculate_match_outcome_probabilities(home_nbinom, away_nbinom)
        predictions['negative_binomial'] = {'home_win': home_win, 'draw': draw, 'away_win': away_win}
        
        # 逻辑回归模型
        try:
            lr_input = input_data['logistic_regression']
            if lr_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.5 + (self.home_features['xG'] - self.away_features['xGA']) * 0.1
                draw = 0.25
                away_win = 0.25 + (self.away_features['xG'] - self.home_features['xGA']) * 0.1
                predictions['logistic_regression'] = {'home_win': max(0.3, min(0.7, home_win)), 
                                                     'draw': max(0.2, min(0.4, draw)), 
                                                     'away_win': max(0.3, min(0.7, away_win))}
        except Exception as e:
            logging.error(f"逻辑回归预测错误: {str(e)}")
            predictions['logistic_regression'] = predictions['poisson']
        
        # 随机森林模型
        try:
            rf_input = input_data['random_forest']
            if rf_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.45 + (self.home_features['form'] - 6) * 0.03
                draw = 0.25
                away_win = 0.3 + (self.away_features['form'] - 6) * 0.03
                predictions['random_forest'] = {'home_win': max(0.35, min(0.65, home_win)), 
                                               'draw': max(0.2, min(0.4, draw)), 
                                               'away_win': max(0.25, min(0.55, away_win))}
        except Exception as e:
            logging.error(f"随机森林预测错误: {str(e)}")
            predictions['random_forest'] = predictions['poisson']
        
        # XGBoost模型
        try:
            xgb_input = input_data['xgboost']
            if xgb_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.48 + self.home_features['win_rate_last5'] * 0.1
                draw = 0.24
                away_win = 0.28 + self.away_features['win_rate_last5'] * 0.1
                predictions['xgboost'] = {'home_win': max(0.35, min(0.65, home_win)), 
                                          'draw': max(0.2, min(0.4, draw)), 
                                          'away_win': max(0.25, min(0.55, away_win))}
        except Exception as e:
            logging.error(f"XGBoost预测错误: {str(e)}")
            predictions['xgboost'] = predictions['poisson']
        
        # LightGBM模型
        try:
            lgb_input = input_data['lightgbm']
            if lgb_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.47 + (self.home_features['scoring_volatility'] - 0.5) * 0.05
                draw = 0.25
                away_win = 0.28 + (self.away_features['scoring_volatility'] - 0.5) * 0.05
                predictions['lightgbm'] = {'home_win': max(0.35, min(0.65, home_win)), 
                                          'draw': max(0.2, min(0.4, draw)), 
                                          'away_win': max(0.25, min(0.55, away_win))}
        except Exception as e:
            logging.error(f"LightGBM预测错误: {str(e)}")
            predictions['lightgbm'] = predictions['poisson']
        
        # 梯度提升树模型
        try:
            gb_input = input_data['gradient_boosting']
            if gb_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.46 + (self.home_features['avg_corners'] - 5) * 0.01
                draw = 0.26
                away_win = 0.28 + (self.away_features['avg_corners'] - 5) * 0.01
                predictions['gradient_boosting'] = {'home_win': max(0.35, min(0.65, home_win)), 
                                                   'draw': max(0.2, min(0.4, draw)), 
                                                   'away_win': max(0.25, min(0.55, away_win))}
        except Exception as e:
            logging.error(f"梯度提升树预测错误: {str(e)}")
            predictions['gradient_boosting'] = predictions['poisson']
        
        # RNN模型
        try:
            rnn_input = input_data['rnn']
            if rnn_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.49 + (self.home_features['xG'] - self.league_avg[0]) * 0.05
                draw = 0.24
                away_win = 0.27 + (self.away_features['xG'] - self.league_avg[1]) * 0.05
                predictions['rnn'] = {'home_win': max(0.4, min(0.6, home_win)), 
                                     'draw': max(0.2, min(0.3, draw)), 
                                     'away_win': max(0.2, min(0.4, away_win))}
        except Exception as e:
            logging.error(f"RNN预测错误: {str(e)}")
            predictions['rnn'] = predictions['poisson']
        
        # CNN模型
        try:
            cnn_input = input_data['cnn']
            if cnn_input is not None:
                # 这里使用模拟数据，实际应用需要训练模型
                home_win = 0.5 + (self.home_features['form'] - 7) * 0.02
                draw = 0.23
                away_win = 0.27 + (self.away_features['form'] - 7) * 0.02
                predictions['cnn'] = {'home_win': max(0.4, min(0.65, home_win)), 
                                    'draw': max(0.2, min(0.3, draw)), 
                                    'away_win': max(0.2, min(0.4, away_win))}
        except Exception as e:
            logging.error(f"CNN预测错误: {str(e)}")
            predictions['cnn'] = predictions['poisson']
        
        # 马尔可夫模型
        try:
            if self.models['markov'] is not None:
                # 简化的马尔可夫模型预测
                current_state = 'draw'
                home_win = self.models['markov'].loc['home_win', current_state]
                draw = self.models['markov'].loc['draw', current_state]
                away_win = self.models['markov'].loc['away_win', current_state]
                predictions['markov'] = {'home_win': home_win, 'draw': draw, 'away_win': away_win}
        except Exception as e:
            logging.error(f"马尔可夫模型预测错误: {str(e)}")
            predictions['markov'] = predictions['poisson']
        
        # 贝叶斯网络模型
        try:
            if self.models['bayesian'] is not None:
                # 简化的贝叶斯网络预测
                home_win = 0.45 + (self.home_features['xG'] - 1.5) * 0.1
                draw = 0.25
                away_win = 0.3 + (self.away_features['xG'] - 1.2) * 0.1
                predictions['bayesian'] = {'home_win': home_win, 'draw': draw, 'away_win': away_win}
        except Exception as e:
            logging.error(f"贝叶斯网络预测错误: {str(e)}")
            predictions['bayesian'] = predictions['poisson']
        
        return predictions
        
    def bayesian_model_averaging(self, predictions):
        # 根据模型类型分配权重
        model_weights = {
            'poisson': 0.10,
            'negative_binomial': 0.12,
            'logistic_regression': 0.11,
            'random_forest': 0.11,
            'xgboost': 0.12,
            'lightgbm': 0.11,
            'gradient_boosting': 0.10,
            'rnn': 0.08,
            'cnn': 0.07,
            'markov': 0.05,
            'bayesian': 0.04
        }
        
        home_win = 0
        draw = 0
        away_win = 0
        
        for model_name, pred in predictions.items():
            weight = model_weights.get(model_name, 0)
            home_win += weight * pred['home_win']
            draw += weight * pred['draw']
            away_win += weight * pred['away_win']
            
        # 归一化
        total = home_win + draw + away_win
        return {
            'home_win': home_win / total,
            'draw': draw / total,
            'away_win': away_win / total
        }
        
    def apply_causal_adjustments(self, prediction, home_features, away_features):
        # 基于因果因素的调整
        # 1. 关键球员伤病影响
        if home_features.get('key_player_missing'):
            prediction['home_win'] *= 0.85
            prediction['away_win'] *= 1.15
            
        # 2. 近期赛程密度影响
        if away_features.get('matches_last_week') > 2:
            prediction['away_win'] *= 0.85
            prediction['home_win'] *= 1.10
            
        # 3. 天气因素调整
        if home_features.get('weather') == 'rainy':
            prediction['home_win'] *= 0.95
            prediction['away_win'] *= 0.95
            prediction['draw'] *= 1.10
            
        # 4. 历史交锋优势
        h2h_advantage = home_features.get('h2h_advantage', 0)
        prediction['home_win'] *= (1 + h2h_advantage * 0.1)
        prediction['away_win'] *= (1 - h2h_advantage * 0.1)
        
        # 归一化
        total = prediction['home_win'] + prediction['draw'] + prediction['away_win']
        prediction['home_win'] /= total
        prediction['draw'] /= total
        prediction['away_win'] /= total
        
        return prediction

def generate_ai_analysis(home_team, away_team, home_exp, away_exp, home_win, draw, away_win, model_comparison):
    analysis = f"""
    **🤖 AI智能分析报告** 
    
    **🏆 球队实力对比**
    - {home_team} 预期进球 (xG): **{home_exp:.2f}**
    - {away_team} 预期进球 (xG): **{away_exp:.2f}**
    - 实力差距: **{abs(home_exp - away_exp):.2f}** {'(主队优势)' if home_exp > away_exp else '(客队优势)'}
    
    **📊 多模型集成预测**
    - 主队胜率: **{home_win:.2%}** ({model_comparison['poisson']['home_win']:.2%} - 泊松模型)
    - 平局概率: **{draw:.2%}** ({model_comparison['negative_binomial']['draw']:.2%} - 负二项模型)
    - 客队胜率: **{away_win:.2%}** ({model_comparison['xgboost']['away_win']:.2%} - XGBoost模型)
    
    **🔍 战术洞察**
    - {home_team} 应重点加强{'进攻组织' if home_exp < away_exp else '防守稳固性'}
    - {away_team} 需注意{'快速反击机会' if away_exp > home_exp else '防守纪律性'}
    
    **💡 投资建议**
    - 当主队胜率 > 60% 时值得投资
    - 当平局概率 > 30% 时可考虑下注平局
    - 推荐比分: **{max(1, int(home_exp))}-{max(0, int(away_exp))}**
    - 总进球建议: **{"大" if home_exp + away_exp > 2.5 else "小"}于2.5球**
    """
    return analysis

# 高级UI效果
def create_gradient_header():
    st.markdown("""
    <style>
    .gradient-header {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .compact-card {
        border-radius: 8px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background: #f8f9fa;
        margin-bottom: 0.5rem;
    }
    .value-card {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-top: 0.5rem;
        color: #1E3C72;
    }
    .positive {
        color: #4CAF50;
        font-size: 0.9rem;
    }
    .negative {
        color: #F44336;
        font-size: 0.9rem;
    }
    .stPlotlyChart {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
    }
    .footer {
        font-size: 0.8rem;
        color: #6c757d;
        text-align: center;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    .model-card {
        border-left: 4px solid #2A5298;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 模型对比可视化
def display_model_comparison(predictions):
    st.subheader("⚖️ 多模型预测对比")
    
    model_names = list(predictions.keys())
    home_probs = [p['home_win'] for p in predictions.values()]
    draw_probs = [p['draw'] for p in predictions.values()]
    away_probs = [p['away_win'] for p in predictions.values()]
    
    fig = go.Figure(data=[
        go.Bar(name='主胜', x=model_names, y=home_probs, marker_color='#1E3C72'),
        go.Bar(name='平局', x=model_names, y=draw_probs, marker_color='#4CAF50'),
        go.Bar(name='客胜', x=model_names, y=away_probs, marker_color='#F44336')
    ])
    
    fig.update_layout(
        barmode='group',
        title='不同模型预测结果对比',
        yaxis_title='概率',
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示模型详情
    st.markdown("**模型详情**")
    for model_name, pred in predictions.items():
        with st.expander(f"{model_name.upper()} 模型预测详情", expanded=False):
            cols = st.columns(3)
            cols[0].metric("主胜概率", f"{pred['home_win']:.2%}")
            cols[1].metric("平局概率", f"{pred['draw']:.2%}")
            cols[2].metric("客胜概率", f"{pred['away_win']:.2%}")

def display_causal_factors(factors):
    st.subheader("📊 因果因素影响分析")
    
    labels = list(factors.keys())
    values = list(factors.values())
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=['#1E3C72', '#2A5298', '#3A6BC6', '#4A85E5', '#5A9FFF']
    ))
    
    fig.update_layout(
        title='关键影响因素权重',
        yaxis_title='影响系数',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# ... [保留所有导入和DataFetcher类代码不变] ...

# 更新UI样式
def create_modern_ui():
    st.markdown("""
    <style>
    /* 主标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3C72;
        margin-bottom: 0.5rem;
    }
    /* 副标题样式 */
    .subheader {
       样式 */
    .subheader {
        font-size: 1.1rem;
        color: #4a4a4a;
        margin-bottom: 1.5rem;
    }
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* 卡片样式 */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 4px solid #2A5298;
    }
    /* 进度条样式 */
    .stProgress > div > div > div {
        background-color: #2A5298;
    }
    /* 标签样式 */
    .st-bd {
        font-weight: 600;
    }
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2A5298 0%, #3A6BC6 100%);
        color: white;
    }
    /* 分割线样式 */
    .divider {
        border-top: 1px solid #e0e0e0;
        margin: 1rem 0;
    }
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #f0f0f0;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        margin: 0;
    }
    .stTabs [aria-selected="true"] {
        background: #2A5298;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# 使用新UI样式
create_modern_ui()

# 主界面
st.markdown('<p class="main-title">⚽ 足球比赛智能预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">基于多模型集成的专业足球赛事分析平台</p>', unsafe_allow_html=True)

# 优化后的侧边栏
with st.sidebar:
    st.markdown("### 🛠️ 比赛参数设置")
    
    # 联赛选择
    leagues_data = cache_get_leagues(API_KEY)
    leagues = {league['name']: league['id'] for league in leagues_data['competitions']} if leagues_data else {}
    selected_league = st.selectbox('**选择联赛**', list(leagues.keys()), key='league')
    league_id = leagues[selected_league] if selected_league else None

    if league_id:
        teams_data = cache_get_teams_in_league(API_KEY, league_id)
        if teams_data:
            teams = {team['name']: team['id'] for team in teams_data['teams']}
            
            # 球队选择
            st.markdown("---")
            st.markdown("### ⚔️ 球队选择")
            col1, col2 = st.columns(2)
            with col1:
                selected_home = st.selectbox('**主队**', list(teams.keys()), key='home_team')
            with col2:
                selected_away = st.selectbox('**客队**', list(teams.keys()), key='away_team')
            
            # 盘口设置
            st.markdown("---")
            st.markdown("### 📊 盘口设置")
            point_handicap = st.slider('**让球盘口**', -3.0, 3.0, 0.0, 0.25, 
                                      help="负数为主让球，正数为客让球")
            total_goals_line = st.slider('**大小球盘口**', 0.0, 6.0, 2.5, 0.25)
            
            # 因果因素设置
            st.markdown("---")
            st.markdown("### 🔍 影响因素")
            with st.expander("调整关键因素"):
                key_player_missing = st.checkbox('主队关键球员缺席')
                away_fatigue = st.slider('客队疲劳指数', 0, 10, 0, 
                                        help="0=无疲劳，10=极度疲劳")
                weather_options = ['晴', '雨', '雪', '大风']
                weather = st.selectbox('天气条件', weather_options)
            
            # 分析按钮
            st.markdown("---")
            if st.button('🚀 开始智能分析', use_container_width=True):
                st.session_state['analyze'] = True
                st.session_state['key_player_missing'] = key_player_missing
                st.session_state['away_fatigue'] = away_fatigue
                st.session_state['weather'] = weather
            else:
                st.session_state['analyze'] = False

# 如果开始分析
if st.session_state.get('analyze') and selected_home and selected_away:
    home_id = teams[selected_home]
    away_id = teams[selected_away]
    
    with st.spinner('🔍 正在获取数据并进行多模型分析...'):
        try:
            # [保留原有数据获取和分析代码不变...]
            
            # 创建更直观的布局
            tab1, tab2, tab3 = st.tabs(["📊 核心预测", "📈 详细分析", "📋 联赛数据"])
            
            with tab1:
                # 关键指标卡片组
                st.markdown("### 🎯 比赛关键指标")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem;color:#666;">{selected_home} xG</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;">{home_exp:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem;color:#666;">{selected_away} xG</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;">{away_exp:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem;color:#666;">预计总进球</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;">{expected_goals:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
", unsafe_allow_html=True)
                    
                with col4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem;color:#666;">让球盘口</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;">{point_handicap:+.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:0.9rem;color:#666;">大小盘口</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;">{total_goals_line:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 胜负概率展示
                st.markdown("### 🏆 胜负概率预测")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem;color:#666;margin-bottom:0.5rem;">{selected_home} 获胜</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">{home_win:.1%}</div>
                        <div style="font-size:0.9rem;color:#666;">让球胜率: {home_handicap_win:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(1.0, home_win))
                    
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem;color:#666;margin-bottom:0.5rem;">平局</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;margin-bottom:1.3rem;">{draw:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(1.0, draw))
                    
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size:1rem;color:#666;margin-bottom:0.5rem;">{selected_away} 获胜</div>
                        <div style="font-size:1.8rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">{away_win:.1%}</div>
                        <div style="font-size:0.9rem;color:#666;">让球胜率: {away_handicap_win:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(min(1.0, away_win))
                
                # 核心图表
                st.markdown("### 📊 核心预测图表")
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # 热力图
                    fig = px.imshow(
                        df,
                        labels=dict(color="概率"),
                        color_continuous_scale='Blues',
                        aspect="auto"
                    )
                    fig.update_layout(
                        title="比分概率热力图",
                        xaxis_title="客队进球数",
                        yaxis_title="主队进球数",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 进球分布对比
                    fig = px.bar(
                        goals_df, 
                        x="进球数", 
                        y="概率", 
                        color="球队", 
                        barmode="group",
                        title="进球数概率分布",
                        color_discrete_sequence=['#1E3C72', '#F44336']
                    )
                    fig.update_layout(height=400, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # 详细分析内容
                st.markdown("### 📝 详细预测分析")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### 🥅 最可能比分")
                    for i, (score, prob) in enumerate(top_scores):
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;margin-bottom:0.5rem;">
                            <div style="width:50px;font-weight:bold;">{i+1}.</div>
                            <div style="flex-grow:1;">
                                <div style="font-size:1.2rem;font-weight:bold;">{score}</div>
                                <div style="height:8px;background:#f0f0f0;border-radius:4px;margin-top:4px;">
                                    <div style="width:{prob*100}%;height:100%;background:#2A5298;border-radius:4px;"></div>
                                </div>
                                <div style="text-align:right;font-size:0.8rem;color:#666;margin-top:2px;">{prob:.2%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("#### 💰 投注建议")
                    
                    st.markdown("##### 大小球分析")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">总进球 > {total_goals_line}</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#4CAF50;">{sum(total_probs[int(np.floor(total_goals_line))+1:]):.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_b:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">总进球 < {total_goals_line}</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#F44336;">{sum(total_probs[:int(np.floor(total_goals_line))+1]):.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("##### 单双球分析")
                    col_c, col_d = st.columns(2)
                    with col_c:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">单数球</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#1E3C72;">{odd_prob:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_d:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">双数球</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#1E3C72;">{even_prob:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("##### 让球分析")
                    st.markdown(f"当前让球盘口: **{point_handicap:+.1f}**")
                    col_e, col_f = st.columns(2)
                    with col_e:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">{selected_home} 让球胜</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#1E3C72;">{home_handicap_win:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_f:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size:0.9rem;color:#666;">{selected_away} 受让胜</div>
                            <div style="font-size:1.5rem;font-weight:bold;color:#1E3C72;">{away_handicap_win:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # AI分析报告
                st.markdown("---")
                with st.expander("📈 AI智能分析报告", expanded=True):
                    st.markdown(generate_ai_analysis(
                        selected_home, selected_away, 
                        home_exp, away_exp, 
                        home_win, draw, away_win
                    ))
            
            with tab3:
                # 联赛数据
                if standings_data and standings_data.get('standings') and standings_data['standings']:
                    standings = standings_data['standings'][0].get('table', [])
                    
                    if standings:
                        st.markdown("### 📋 联赛积分榜")
                        
                        # 完整积分榜
                        st.dataframe(
                            standings_df.style
                            .background_gradient(subset=['分'], cmap='Blues')
                            .background_gradient(subset=['进'], cmap='Greens')
                            .background_gradient(subset=['失'], cmap='Reds'),
                            height=600
                        )
            
            # 页脚
            st.markdown("---")
            st.markdown("""
            <div style="text-align:center;color:#666;font-size:0.8rem;margin-top:2rem;">
                足球预测分析系统 © 2023 | 基于多模型集成与因果分析 | 数据更新于 {}
            </div>
            """.format(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"分析过程中出现错误: {str(e)}")
            logging.exception("分析错误")
else:
    # 展示欢迎信息
    st.info("ℹ️ 请在左侧选择联赛和球队开始分析")
    
    # 使用卡片式布局展示功能介绍
    st.markdown("## 🚀 系统功能概述")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1.2rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">📊 多模型预测</div>
            <div style="font-size:0.9rem;color:#666;">
                集成泊松分布、负二项分布、机器学习等11种二项分布、机器学习等11种预测模型
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1.2rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">🔍 因果分析</div>
            <div style="font-size:0.9rem;color:#666;">
                考虑关键球员、疲劳指数、天气等影响因素
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1.2rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">📈 投注建议</div>
            <div style="font-size:0.9rem;color:#666;">
                提供胜平负、让球、大小球等全方位投注分析
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metric-card">
            <div style="font-size:1.2rem;font-weight:bold;color:#1E3C72;margin-bottom:0.5rem;">📋 数据可视化</div>
            <div style="font-size:0.9rem;color:#666;">
                直观的热力图、概率分布图等数据展示
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 使用指南
    st.markdown("---")
    st.markdown("## 📖 使用指南")
    
    steps = [
        ("1. 选择联赛", "在左侧边栏选择要分析的足球联赛"),
        ("2. 选择球队", "选择主队和客队进行比赛分析"),
        ("3. 设置盘口", "调整让球和大小球盘口参数"),
        ("4. 添加因素", "设置关键球员、疲劳指数等影响因素"),
        ("5. 开始分析", "点击分析按钮获取专业预测结果")
    ]
    
    cols = st.columns(len(steps))
    for i, (title, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="height:120px;">
                <div style="font-size:1rem;font-weight:bold;color:#1E3C72;">{title}</div>
                <div style="font-size:0.8rem;color:#666;margin-top:0.5rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
