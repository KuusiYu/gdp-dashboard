import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
from scipy.stats import poisson
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 设置中文字体，确保可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 固定 API 密钥
API_KEY = '0c2379b28acb446bb97bd417f2666f81'  # 请替换为你的实际 API 密钥

# 设置日志记录
import logging
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key

    def get_data_with_retries(self, url, headers, params=None, retries=5, delay=2):
        """
        带有重试机制的 API 请求方法
        :param url: API 请求的 URL
        :param headers: 请求头
        :param params: 请求参数（可选）
        :param retries: 重试次数
        :param delay: 每次重试之间的延迟时间（秒）
        :return: API 响应的 JSON 数据
        """
        for attempt in range(retries):
            response = requests.get(url, headers=headers, params=params)  # 添加 params 参数
            logging.info(f"尝试第 {attempt + 1} 次请求，状态码: {response.status_code}, 响应内容: {response.text}")
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

    def get_team_matches(self, team_id, venue=None):
        """
        获取某个球队在特定场地（主场或客场）的所有比赛数据
        """
        url = f'https://api.football-data.org/v4/teams/{team_id}/matches'
        headers = {'X-Auth-Token': self.api_key}
        params = {'venue': venue} if venue else None  # 添加 params 参数
        return self.get_data_with_retries(url, headers, params=params)  # 传递 params 参数

    def get_league_matches(self, league_id):
        """
        获取某个联赛的所有比赛数据
        """
        url = f'https://api.football-data.org/v4/competitions/{league_id}/matches'
        headers = {'X-Auth-Token': self.api_key}
        return self.get_data_with_retries(url, headers)

# 初始化 DataFetcher 实例
fetcher = DataFetcher(API_KEY)

@st.cache_data(ttl=3600)  # 设置缓存有效期为 1 小时
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

def calculate_average_goals_for_team(matches, team_id, venue=None):
    """
    计算某个球队在特定场地（主场或客场）的场均进球数和丢球数
    """
    goals_scored = []
    goals_conceded = []

    for match in matches['matches']:
        if venue and match['homeTeam']['id'] != team_id and match['awayTeam']['id'] != team_id:
            continue

        # 检查比赛结果是否有效
        if match['score']['fullTime']['home'] is None or match['score']['fullTime']['away'] is None:
            continue  # 跳过无效比赛

        if match['homeTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['home'])
            goals_conceded.append(match['score']['fullTime']['away'])
        elif match['awayTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['away'])
            goals_conceded.append(match['score']['fullTime']['home'])

    # 过滤掉 None 值
    goals_scored = [g for g in goals_scored if g is not None]
    goals_conceded = [g for g in goals_conceded if g is not None]

    avg_goals_scored = np.mean(goals_scored) if goals_scored else 0
    avg_goals_conceded = np.mean(goals_conceded) if goals_conceded else 0
    return avg_goals_scored, avg_goals_conceded

def calculate_league_average_goals(league_matches):
    """
    计算整个联盟的主场场均进球数和丢球数
    """
    home_goals = [match['score']['fullTime']['home'] for match in league_matches['matches'] if match['score']['fullTime']['home'] is not None]
    away_goals = [match['score']['fullTime']['away'] for match in league_matches['matches'] if match['score']['fullTime']['away'] is not None]
    avg_home_goals = np.mean(home_goals) if home_goals else 0
    avg_away_goals = np.mean(away_goals) if away_goals else 0
    return avg_home_goals, avg_away_goals

def poisson_prediction(avg_goals, max_goals=10):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

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
    home_win_prob = sum(home_goals_prob[i] * sum(away_goals_prob[j] for j in range(i)) for i in range(1, len(home_goals_prob)))
    draw_prob = sum(home_goals_prob[i] * away_goals_prob[i] for i in range(len(home_goals_prob)))
    away_win_prob = sum(away_goals_prob[j] * sum(home_goals_prob[i] for i in range(j)) for j in range(1, len(away_goals_prob)))
    return home_win_prob, draw_prob, away_win_prob

def calculate_odds(home_win_prob, draw_prob, away_win_prob):
    home_odds = 1 / home_win_prob if home_win_prob > 0 else float('inf')
    draw_odds = 1 / draw_prob if draw_prob > 0 else float('inf')
    away_odds = 1 / away_win_prob if away_win_prob > 0 else float('inf')
    return home_odds, draw_odds, away_odds

def generate_ai_analysis(home_team_name, away_team_name, predicted_home_goals, predicted_away_goals, home_win_prob, draw_prob, away_win_prob):
    analysis = f"""
    **AI 分析报告**

    根据模型预测，{home_team_name} 的预期进球数为 {predicted_home_goals:.2f}，而 {away_team_name} 的预期进球数为 {predicted_away_goals:.2f}。

    - **主队胜率**: {home_win_prob:.2%}
    - **平局概率**: {draw_prob:.2%}
    - **客队胜率**: {away_win_prob:.2%}

    综合来看，如果主队的进球数高于客队，主队更有可能获胜；反之，客队更有可能获胜。如果两队进球数接近，则比赛更有可能以平局结束。
    """
    return analysis

st.title('足球比赛预测器')

# 设置侧边栏参数
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
        point_handicap = st.sidebar.number_input('输入受让/让球盘口，主让球为负，客让球为正', min_value=-5.0, max_value=5.0, value=0.0)
        total_goals_line = st.sidebar.number_input('输入大小球盘口', min_value=0.0, max_value=10.0, value=2.5)

        if confirm_button:
            with st.spinner("请耐心等待30秒，程序正在运行。"):
                home_team_id = teams[selected_home_team_name]
                away_team_id = teams[selected_away_team_name]

                # 获取主队和客队的比赛数据
                home_team_home_matches = cache_get_team_matches(API_KEY, home_team_id, venue='HOME')
                away_team_away_matches = cache_get_team_matches(API_KEY, away_team_id, venue='AWAY')

                # 计算主队和客队的场均进球数和丢球数
                home_avg_goals_scored, home_avg_goals_conceded = calculate_average_goals_for_team(home_team_home_matches, home_team_id, venue='HOME')
                away_avg_goals_scored, away_avg_goals_conceded = calculate_average_goals_for_team(away_team_away_matches, away_team_id, venue='AWAY')

                # 获取整个联盟的比赛数据
                league_matches = cache_get_league_matches(API_KEY, league_id)
                league_avg_home_goals, league_avg_away_goals = calculate_league_average_goals(league_matches)

                # 计算主队和客队的预期进球数
                home_expected_goals = home_avg_goals_scored * away_avg_goals_conceded / league_avg_away_goals
                away_expected_goals = away_avg_goals_scored * home_avg_goals_conceded / league_avg_home_goals

                st.markdown(f"<h3 style='color: green;'>预测主队进球数: {home_expected_goals:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: purple;'>预测客队进球数: {away_expected_goals:.2f}</h3>", unsafe_allow_html=True)

                home_goals_prob = poisson_prediction(home_expected_goals)
                away_goals_prob = poisson_prediction(away_expected_goals)
                home_goals_prob = np.array(home_goals_prob)
                away_goals_prob = np.array(away_goals_prob)
                home_goals_prob /= home_goals_prob.sum()
                away_goals_prob /= away_goals_prob.sum()

                # 蒙特卡罗模拟
                def monte_carlo_simulation(home_goals_prob, away_goals_prob, simulations=75330):
                    home_wins = 0
                    draws = 0
                    away_wins = 0

                    for _ in range(simulations):
                        home_goals = np.random.choice(range(len(home_goals_prob)), p=home_goals_prob)
                        away_goals = np.random.choice(range(len(away_goals_prob)), p=away_goals_prob)

                        if home_goals > away_goals:
                            home_wins += 1
                        elif home_goals == away_goals:
                            draws += 1
                        else:
                            away_wins += 1

                    return home_wins / simulations, draws / simulations, away_wins / simulations

                # 执行蒙特卡罗模拟
                home_win_prob, draw_prob, away_win_prob = monte_carlo_simulation(home_goals_prob, away_goals_prob)

                # 计算总进球数概率
                total_goals_prob = calculate_total_goals_prob(home_goals_prob, away_goals_prob)

                # 计算主队胜、平局和客队胜的概率
                home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob)

                # 根据概率提供博彩建议
                total_goals_line_int = int(total_goals_line)
                total_goals_line_ceil = np.ceil(total_goals_line * 4) / 4  # 向上取整以适应0.25, 0.5的盘口阶段

                home_odds, draw_odds, away_odds = calculate_odds(home_win_prob, draw_prob, away_win_prob)

                st.write(f"主队胜的概率: {home_win_prob:.2%}")
                st.write(f"平局的概率: {draw_prob:.2%}")
                st.write(f"客队胜的概率: {away_win_prob:.2%}")

                total_goals_line_int = int(total_goals_line)
                # 检查总进球数概率与盘口的关系
                if np.sum(total_goals_prob[total_goals_line_int:]) > 0.5:
                    st.write(f"建议：投注总进球数大于{total_goals_line}的盘口")
                elif np.sum(total_goals_prob[:total_goals_line_int]) > 0.5:
                    st.write(f"建议：投注总进球数小于{total_goals_line}的盘口")
                else:
                    st.write("建议：根据当前概率，没有明确的投注方向")

                # 比较主客队预测进球数，提供让球建议
                if home_expected_goals + point_handicap > away_expected_goals:
                    if home_win_prob > 0.5:  # 如果主队胜率超过50%，则建议投注主队
                        st.write(f"建议：投注主队让{point_handicap}球胜")
                    else:
                        st.write("建议：考虑其他投注选项，主队胜率不高")
                elif home_expected_goals - point_handicap < away_expected_goals:
                    if away_win_prob > 0.5:  # 如果客队胜率超过50%，则建议投注客队
                        st.write(f"建议：投注客队受{point_handicap}球胜")
                    else:
                        st.write("建议：考虑其他投注选项，客队胜率不高")
                else:
                    st.write("建议：投注受球方")

                # 比分概率热力图
                score_probs = score_probability(home_goals_prob, away_goals_prob)

                # 将 range 对象转换为列表
                x_labels = list(range(len(away_goals_prob)))
                y_labels = list(range(len(home_goals_prob)))

                # 创建 DataFrame
                score_probs_df = pd.DataFrame(score_probs, index=y_labels, columns=x_labels)

                fig = px.imshow(
                    score_probs_df,
                    labels=dict(x="客队进球数", y="主队进球数", color="概率 (%)"),
                    x=x_labels,
                    y=y_labels,
                    color_continuous_scale='Blues'
                )
                
                # 设置x轴和y轴的刻度
                fig.update_xaxes(tickmode='array', tickvals=np.arange(len(x_labels)), ticktext=x_labels)
                fig.update_yaxes(tickmode='array', tickvals=np.arange(len(y_labels)), ticktext=y_labels)

                fig.update_layout(
                    title="比分概率热力图",
                    xaxis_title="客队进球数",
                    yaxis_title="主队进球数"
                )
                st.plotly_chart(fig)

                # 创建数据框
                home_goal_probs_df = pd.DataFrame({
                    'Goals': range(len(home_goals_prob)),
                    'Probability': home_goals_prob,
                })

                away_goal_probs_df = pd.DataFrame({
                    'Goals': range(len(away_goals_prob)),
                    'Probability': away_goals_prob,
                })

                # 确保概率值在有效范围内
                home_goal_probs_df['Probability'] = home_goal_probs_df['Probability'].clip(lower=0, upper=1)
                away_goal_probs_df['Probability'] = away_goal_probs_df['Probability'].clip(lower=0, upper=1)
                max_goals = max(home_goal_probs_df['Goals'].max(), away_goal_probs_df['Goals'].max()) + 1
                max_prob = max(home_goal_probs_df['Probability'].max(), away_goal_probs_df['Probability'].max())

                # 创建对称条形图
                fig = go.Figure()

                # 添加主队条形图（左侧）
                fig.add_trace(go.Bar(
                    y=home_goal_probs_df['Goals'],
                    x=-home_goal_probs_df['Probability'],
                    name= f'{selected_home_team_name} (主队)',
                    marker_color='pink',
                    orientation='h',
                    text=home_goal_probs_df['Probability'],
                    texttemplate='%{text:.2f}',  # 显示概率，保留两位小数
                    textposition='outside'
                ))

                # 添加客队条形图（右侧）
                fig.add_trace(go.Bar(
                    y=away_goal_probs_df['Goals'],
                    x=away_goal_probs_df['Probability'],
                    name= f'{selected_away_team_name} (客队)',
                    marker_color='lightgreen',
                    orientation='h',
                    text=away_goal_probs_df['Probability'],  # 显示概率，去掉负号
                    texttemplate='%{text:.2f}',  # 显示概率，保留两位小数
                    textposition='outside',
                    yaxis='y2'
                ))

                # 更新布局
                fig.update_layout(
                    title='两队进球数概率对比',
                    barmode='group',
                    yaxis=dict(title='进球数', tick0=0, dtick=1),
                    yaxis2=dict(title='进球数', tick0=0, dtick=1, overlaying='y', side='right'),
                    xaxis=dict(title='概率')
                )

                # 在Streamlit中显示图形
                st.plotly_chart(fig)

                total_goals_prob_df = pd.DataFrame({
                    "总进球数": np.arange(len(total_goals_prob)),
                    "概率 (%)": total_goals_prob * 100
                })
                total_goals_prob_df = total_goals_prob_df[total_goals_prob_df["概率 (%)"] > 0]
                
                fig = px.bar(
                    total_goals_prob_df,
                    x="总进球数",
                    y="概率 (%)",
                    title="总进球数概率分布",
                    color="概率 (%)",  # 根据概率值调整颜色
                    labels={"总进球数": "总进球数", "概率 (%)": "概率 (%)"},
                    text_auto=True
                )
               
                # 设置x轴的刻度间隔为1
                fig.update_xaxes(dtick=1)
 
                st.plotly_chart(fig)

                # AI 分析报告
                ai_analysis = generate_ai_analysis(
                    selected_home_team_name,
                    selected_away_team_name,
                    home_expected_goals,
                    away_expected_goals,
                    home_win_prob,
                    draw_prob,
                    away_win_prob
                )
                st.markdown(ai_analysis)

        else:
            st.error("未能加载该联赛的球队，请检查 API。")
    else:
        st.error("没有可用的联赛数据。")
else:
    st.error("无法连接到足球数据 API。")
