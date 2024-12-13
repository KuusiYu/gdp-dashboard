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

    def get_data_with_retries(self, url, headers, retries=5, delay=2):
        for attempt in range(retries):
            response = requests.get(url, headers=headers)
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
                    (match['homeTeam']['id'], match['score']['fullTime']['home'], match['score']['fullTime']['away'])
                    if match['homeTeam']['id'] == team_id else
                    (match['awayTeam']['id'], match['score']['fullTime']['away'], match['score']['fullTime']['home'])
                    for match in matches[:limit]
                    if match['score']['fullTime']['home'] is not None and match['score']['fullTime']['away'] is not None
                ]
                if len(history) < limit:
                    logging.warning(f"获取的历史比赛数据不足，仅获取到 {len(history)} 场比赛。")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                try:
                    joblib.dump(history, cache_file)
                    logging.info(f"Saved team history to cache: {cache_file}")
                except Exception as e:
                    logging.error(f"写入缓存文件失败: {e}")
                return history[:limit]
            else:
                logging.error("无法获取历史比赛数据")
                st.error("无法获取历史比赛数据")
                return []

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
def cache_get_team_history(api_key, team_id, limit=100):
    fetcher = DataFetcher(api_key)
    return fetcher.get_team_history(team_id, limit)

def poisson_prediction(avg_goals, max_goals=10):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

def calculate_weighted_average_goals(history, n=100):
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
    home_win_prob = sum(home_goals_prob[i] * sum(away_goals_prob[j] for j in range(i)) for i in range(1, len(home_goals_prob)))
    draw_prob = sum(home_goals_prob[i] * away_goals_prob[i] for i in range(len(home_goals_prob)))
    away_win_prob = sum(away_goals_prob[j] * sum(home_goals_prob[i] for i in range(j)) for j in range(1, len(away_goals_prob)))
    return home_win_prob, draw_prob, away_win_prob

def calculate_odds(home_win_prob, draw_prob, away_win_prob):
    home_odds = 1 / home_win_prob if home_win_prob > 0 else float('inf')
    draw_odds = 1 / draw_prob if draw_prob > 0 else float('inf')
    away_odds = 1 / away_win_prob if away_win_prob > 0 else float('inf')
    return home_odds, draw_odds, away_odds

def train_models(history):
    goals = [match[1] for match in history]
    X = np.arange(len(goals)).reshape(-1, 1)
    y = np.array(goals)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror'),
        "LightGBM": lgb.LGBMRegressor(n_estimators=100, min_data_in_leaf=5)
    }

    for name, model in models.items():
        model.fit(X, y)
    return models

def predict_goals(models, X):
    predictions = {name: model.predict(X) for name, model in models.items()}
    return np.mean(list(predictions.values()))

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

    # 提取并返回生成的报告
    return response['choices'][0]['message']['content'].strip()
st.title('⚽ 足球比赛预测')

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
        point_handicap = st.sidebar.number_input('输入受让/让球盘口', min_value=-5.0, max_value=5.0, value=0.0)
        total_goals_line = st.sidebar.number_input('输入大小球盘口', min_value=0.0, max_value=10.0, value=2.5)

        if confirm_button:
            with st.spinner("请耐心等待30秒，程序正在运行。"):
                home_team_id = teams[selected_home_team_name]
                away_team_id = teams[selected_away_team_name]

                home_history = cache_get_team_history(API_KEY, home_team_id, limit=6)
                away_history = cache_get_team_history(API_KEY, away_team_id, limit=6)
                avg_home_goals, avg_away_goals = calculate_average_goals(home_history, away_history)

                home_models = train_models(home_history)
                away_models = train_models(away_history)

                predicted_home_goals = predict_goals(home_models, [[0]])
                predicted_away_goals = predict_goals(away_models, [[0]])

                st.header("⚽ 预测结果")
                st.markdown(f"<h3 style='color: green;'>预测主队进球数: {predicted_home_goals:.2f}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: purple;'>预测客队进球数: {predicted_away_goals:.2f}</h3>", unsafe_allow_html=True)

                home_goals_prob = poisson_prediction(predicted_home_goals)
                away_goals_prob = poisson_prediction(predicted_away_goals)
                
                # 计算总进球数概率
                total_goals_prob = calculate_total_goals_prob(home_goals_prob, away_goals_prob)

                # 计算主队胜、平局和客队胜的概率
                home_win_prob, draw_prob, away_win_prob = calculate_match_outcome_probabilities(home_goals_prob, away_goals_prob)

                # 根据概率提供博彩建议
                total_goals_line_int = int(total_goals_line)
                total_goals_line_ceil = np.ceil(total_goals_line)  # 向上取整以适应0.25, 0.5的盘口阶段

                home_odds, draw_odds, away_odds = calculate_odds(home_win_prob, draw_prob, away_win_prob)

                st.header("⚽ 比赛结果概率")
                st.write(f"主队胜的概率: {home_win_prob:.2%}")
                st.write(f"平局的概率: {draw_prob:.2%}")
                st.write(f"客队胜的概率: {away_win_prob:.2%}")

                st.header("📈 博彩建议")
                total_goals_line_int = int(total_goals_line)
                # 检查总进球数概率与盘口的关系
                if np.sum(total_goals_prob[total_goals_line_int:]) > 0.5:
                    st.write(f"建议：投注总进球数大于或等于{total_goals_line}的盘口")
                elif np.sum(total_goals_prob[:total_goals_line_int]) > 0.5:
                    st.write(f"建议：投注总进球数小于{total_goals_line}的盘口")
                else:
                    st.write("建议：根据当前概率，没有明确的投注方向")

                # 比较主客队预测进球数，提供让球建议
                if predicted_home_goals > predicted_away_goals:
                    if home_win_prob > 0.5:  # 如果主队胜率超过50%，则建议投注主队
                        st.write(f"建议：投注主队让{point_handicap}球胜")
                    else:
                        st.write("建议：考虑其他投注选项，主队胜率不高")
                elif predicted_home_goals < predicted_away_goals:
                    if away_win_prob > 0.5:  # 如果客队胜率超过50%，则建议投注客队
                        st.write(f"建议：投注客队受{point_handicap}球胜")
                    else:
                        st.write("建议：考虑其他投注选项，客队胜率不高")
                else:
                    st.write("建议：投注平局")

                # 比分概率热力图
                st.header("📈 比分概率热力图")
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
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(
                    title="比分概率热力图",
                    xaxis_title="客队进球数",
                    yaxis_title="主队进球数"
                )
                st.plotly_chart(fig)

                # 各队进球数概率
                st.header("⚽ 各队进球数概率")

                # 创建数据框
                home_goal_probs_df = pd.DataFrame({
                    'Goals': range(len(home_goals_prob)),
                    'Probability': home_goals_prob,
                })

                away_goal_probs_df = pd.DataFrame({
                    'Goals': range(1, len(away_goals_prob)+1),  # 进球数应为正数
                    'Probability': away_goals_prob,
                })

                # 创建对称条形图
                fig = go.Figure()

                # 添加主队条形图（左侧）
                fig.add_trace(go.Bar(
                    x=-home_goal_probs_df['Goals'],  # 取负值以将条形图放置在左侧
                    y=home_goal_probs_df['Probability'],
                    name=f'{selected_home_team_name} (主队)',
                    marker=dict(
                        color=home_goal_probs_df['Probability'],
                        colorscale='Blues',  # 使用 Blues 颜色渐变
                        cmin=min(home_goal_probs_df['Probability']),  # 设置颜色刻度的最小值
                        cmax=max(home_goal_probs_df['Probability']),  # 设置颜色刻度的最大值
                        showscale=True  # 显示颜色刻度尺
                    ),
                    orientation='v'  # 竖直方向
                ))

                # 添加客队条形图（右侧）
                fig.add_trace(go.Bar(
                    x=away_goal_probs_df['Goals'],  # 正值以将条形图放置在右侧
                    y=away_goal_probs_df['Probability'],
                    name=f'{selected_away_team_name} (客队)',
                    marker=dict(
                        color=away_goal_probs_df['Probability'],
                        colorscale='Cyan',  # 使用 Cyan 颜色渐变
                        cmin=min(away_goal_probs_df['Probability']),  # 设置颜色刻度的最小值
                        cmax=max(away_goal_probs_df['Probability']),  # 设置颜色刻度的最大值
                        showscale=True  # 显示颜色刻度尺
                    ),
                    orientation='v'  # 竖直方向
                ))

                # 更新布局
                fig.update_layout(
                    title=f"{selected_home_team_name} vs {selected_away_team_name} 进球数概率分布",
                    xaxis_title="进球数",
                    yaxis_title="概率",
                    barmode='overlay',  # 条形图重叠
                    legend_title="队伍",
                    legend=dict(orientation="h"),  # 图例水平显示
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(-home_goal_probs_df['Goals']) + list(away_goal_probs_df['Goals']),  # 包含负数和正数
                        ticktext=[f"{selected_home_team_name} {g}" for g in home_goal_probs_df['Goals']] + 
                                 [f"{selected_away_team_name} {g}" for g in away_goal_probs_df['Goals']]
                    )
                )

                # 设置 x 轴范围以确保图形居中
                max_goals = max(home_goal_probs_df['Goals'].max(), away_goal_probs_df['Goals'].max())
                fig.update_xaxes(range=[-max_goals-1, max_goals+1])

                # 在Streamlit中显示图形
                st.plotly_chart(fig)

                # 总进球数概率
                st.header("⚽ 总进球数概率")
                total_goals_prob_df = pd.DataFrame({
                    "总进球数": np.arange(len(total_goals_prob)),
                    "概率 (%)": total_goals_prob * 100
                })
                total_goals_prob_df = total_goals_prob_df[total_goals_prob_df["概率 (%)"] > 0]
                st.write(total_goals_prob_df)
                
                # 总进球数概率柱状图
                st.header("📊 总进球数概率柱状图")
                fig = px.bar(
                    total_goals_prob_df,
                    x="总进球数",
                    y="概率 (%)",
                    title="总进球数概率分布",
                    color="概率 (%)",  # 根据概率值调整颜色
                    labels={"总进球数": "总进球数", "概率 (%)": "概率 (%)"},
                    text_auto=True
                )
                st.plotly_chart(fig)

                # AI 分析报告
                ai_analysis = generate_ai_analysis(
                    selected_home_team_name,
                    selected_away_team_name,
                    predicted_home_goals,
                    predicted_away_goals,
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
