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

    def get_league_standings(self, league_id):
        """
        获取联赛积分榜
        """
        url = f'https://api.football-data.org/v4/competitions/{league_id}/standings'
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

@st.cache_data(ttl=3600)
def cache_get_league_standings(api_key, league_id):
    fetcher = DataFetcher(api_key)
    return fetcher.get_league_standings(league_id)

def calculate_average_goals_for_team(matches, team_id, venue=None):
    """
    计算某个球队在特定场地（主场或客场）的场均进球数和丢球数
    """
    goals_scored = []
    goals_conceded = []
    
    if not matches or 'matches' not in matches:
        st.warning(f"警告：没有找到有效的比赛数据")
        return 0, 0

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
    
    # 如果找不到比赛数据，显示警告
    if not goals_scored:
        st.warning(f"警告：找不到{team_id}在{'主场' if venue == 'HOME' else '客场'}的有效比赛数据")
        return 0, 0

    avg_goals_scored = np.mean(goals_scored) 
    avg_goals_conceded = np.mean(goals_conceded)
    return avg_goals_scored, avg_goals_conceded

def calculate_league_average_goals(league_matches):
    """
    计算整个联盟的主场场均进球数和丢球数
    """
    home_goals = []
    away_goals = []
    
    if not league_matches or 'matches' not in league_matches:
        st.warning("警告：没有找到有效的联赛比赛数据")
        return 1.5, 1.2  # 返回默认平均值

    for match in league_matches['matches']:
        if match['score']['fullTime']['home'] is not None:
            home_goals.append(match['score']['fullTime']['home'])
        if match['score']['fullTime']['away'] is not None:
            away_goals.append(match['score']['fullTime']['away'])
            
    # 如果没有数据，使用默认值
    if not home_goals or not away_goals:
        return 1.5, 1.2  # 返回默认平均值
        
    avg_home_goals = np.mean(home_goals)
    avg_away_goals = np.mean(away_goals)
    return avg_home_goals, avg_away_goals

def poisson_prediction(avg_goals, max_goals=7):
    return [poisson.pmf(i, avg_goals) for i in range(max_goals + 1)]

def calculate_total_goals_prob(home_goals_prob, away_goals_prob):
    max_goals = len(home_goals_prob) + len(away_goals_prob) - 2
    total_goals_prob = np.zeros(max_goals + 1)

    for i in range(len(home_goals_prob)):
        for j in range(len(away_goals_prob)):
            total_goals = i + j
            total_goals_prob[total_goals] += home_goals_prob[i] * away_goals_prob[j]

    return total_goals_prob

# 计算总进球数的期望值和标准差
def calculate_expected_goals_and_std(total_goals_prob):
    expected_goals = sum(i * total_goals_prob[i] for i in range(len(total_goals_prob)))
    variance = sum((i - expected_goals) ** 2 * total_goals_prob[i] for i in range(len(total_goals_prob)))
    std_dev = np.sqrt(variance)
    return expected_goals, std_dev

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

# 计算受让球建议（让球）
def calculate_handicap_suggestion(home_goals_prob, away_goals_prob, point_handicap):
    home_wins = 0
    away_wins = 0
    simulations = 55555  # 增加模拟次数，提高结果的精确度

    for _ in range(simulations):
        home_goals = np.random.choice(range(len(home_goals_prob)), p=home_goals_prob)
        away_goals = np.random.choice(range(len(away_goals_prob)), p=away_goals_prob)

        # 判断盘口的正负，调整比分
        if point_handicap < 0:  # 主队让球
            home_goals_adjusted = home_goals + point_handicap
            if home_goals_adjusted > away_goals:
                home_wins += 1
            elif home_goals_adjusted < away_goals:
                away_wins += 1
        else:  # 客队让球
            away_goals_adjusted = away_goals - point_handicap
            if home_goals > away_goals_adjusted:
                home_wins += 1
            elif home_goals < away_goals_adjusted:
                away_wins += 1

    home_win_prob = home_wins / simulations
    away_win_prob = away_wins / simulations

    return home_win_prob, away_win_prob

def generate_ai_analysis(home_team_name, away_team_name, predicted_home_goals, predicted_away_goals, home_win_prob, draw_prob, away_win_prob):
    analysis = f"""
    **受/让球胜率预测**
    根据模型预测，{home_team_name} 的预期进球数为 {predicted_home_goals:.2f}，而 {away_team_name} 的预期进球数为 {predicted_away_goals:.2f}。
    - **主队胜率**: {home_win_prob:.2%}
    - **平局概率**: {draw_prob:.2%}
    - **客队胜率**: {away_win_prob:.2%}
    
    **投注建议**:
    1. 胜平负策略：如果主队胜率高于55%，可考虑支持主队胜
    2. 比分策略：关注{max(1, int(predicted_home_goals))}-{max(0, int(predicted_away_goals))}等核心比分
    3. 总进球策略：考虑支持总进球数 {'大' if predicted_home_goals + predicted_away_goals > 2.5 else '小'}于2.5球的选项
    """
    return analysis

def bayesian_adjustment(prior_mean, prior_var, observed_mean, observed_var):
    """
    使用贝叶斯方法调整泊松分布的平均值
    """
    denominator = prior_var + observed_var
    if denominator <= 0:  # 防止除0错误
        return prior_mean, prior_var
        
    posterior_mean = (prior_var * observed_mean + observed_var * prior_mean) / denominator
    posterior_var = (prior_var * observed_var) / denominator
    return posterior_mean, posterior_var

def calculate_exact_score_probabilities(home_goals_prob, away_goals_prob):
    """
    计算精准比分概率
    """
    exact_score_probs = {}
    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            score = f"{i}-{j}"
            exact_score_probs[score] = home_prob * away_prob
    return exact_score_probs

def calculate_odd_even_probabilities(home_goals_prob, away_goals_prob):
    """
    计算单双球概率
    """
    odd_prob = 0
    even_prob = 0

    for i, home_prob in enumerate(home_goals_prob):
        for j, away_prob in enumerate(away_goals_prob):
            total_goals = i + j
            if total_goals % 2 == 0:
                even_prob += home_prob * away_prob
            else:
                odd_prob += home_prob * away_prob

    return odd_prob, even_prob

def get_top_scores(exact_score_probs, n=5):
    """获取概率最高的前n个比分"""
    sorted_scores = sorted(exact_score_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:n]

st.title('⚽ 足球比赛预测分析')

# 设置侧边栏参数
st.sidebar.title("比赛参数设置")

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

        # 让用户选择分析维度
        analysis_option = st.sidebar.selectbox('分析维度', ['基础预测', '进阶分析', '投注策略'])

        point_handicap = st.sidebar.number_input('让球盘口（负数为主让，正数为客让）', min_value=-5.0, max_value=5.0, value=0.0)
        total_goals_line = st.sidebar.number_input('大小球盘口', min_value=0.0, max_value=10.0, value=2.5)

        confirm_button = st.sidebar.button("开始分析")

        if confirm_button:
            with st.spinner(f"正在分析 {selected_home_team_name} vs {selected_away_team_name}..."):
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
                home_expected_goals = home_avg_goals_scored
                away_expected_goals = away_avg_goals_scored
                
                # 应用调整因子（避免除0错误）
                if league_avg_away_goals > 0:
                    home_expected_goals = home_avg_goals_scored * (away_avg_goals_conceded / league_avg_away_goals)
                if league_avg_home_goals > 0:
                    away_expected_goals = away_avg_goals_scored * (home_avg_goals_conceded / league_avg_home_goals)

                # 使用贝叶斯方法调整预期进球数
                prior_var = 1.0  # 先验方差
                observed_var = 0.5  # 观测方差
                home_expected_goals, _ = bayesian_adjustment(home_expected_goals, prior_var, home_avg_goals_scored, observed_var)
                away_expected_goals, _ = bayesian_adjustment(away_expected_goals, prior_var, away_avg_goals_scored, observed_var)

                # 获取并显示联赛积分榜
                standings_data = cache_get_league_standings(API_KEY, league_id)
                if standings_data and standings_data.get('standings'):
                    # 确保standings列表不为空
                    if standings_data['standings']:
                        standings = standings_data['standings'][0].get('table', [])
                        
                        if standings:
                            # 转换为 DataFrame
                            standings_df = pd.DataFrame(standings)

                            # 提取嵌套字段 'team.name' 并添加为新列
                            standings_df['球队名称'] = standings_df['team'].apply(lambda x: x['name'])
            
                            # 计算额外数据
                            standings_df['赛'] = standings_df['playedGames']
                            standings_df['胜'] = standings_df['won']
                            standings_df['平'] = standings_df['draw']
                            standings_df['负'] = standings_df['lost']
                            standings_df['得'] = standings_df['goalsFor']
                            standings_df['失'] = standings_df['goalsAgainst']
                            standings_df['净'] = standings_df['goalDifference']
                            standings_df['均得'] = standings_df['得'] / standings_df['赛']
                            standings_df['均失'] = standings_df['失'] / standings_df['赛']
                            standings_df['积分'] = standings_df['points']

                            # 格式化数据
                            standings_df['均得'] = standings_df['均得'].apply(lambda x: f"{x:.2f}")
                            standings_df['均失'] = standings_df['均失'].apply(lambda x: f"{x:.2f}")

                            # 保留并重命名需要展示的列
                            standings_df = standings_df[['position', '球队名称', '赛', '胜', '平', '负', '得', '失', '净', '均得', '均失', '积分']]
                            standings_df.rename(columns={'position': '排名'}, inplace=True)

                            # 使用 Streamlit 显示表格
                            st.write("### 联赛积分榜")
                            st.dataframe(standings_df, use_container_width=True)
                        else:
                            st.warning("该联赛积分榜为空")
                    else:
                        st.warning("该联赛暂无积分榜数据")
                else:
                    st.warning("无法获取联赛积分榜数据")

                # 显示关键指标
                col1, col2 = st.columns(2)

                with col1:
                    st.metric(f"{selected_home_team_name} 预期进球", f"{home_expected_goals:.2f}", 
                               delta=f"主场均进 {home_avg_goals_scored:.2f} 失 {home_avg_goals_conceded:.2f}")

                with col2:
                    st.metric(f"{selected_away_team_name} 预期进球", f"{away_expected_goals:.2f}", 
                               delta=f"客场均进 {away_avg_goals_scored:.2f} 失 {away_avg_goals_conceded:.2f}")

                # 进球概率计算
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
                expected_goals, std_dev = calculate_expected_goals_and_std(total_goals_prob)

                # 胜率展示
                st.write("### 胜率预测")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"<h4 style='font-size: 40px; color: lightgreen;'>主胜</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='font-size: 20px;'>{home_win_prob:.2%}</h4>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"<h4 style='font-size: 40px; color: navy;'>平局</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='font-size: 20px;'>{draw_prob:.2%}</h4>", unsafe_allow_html=True)

                with col3:
                    st.markdown(f"<h4 style='font-size: 40px; color: pink;'>客胜</h4>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='font-size: 20px;'>{away_win_prob:.2%}</h4>", unsafe_allow_html=True)

                # 让球分析
                st.write("### 让球分析")
                home_handicap_win_prob, away_handicap_win_prob = calculate_handicap_suggestion(home_goals_prob, away_goals_prob, point_handicap)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{selected_home_team_name} 让{abs(point_handicap) if point_handicap < 0 else '受'+str(point_handicap)}球胜率", 
                              f"{home_handicap_win_prob:.2%}")
                with col2:
                    st.metric(f"{selected_away_team_name} {'受'+str(abs(point_handicap)) if point_handicap < 0 else '让'+str(point_handicap)}球胜率", 
                              f"{away_handicap_win_prob:.2%}")
                
                # 投注建议
                advice = ""
                if home_handicap_win_prob > 0.6:
                    advice = f"建议投注主队让{abs(point_handicap) if point_handicap < 0 else '受'+str(point_handicap)}球胜"
                elif away_handicap_win_prob > 0.6:
                    advice = f"建议投注客队{'受'+str(abs(point_handicap)) if point_handicap < 0 else '让'+str(point_handicap)}球胜"
                else:
                    advice = "建议观望或考虑其他投注选项"
                
                st.info(advice)

                # 总进球数分析
                st.write("### 总进球数分析")
                over_prob = sum(total_goals_prob[int(np.floor(total_goals_line))+1:])
                under_prob = sum(total_goals_prob[:int(np.floor(total_goals_line))+1]) - total_goals_prob[int(np.floor(total_goals_line))+1] * (1 - (total_goals_line % 1))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"总进球大 {total_goals_line}", f"{over_prob:.2%}")
                with col2:
                    st.metric(f"总进球小 {total_goals_line}", f"{under_prob:.2%}")
                
                # 单双球分析
                odd_prob, even_prob = calculate_odd_even_probabilities(home_goals_prob, away_goals_prob)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("单数球概率", f"{odd_prob:.2%}")
                with col2:
                    st.metric("双数球概率", f"{even_prob:.2%}")

                # AI 分析报告
                st.write("### AI分析报告")
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

                # 预测最可能比分
                st.write("### 最可能比分")
                exact_score_probs = calculate_exact_score_probabilities(home_goals_prob, away_goals_prob)
                top_scores = get_top_scores(exact_score_probs)
                
                for i, (score, prob) in enumerate(top_scores):
                    col1, col2, col3 = st.columns([1, 2, 1])
                    col1.write(f"第{i+1}可能比分")
                    col2.metric(score, f"{prob:.2%}")
                    col3.progress(min(100, int(prob * 200)))  # 放大显示

                # 比分概率热力图
                st.write("### 比分概率热力图")
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
                    yaxis_title="主队进球数",
                    height=600
                )
                st.plotly_chart(fig)

                # 两队进球分布对比图
                st.write("### 两队进球分布对比")
                # 创建数据框
                home_goal_probs_df = pd.DataFrame({
                    'Goals': range(len(home_goals_prob)),
                    'Probability': home_goals_prob,
                    'Team': [f'{selected_home_team_name} (主队)'] * len(home_goals_prob)
                })

                away_goal_probs_df = pd.DataFrame({
                    'Goals': range(len(away_goals_prob)),
                    'Probability': away_goals_prob,
                    'Team': [f'{selected_away_team_name} (客队)'] * len(away_goals_prob)
                })
                
                goal_probs_df = pd.concat([home_goal_probs_df, away_goal_probs_df])
                goal_probs_df = goal_probs_df[goal_probs_df['Probability'] > 0.001]  # 只显示概率大于0.1%的结果

                fig = px.bar(goal_probs_df, 
                             x='Goals', 
                             y='Probability', 
                             color='Team', 
                             barmode='group',
                             text=goal_probs_df['Probability'].apply(lambda x: f'{x:.1%}' if x >= 0.01 else '<1%'),
                             labels={'Probability': '概率', 'Goals': '进球数'},
                             height=500)
                
                fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
                fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig)

                # 总进球数分布图
                st.write("### 总进球数分布")
                total_goals_prob_df = pd.DataFrame({
                    "总进球数": np.arange(len(total_goals_prob)),
                    "概率 (%)": total_goals_prob * 100
                })
                total_goals_prob_df = total_goals_prob_df[total_goals_prob_df["概率 (%)"] > 0.1]  # 只显示大于0.1%的
                
                fig = px.bar(
                    total_goals_prob_df,
                    x="总进球数",
                    y="概率 (%)",
                    title="总进球数概率分布",
                    color="概率 (%)",
                    text=total_goals_prob_df["概率 (%)"].apply(lambda x: f'{x:.2f}%'),
                    labels={"总进球数": "总进球数", "概率 (%)": "概率 (%)"}
                )
               
                # 设置x轴的刻度间隔为1
                fig.update_xaxes(dtick=1)
                fig.update_traces(textfont_size=12, textangle=0, cliponaxis=False)
                fig.update_layout(height=500)
                st.plotly_chart(fig)
    else:
        st.error("该联赛暂无球队数据")
else:
    st.error("无法获取联赛数据，请检查API密钥或网络连接")
