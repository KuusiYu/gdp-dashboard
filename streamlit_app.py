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
import plotly.colors as colors

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="足球比赛预测分析",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 固定 API 密钥
API_KEY = '0c2379b28acb446bb97bd417f2666f81'

# 设置日志记录
import logging
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

def calculate_average_goals_for_team(matches, team_id, venue=None):
    if not matches or 'matches' not in matches:
        return 0, 0

    goals_scored, goals_conceded = [], []
    
    for match in matches['matches']:
        if venue and match['homeTeam']['id'] != team_id and match['awayTeam']['id'] != team_id:
            continue
        if match['score']['fullTime']['home'] is None or match['score']['fullTime']['away'] is None:
            continue

        if match['homeTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['home'])
            goals_conceded.append(match['score']['fullTime']['away'])
        elif match['awayTeam']['id'] == team_id:
            goals_scored.append(match['score']['fullTime']['away'])
            goals_conceded.append(match['score']['fullTime']['home'])
            
    if not goals_scored:
        return 0, 0

    return np.mean(goals_scored), np.mean(goals_conceded)

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

def generate_ai_analysis(home_team, away_team, home_exp, away_exp, home_win, draw, away_win):
    return f"""
    **AI分析报告**
    
    🧠 **战术分析**: 
    {home_team}的进攻实力估计为 **{home_exp:.2f}** 个预期进球 (xG)，而{away_team}在客场的防守弱点可能导致对方获得更多机会。
    预测比分为 **{round(home_exp)}-{round(away_exp)}** 的概率最高。
    
    📊 **概率分析**:
    - {home_team} 获胜概率: **{home_win:.2%}**
    - 平局概率: **{draw:.2%}**
    - {away_team} 获胜概率: **{away_win:.2%}**
    
    💡 **投资建议**:
    - 当主队获胜概率 > 60% 时值得投资
    - 当平局概率 > 30% 时可考虑下注X
    - 比分建议关注 **{max(1, int(home_exp))}-{max(0, int(away_exp))}**
    - 总进球建议 **{"大" if home_exp + away_exp > 2.5 else "小"}于2.5球**
    """

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
    </style>
    """, unsafe_allow_html=True)

# 使用高级UI效果
create_gradient_header()

# 主界面
st.title('⚽ 足球比赛智能预测系统')
st.caption("基于泊松分布与AI算法的高级足球赛事分析")

# 侧边栏设置
with st.sidebar:
    st.markdown("### 比赛参数设置")
    
    leagues_data = cache_get_leagues(API_KEY)
    leagues = {league['name']: league['id'] for league in leagues_data['competitions']} if leagues_data else {}
    selected_league = st.selectbox('选择联赛', list(leagues.keys()), key='league')
    league_id = leagues[selected_league] if selected_league else None

    if league_id:
        teams_data = cache_get_teams_in_league(API_KEY, league_id)
        if teams_data:
            teams = {team['name']: team['id'] for team in teams_data['teams']}
            col1, col2 = st.columns(2)
            with col1:
                selected_home = st.selectbox('主队', list(teams.keys()), key='home_team')
            with col2:
                selected_away = st.selectbox('客队', list(teams.keys()), key='away_team')
            
            point_handicap = st.slider('让球盘口', -3.0, 3.0, 0.0, 0.25, 
                                      help="负数为主让球，正数为客让球")
            total_goals_line = st.slider('大小球盘口', 0.0, 6.0, 2.5, 0.25)
            
            if st.button('开始分析', use_container_width=True):
                st.session_state['analyze'] = True
            else:
                st.session_state['analyze'] = False

# 如果开始分析
if st.session_state.get('analyze') and selected_home and selected_away:
    home_id = teams[selected_home]
    away_id = teams[selected_away]
    
    with st.spinner('正在获取数据并分析...'):
        try:
            # 获取比赛数据
            home_matches = cache_get_team_matches(API_KEY, home_id, 'HOME')
            away_matches = cache_get_team_matches(API_KEY, away_id, 'AWAY')
            league_matches = cache_get_league_matches(API_KEY, league_id)
            
            # 计算进球数据
            home_scored, home_conceded = calculate_average_goals_for_team(home_matches, home_id, 'HOME')
            away_scored, away_conceded = calculate_average_goals_for_team(away_matches, away_id, 'AWAY')
            league_home_avg, league_away_avg = calculate_league_average_goals(league_matches)
            
            # 计算预期进球
            home_exp = home_scored * (away_conceded / league_away_avg) if league_away_avg else home_scored
            away_exp = away_scored * (home_conceded / league_home_avg) if league_home_avg else away_scored
            
            # 贝叶斯调整
            home_exp, _ = bayesian_adjustment(home_exp, 1.0, home_scored, 0.5)
            away_exp, _ = bayesian_adjustment(away_exp, 1.0, away_scored, 0.5)
            
            # 创建概率分布
            home_probs = np.array(poisson_prediction(home_exp))
            home_probs /= home_probs.sum()
            away_probs = np.array(poisson_prediction(away_exp))
            away_probs /= away_probs.sum()
            
            # 模拟结果
            home_win, draw, away_win = calculate_match_outcome_probabilities(home_probs, away_probs)
            home_handicap_win, away_handicap_win = calculate_handicap_suggestion(home_probs, away_probs, point_handicap)
            
            # 总进球数概率
            total_probs = calculate_total_goals_prob(home_probs, away_probs)
            expected_goals = home_exp + away_exp
            
            # 单双球概率
            odd_prob, even_prob = calculate_odd_even_probabilities(home_probs, away_probs)
            
            # 创建UI布局
            
            # 关键指标展示
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.markdown(f"<div class='compact-card'>{selected_home}<br><span class='value-card'>{home_exp:.2f}</span>xG</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown(f"<div class='compact-card'>{selected_away}<br><span class='value-card'>{away_exp:.2f}</span>xG</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown(f"<div class='compact-card'>预计总进球<br><span class='value-card'>{expected_goals:.2f}</span></div>", unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"<div class='compact-card'>让球盘口<br><span class='value-card'>{point_handicap:+.1f}</span></div>", unsafe_allow_html=True)
                
            with col5:
                st.markdown(f"<div class='compact-card'>大小盘口<br><span class='value-card'>{total_goals_line:.1f}</span></div>", unsafe_allow_html=True)
            
            # 胜平负概率展示
            st.subheader("比赛胜负预测")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"{selected_home} 获胜", 
                    value=f"{home_win:.1%}", 
                    delta=f"让球胜率: {home_handicap_win:.1%}",
                    delta_color="inverse" if home_handicap_win < 0.5 else "normal"
                )
                st.progress(min(1.0, home_handicap_win), text=None)
                
            with col2:
                st.metric(
                    label="平局", 
                    value=f"{draw:.1%}"
                )
                st.progress(min(1.0, draw), text=None)
                
            with col3:
                st.metric(
                    label=f"{selected_away} 获胜", 
                    value=f"{away_win:.1%}", 
                    delta=f"让球胜率: {away_handicap_win:.1%}",
                    delta_color="inverse" if away_handicap_win < 0.5 else "normal"
                )
                st.progress(min(1.0, away_handicap_win), text=None)
            
            # 核心预测图表
            st.subheader("核心预测")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # 热力图
                score_probs = score_probability(home_probs, away_probs)
                df = pd.DataFrame(score_probs, 
                                columns=[f"客{i}" for i in range(len(away_probs))],
                                index=[f"主{i}" for i in range(len(home_probs))])
                
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
                goals_df = pd.DataFrame({
                    "进球数": list(range(len(home_probs))) + list(range(len(away_probs))),
                    "概率": list(home_probs) + list(away_probs),
                    "球队": [selected_home]*len(home_probs) + [selected_away]*len(away_probs)
                })
                
                fig = px.bar(
                    goals_df, 
                    x="进球数", 
                    y="概率", 
                    color="球队", 
                    barmode="group",
                    title="进球数概率分布"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # 总进球概率
                total_df = pd.DataFrame({
                    "总进球数": np.arange(len(total_probs)),
                    "概率": total_probs
                })
                fig = px.line(
                    total_df, 
                    x="总进球数", 
                    y="概率", 
                    title="总进球数分布",
                    markers=True
                )
                fig.update_layout(height=250)
                fig.add_vline(x=total_goals_line, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # 详细预测分析
            st.subheader("详细预测")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # 最可能比分
                top_scores = get_top_scores(home_probs, away_probs)
                st.markdown("**最可能比分**")
                for i, (score, prob) in enumerate(top_scores):
                    col_a, col_b = st.columns([1, 4])
                    col_a.metric(label=f"{i+1}.", value=score)
                    col_b.progress(prob, text=f"{prob:.2%}")
            
            with col2:
                # 其他投注分析
                st.markdown("**其他投注概率**")
                
                col1, col2 = st.columns(2)
                col1.metric("总进球 > 盘口", f"{sum(total_probs[int(np.floor(total_goals_line))+1:]):.2%}")
                col2.metric("总进球 < 盘口", f"{sum(total_probs[:int(np.floor(total_goals_line))+1]):.2%}")
                
                col3, col4 = st.columns(2)
                col3.metric("单数球", f"{odd_prob:.2%}")
                col4.metric("双数球", f"{even_prob:.2%}")
                
                st.markdown("---")
                st.markdown(f"**让球分析 ({point_handicap:+.1f})**")
                col5, col6 = st.columns(2)
                col5.metric(f"{selected_home} 让球胜", f"{home_handicap_win:.2%}")
                col6.metric(f"{selected_away} 受让胜", f"{away_handicap_win:.2%}")
            
            # AI分析报告
            with st.expander("📈 AI分析报告", expanded=True):
                st.markdown(generate_ai_analysis(
                    selected_home, selected_away, 
                    home_exp, away_exp, 
                    home_win, draw, away_win
                ))
                
            # 联赛积分榜
            standings_data = cache_get_league_standings(API_KEY, league_id)
            if standings_data and standings_data.get('standings') and standings_data['standings']:
                standings = standings_data['standings'][0].get('table', [])
                
                if standings:
                    st.subheader("联赛积分榜")
                    
                    standings_df = pd.DataFrame(standings)
                    standings_df['球队名称'] = standings_df['team'].apply(lambda x: x['name'])
                    standings_df = standings_df[[
                        'position', 'playedGames', 'won', 'draw', 'lost', 
                        'goalsFor', 'goalsAgainst', 'goalDifference', 'points', '球队名称'
                    ]]
                    standings_df.columns = [
                        '排名', '场', '胜', '平', '负', 
                        '进', '失', '净', '分', '球队'
                    ]
                    
                    # 简洁展示前6名和最后6名
                    top6 = standings_df.head(6)
                    bottom6 = standings_df.tail(6)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**积分榜前列**")
                        st.dataframe(top6.style.background_gradient(subset=['分', '进'], cmap='Greens'), hide_index=True)
                    
                    with col2:
                        st.markdown("**积分榜末尾**")
                        st.dataframe(bottom6.style.background_gradient(subset=['分', '进'], cmap='Reds'), hide_index=True)
            
            st.markdown("---")
            st.markdown('<div class="footer">足球预测分析系统 © 2023 | 基于足球数据API与泊松分布模型</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"分析过程中出现错误: {str(e)}")
            logging.exception("分析错误")
else:
    # 展示欢迎信息
    st.info("请在左侧选择联赛和球队开始分析")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("https://img.freepik.com/free-vector/soccer-stadium-background_52683-43536.jpg?w=2000", caption="足球赛事预测分析平台")
    with col2:
        st.markdown("""
        ## 足球赛事预测分析平台
        
        🔍 本系统使用先进的泊松分布模型和实时足球数据，提供专业的比赛预测分析。
        
        ### 功能特点：
        - 实时比赛数据接入
        - 胜平负概率预测
        - 让球盘口分析
        - 大小球盘口分析
        - 最可能比分预测
        - AI赛事分析报告
        - 联赛积分榜查看
        
        ### 使用指南：
        1. 在左侧选择联赛
        2. 选择主队和客队
        3. 设置让球和大小球盘口
        4. 点击"开始分析"获取预测
        """)
