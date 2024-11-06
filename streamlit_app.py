import streamlit as st
import numpy as np
import pandas as pd

# 计算凯利指数
def calculate_kelly(probability, odds):
    if odds <= 1:  # 如果赔率小于等于1，凯利公式不适用，这里返回0
        return 0
    q = 1 - probability
    b = odds - 1
    return (b * probability - q) / b if b > 0 else 0

# 模拟球队得分
def generate_scores(avg_goals, avg_conceded, size, factors):
    goals = np.random.poisson(avg_goals * factors, size)
    conceded = np.random.poisson(avg_conceded * factors, size)
    return goals, conceded

# 分析数据来计算次数和百分比
def analyze_data(data, column_name):
    counts = data.value_counts()
    percentages = 100 * data.value_counts(normalize=True)
    return pd.DataFrame({column_name: counts.index, '次数': counts.values, '百分比': percentages.values})

# 计算比赛的盘口结果
def calculate_handicap_result(home_goals, away_goals, handicap):
    if home_goals + handicap > away_goals:
        return '胜'
    elif home_goals + handicap < away_goals:
        return '负'
    else:
        return '平'

# 计算大小球盘口结果
def calculate_over_under_result(total_goals, threshold):
    if total_goals > threshold:
        return '大'
    elif total_goals < threshold:
        return '小'
    else:
        return '平'

# 计算爆冷指数
def calculate_upset_index(away_strength, home_strength):
    if home_strength == 0:
        return 0
    return min(1, np.log(away_strength / home_strength) / np.log(2))

# Streamlit 应用
st.title('杯赛预测模拟器')

# 输入参数
st.sidebar.title("输入参数")
home_team_name = st.sidebar.text_input('主队名称', '主队')
away_team_name = st.sidebar.text_input('客队名称', '客队')
home_avg_goals = st.sidebar.number_input('主队场均进球', value=1.5, format="%.1f")
away_avg_goals = st.sidebar.number_input('客队场均进球', value=1.2, format="%.1f")

# 输入场均丢球数
home_avg_conceded = st.sidebar.number_input('主队场均失球', value=1.1, format="%.1f")
away_avg_conceded = st.sidebar.number_input('客队场均失球', value=1.3, format="%.1f")

n_simulations = st.sidebar.number_input('模拟次数', value=10000, step=100)

# 获取输入的球队实力因子
home_strength = st.sidebar.number_input('主队实力因子', value=1.0, format="%.2f")
away_strength = st.sidebar.number_input('客队实力因子', value=1.0, format="%.2f")

# 让球和大小球盘口输入
selected_handicap = st.sidebar.slider('选择让球盘口', -5.0, 5.5, 0.0, step=0.25)
handicap_odds_home = st.sidebar.slider('让球盘口赔率 (主队赢)', 1.0, 5.0, 2.0)
handicap_odds_away = st.sidebar.slider('让球盘口赔率 (客队赢)', 1.0, 5.0, 2.0)

selected_ou_threshold = st.sidebar.slider('选择大小球盘口', 0.0, 10.5, 2.5, step=0.25)
ou_odds_over = st.sidebar.slider('大分赔率', 1.0, 5.0, 2.0)
ou_odds_under = st.sidebar.slider('小分赔率', 1.0, 5.0, 2.0)

# 输入本金
capital = st.sidebar.number_input('本金', min_value=0.0, value=1000.0, format="%.2f")

# 模拟数据
home_goals_list, home_conceded_list = generate_scores(home_avg_goals, home_avg_conceded, n_simulations, np.random.normal(1.0, 0.1, n_simulations))
away_goals_list, away_conceded_list = generate_scores(away_avg_goals, away_avg_conceded, n_simulations, np.random.normal(1.0, 0.1, n_simulations))

# 计算比赛结果
total_goals_list = home_goals_list + away_goals_list
match_results = np.where(home_goals_list > away_goals_list, '胜', 
                         np.where(home_goals_list < away_goals_list, '负', '平'))

# 结果分析展示
results_analysis = analyze_data(pd.Series(match_results), '比赛结果')

# 获取比赛结果（胜平负）概率
prob_win = results_analysis.loc[results_analysis['比赛结果'] == '胜', '百分比'].sum() / 100
prob_draw = results_analysis.loc[results_analysis['比赛结果'] == '平', '百分比'].sum() / 100
prob_lose = results_analysis.loc[results_analysis['比赛结果'] == '负', '百分比'].sum() / 100

# 计算总进球数统计
total_goals_analysis = analyze_data(pd.Series(total_goals_list), '总进球数')

# 计算让球盘口结果
handicap_results = [calculate_handicap_result(home_goals, away_goals, selected_handicap) 
                    for home_goals, away_goals in zip(home_goals_list, away_goals_list)]
handicap_results_analysis = analyze_data(pd.Series(handicap_results), '让球结果')
handicap_prob_home = handicap_results_analysis.loc[handicap_results_analysis['让球结果'] == '胜', '百分比'].sum() / 100
handicap_prob_away = handicap_results_analysis.loc[handicap_results_analysis['让球结果'] == '负', '百分比'].sum() / 100

# 计算大小球结果
over_under_results = [calculate_over_under_result(total_goals, selected_ou_threshold) 
                      for total_goals in total_goals_list]
ou_results_analysis = analyze_data(pd.Series(over_under_results), '大小球结果')
ou_prob_over = ou_results_analysis.loc[ou_results_analysis['大小球结果'] == '大', '百分比'].sum() / 100
ou_prob_under = ou_results_analysis.loc[ou_results_analysis['大小球结果'] == '小', '百分比'].sum() / 100

# 计算爆冷指数
upset_index = calculate_upset_index(away_strength, home_strength)
upset_probability = upset_index * 100  # 转换为百分比

# 计算凯利下注建议
kelly_bet_amount_home = calculate_kelly(handicap_prob_home, handicap_odds_home) * capital if handicap_prob_home > 0 else 0
kelly_bet_amount_away = calculate_kelly(handicap_prob_away, handicap_odds_away) * capital if handicap_prob_away > 0 else 0
kelly_bet_amount_over = calculate_kelly(ou_prob_over, ou_odds_over) * capital if ou_prob_over > 0 else 0
kelly_bet_amount_under = calculate_kelly(ou_prob_under, ou_odds_under) * capital if ou_prob_under > 0 else 0

# 显示比赛结果统计
st.header(f"{home_team_name} vs {away_team_name} 比赛结果统计")
for index, row in results_analysis.iterrows():
    result_prob = row['百分比'] / 100
    kelly_index = calculate_kelly(result_prob, 2.0)  # 使用默认赔率2.0
    st.write(f"结果: {row['比赛结果']} - 次数: {row['次数']} - 百分比: {row['百分比']:.2f}% - 凯利指数: {kelly_index:.4f}")

# 显示总进球数统计
st.header("总进球数统计")
for index, row in total_goals_analysis.iterrows():
    prob = row['百分比'] / 100
    kelly_index = calculate_kelly(prob, 2.0)  # 设置默认赔率为2.0
    st.write(f"总进球数: {row['总进球数']} - 次数: {row['次数']} - 百分比: {row['百分比']:.2f}% - 凯利指数: {kelly_index:.4f}")

# 主队与客队进球数统计
home_goals_analysis = analyze_data(pd.Series(home_goals_list), '主队进球数')
away_goals_analysis = analyze_data(pd.Series(away_goals_list), '客队进球数')

st.header("主队进球数统计")
for index, row in home_goals_analysis.iterrows():
    prob = row['百分比'] / 100
    kelly_index = calculate_kelly(prob, 2.0)
    st.write(f"进球数: {row['主队进球数']} - 次数: {row['次数']} - 百分比: {row['百分比']:.2f}% - 凯利指数: {kelly_index:.4f}")

st.header("客队进球数统计")
for index, row in away_goals_analysis.iterrows():
    prob = row['百分比'] / 100
    kelly_index = calculate_kelly(prob, 2.0)
    st.write(f"进球数: {row['客队进球数']} - 次数: {row['次数']} - 百分比: {row['百分比']:.2f}% - 凯利指数: {kelly_index:.4f}")

# 显示让球盘口统计
st.header(f"盘口为 {selected_handicap} 的让球盘口统计")
st.write(f"主队胜出概率: {handicap_prob_home * 100:.2f}%，凯利指数: {calculate_kelly(handicap_prob_home, handicap_odds_home):.4f}")
st.write(f"客队胜出概率: {handicap_prob_away * 100:.2f}%，凯利指数: {calculate_kelly(handicap_prob_away, handicap_odds_away):.4f}")

# 显示大小球盘口统计
st.header(f"大小球盘口为 {selected_ou_threshold} 的统计")
st.write(f"大球概率: {ou_prob_over * 100:.2f}%，凯利指数: {calculate_kelly(ou_prob_over, ou_odds_over):.4f}")
st.write(f"小球概率: {ou_prob_under * 100:.2f}%，凯利指数: {calculate_kelly(ou_prob_under, ou_odds_under):.4f}")

# 显示爆冷指数
st.header("爆冷统计")
st.write(f"爆冷指数（客队胜率）: {upset_probability:.2f}%")

# 显示凯利下注建议
st.header("凯利下注建议")
st.write(f"建议在主队投注金额: {kelly_bet_amount_home:.2f}")
st.write(f"建议在客队投注金额: {kelly_bet_amount_away:.2f}")
st.write(f"建议在大球投注金额: {kelly_bet_amount_over:.2f}")
st.write(f"建议在小球投注金额: {kelly_bet_amount_under:.2f}")
