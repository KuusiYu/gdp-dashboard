import streamlit as st
import numpy as np
import pandas as pd

# 计算凯利指数
def calculate_kelly(probability, odds):
    kelly_index = (probability * (odds - 1) - (1 - probability)) / (odds - 1)
    return np.maximum(0, kelly_index)

# 模拟数据生成函数
def generate_scores(avg_goals, size, factors):
    adjusted_goals = avg_goals * factors
    return np.random.poisson(adjusted_goals, size)

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

def analyze_handicaps(selected_handicap, home_goals_list, away_goals_list):
    results = [calculate_handicap_result(home_goals, away_goals, selected_handicap) 
               for home_goals, away_goals in zip(home_goals_list, away_goals_list)]
    results_analysis = analyze_data(pd.Series(results), '比赛结果')
    return results_analysis

def analyze_over_under(selected_threshold, total_goals_list):
    results = [calculate_over_under_result(total_goals, selected_threshold) 
               for total_goals in total_goals_list]
    results_analysis = analyze_data(pd.Series(results), '大小球结果')
    return results_analysis

# Streamlit应用
st.title('足球比赛模拟器')

# 输入参数
st.sidebar.title("输入参数")
home_avg_goals = st.sidebar.number_input('主队场均进球', value=1.5, format="%.1f")
away_avg_goals = st.sidebar.number_input('客队场均进球', value=1.2, format="%.1f")
home_avg_conceded = st.sidebar.number_input('主队场均失球', value=1.1, format="%.1f")
away_avg_conceded = st.sidebar.number_input('客队场均失球', value=1.3, format="%.1f")
n_simulations = st.sidebar.number_input('模拟次数', value=7500, step=100)

selected_handicap = st.sidebar.slider('选择让球盘口', -5.0, 5.5, 0.0, step=0.25)
handicap_odds = st.sidebar.slider('让球盘口赔率', 1.0, 5.0, 2.0)

selected_ou_threshold = st.sidebar.slider('选择大小球盘口', 0.0, 10.5, 2.5, step=0.25)
ou_odds = st.sidebar.slider('大小球盘口赔率', 1.0, 5.0, 2.0)

# 模拟数据
weather_factors = np.random.normal(1.0, 0.1, n_simulations)
team_factors = np.random.normal(1.0, 0.1, n_simulations)
home_away_factors = np.random.normal(1.0, 0.05, n_simulations)
card_factors = np.random.normal(1.0, 0.05, n_simulations)

home_goals_list = generate_scores(home_avg_goals, n_simulations, weather_factors * team_factors * home_away_factors * card_factors)
away_goals_list = generate_scores(away_avg_goals, n_simulations, weather_factors * team_factors / home_away_factors * card_factors)

# 计算统计数据
total_goals_list = home_goals_list + away_goals_list
match_scores_list = [f"{hg}-{ag}" for hg, ag in zip(home_goals_list, away_goals_list)]
match_results = np.where(home_goals_list > away_goals_list, '胜', np.where(home_goals_list < away_goals_list, '负', '平'))

# 获取自动计算的盘口概率
handicap_results_analysis = analyze_handicaps(selected_handicap, home_goals_list, away_goals_list)
handicap_prob = handicap_results_analysis.loc[handicap_results_analysis['比赛结果'] == '胜', '百分比'].sum()

ou_results_analysis = analyze_over_under(selected_ou_threshold, total_goals_list)
ou_prob = ou_results_analysis.loc[ou_results_analysis['大小球结果'] == '大', '百分比'].sum()

# 添加赔率和凯利指数
handicap_results_analysis['赔率'] = 100.0 / handicap_results_analysis['百分比']
handicap_results_analysis['凯利指数'] = calculate_kelly(handicap_results_analysis['百分比'] / 100, handicap_odds)

ou_results_analysis['赔率'] = 100.0 / ou_results_analysis['百分比']
ou_results_analysis['凯利指数'] = calculate_kelly(ou_results_analysis['百分比'] / 100, ou_odds)

# 结果分析展示
results_analysis = analyze_data(pd.Series(match_results), '比赛结果')
home_goals_analysis = analyze_data(pd.Series(home_goals_list), '进球数')
away_goals_analysis = analyze_data(pd.Series(away_goals_list), '进球数')
total_goals_analysis = analyze_data(pd.Series(total_goals_list), '总进球数')
match_scores_analysis = analyze_data(pd.Series(match_scores_list), '比分')

# 显示比赛结果统计
st.header("比赛结果统计")
st.dataframe(results_analysis)

st.header("主队进球数统计")
st.dataframe(home_goals_analysis)

st.header("客队进球数统计")
st.dataframe(away_goals_analysis)

st.header("总进球数统计")
st.dataframe(total_goals_analysis)

st.header("比分统计（前十）")
st.dataframe(match_scores_analysis.nlargest(10, '百分比'))

st.header(f"盘口为 {selected_handicap} 的让球盘口统计")
st.dataframe(handicap_results_analysis.style.highlight_max(axis=0, subset=['凯利指数'], color='lightgreen'))

st.header(f"大小球盘口为 {selected_ou_threshold} 的统计")
st.dataframe(ou_results_analysis.style.highlight_max(axis=0, subset=['凯利指数'], color='lightgreen'))

# 自定义凯利指数计算器
st.sidebar.header("自定义凯利指数计算器")
prob_input = st.sidebar.slider("自定义概率", 0.0, 1.0, 0.5)
odds_input = st.sidebar.number_input("自定义赔率", value=2.0, step=0.1)
kelly_result = calculate_kelly(prob_input, odds_input)
st.sidebar.write(f"计算得出的凯利指数: {kelly_result:.2f}")

if kelly_result > 0.1:
    st.sidebar.success("凯利指数显示，此下注可能有利可图。")
elif kelly_result > 0:
    st.sidebar.info("凯利指数较低，风险较大。")
else:
    st.sidebar.error("不建议下注。凯利指数为零。")
