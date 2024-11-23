import streamlit as st
import numpy as np
import pandas as pd

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

# 计算凯利指数
def calculate_kelly(probability, odds):
    if odds <= 1:  # 如果赔率小于等于1，凯利公式不适用，这里返回0
        return 0
    q = 1 - probability
    b = odds - 1
    return (b * probability - q) / b if b > 0 else 0

# Streamlit 应用
st.title('⚽ 足球预测模拟器')

# 输入参数
st.sidebar.title("输入参数设置")
st.sidebar.info("请根据以下选项输入相关数据进行预测。")

with st.sidebar.form(key='input_form'):
    # 场均进球和失球输入
    home_avg_goals = st.number_input('主队场均进球', value=1.5, min_value=0.0, format="%.1f", help="主队在比赛中平均进球数")
    away_avg_goals = st.number_input('客队场均进球', value=1.2, min_value=0.0, format="%.1f", help="客队在比赛中平均进球数")
    home_avg_conceded = st.number_input('主队场均失球', value=1.1, min_value=0.0, format="%.1f", help="主队在比赛中平均失球数")
    away_avg_conceded = st.number_input('客队场均失球', value=1.3, min_value=0.0, format="%.1f", help="客队在比赛中平均失球数")

    # xG输入开关
    use_xg = st.checkbox('启用实时xG进球数计算', value=False)

    if use_xg:
        st.subheader("输入比赛统计数据（用于计算xG）")
        home_attacks = st.number_input('主队进攻次数', value=0, min_value=0, help="主队的进攻次数")
        home_dangerous_attacks = st.number_input('主队危险进攻次数', value=0, min_value=0, help="主队的危险进攻次数")
        home_possession = st.number_input('主队控球率 (%)', value=50, min_value=0, max_value=100, help="主队的控球率")
        home_shots = st.number_input('主队射门次数', value=0, min_value=0, help="主队的射门次数")
        home_shots_on_target = st.number_input('主队射正次数', value=0, min_value=0, help="主队的射正次数")
        home_shots_off_target = st.number_input('主队射偏次数', value=0, min_value=0, help="主队的射偏次数")
        home_corners = st.number_input('主队角球数', value=0, min_value=0, help="主队的角球数")

        away_attacks = st.number_input('客队进攻次数', value=0, min_value=0, help="客队的进攻次数")
        away_dangerous_attacks = st.number_input('客队危险进攻次数', value=0, min_value=0, help="客队的危险进攻次数")
        away_possession = st.number_input('客队控球率 (%)', value=50, min_value=0, max_value=100, help="客队的控球率")
        away_shots = st.number_input('客队射门次数', value=0, min_value=0, help="客队的射门次数")
        away_shots_on_target = st.number_input('客队射正次数', value=0, min_value=0, help="客队的射正次数")
        away_shots_off_target = st.number_input('客队射偏次数', value=0, min_value=0, help="客队的射偏次数")
        away_corners = st.number_input('客队角球数', value=0, min_value=0, help="客队的角球数")
    else:
        st.write("请注意，未启用xG计算，将直接使用场均进球数。")

    # 盘口输入
    selected_handicap = st.slider('选择让球盘口', -5.0, 5.5, 0.0, step=0.25, help="选择主队的让球盘口")
    handicap_odds_home = st.slider('让球盘口赔率 (主队赢)', 1.0, 5.0, 2.0, help="主队赢的赔率")
    handicap_odds_away = st.slider('让球盘口赔率 (客队赢)', 1.0, 5.0, 2.0, help="客队赢的赔率")

    # 大小球盘口输入
    selected_ou_threshold = st.slider('选择大小球盘口', 0.0, 10.5, 2.5, step=0.25, help="总进球数的大小盘口")
    ou_odds_over = st.slider('大分赔率', 1.0, 5.0, 2.0, help="大球的赔率")
    ou_odds_under = st.slider('小分赔率', 1.0, 5.0, 2.0, help="小球的赔率")

    capital = st.number_input('本金', min_value=0.0, value=1000.0, format="%.2f", help="可用于投注的本金")

    submit_button = st.form_submit_button(label='开始模拟')

# 设置模拟次数
n_simulations = 730000  # 固定为730000次模拟

if submit_button:
    # 计算主队和客队的预期进球数 (xG)
    if use_xg:
        # 如果启用xG，则根据输入数据计算xG
        home_xg = (home_shots_on_target * 0.3) + (home_shots_off_target * 0.05) + (home_corners * 0.1)
        away_xg = (away_shots_on_target * 0.3) + (away_shots_off_target * 0.05) + (away_corners * 0.1)
    else:
        # 如果未启用xG，则将xG设为0
        home_xg = 0
        away_xg = 0

    # 显示即时xG
    st.header("📊 实时预期进球数 (xG)")
    st.write(f"主队预期进球数: {home_xg:.2f}")
    st.write(f"客队预期进球数: {away_xg:.2f}")

    # 计算场均进球数与xG结合
    home_avg_goals_with_xg = home_avg_goals + home_xg
    away_avg_goals_with_xg = away_avg_goals + away_xg

    # 模拟得分
    home_goals_list, _ = generate_scores(home_avg_goals_with_xg, home_avg_conceded, n_simulations, np.random.normal(1.0, 0.1, n_simulations))
    away_goals_list, _ = generate_scores(away_avg_goals_with_xg, away_avg_conceded, n_simulations, np.random.normal(1.0, 0.1, n_simulations))

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

    # 计算凯利下注建议
    kelly_bet_amount_home = calculate_kelly(handicap_prob_home, handicap_odds_home) * capital if handicap_prob_home > 0 else 0
    kelly_bet_amount_away = calculate_kelly(handicap_prob_away, handicap_odds_away) * capital if handicap_prob_away > 0 else 0
    kelly_bet_amount_over = calculate_kelly(ou_prob_over, ou_odds_over) * capital if ou_prob_over > 0 else 0
    kelly_bet_amount_under = calculate_kelly(ou_prob_under, ou_odds_under) * capital if ou_prob_under > 0 else 0

    # 显示比赛结果统计
    st.header("🏆 比赛结果统计")
    results_analysis = results_analysis.sort_values(by='次数', ascending=False).head(10)  # 只显示前10
    st.table(results_analysis)

    # 显示总进球数统计
    st.header("⚽ 总进球数统计")
    total_goals_analysis = total_goals_analysis.sort_values(by='次数', ascending=False).head(10)  # 只显示前10
    st.table(total_goals_analysis)

    # 比分统计
    score_analysis = pd.DataFrame({
        '比分': [f"{home_goals} : {away_goals}" for home_goals, away_goals in zip(home_goals_list, away_goals_list)],
        '次数': np.ones(n_simulations),
    })
    score_analysis = score_analysis.groupby('比分').count().reset_index()
    score_analysis['百分比'] = (score_analysis['次数'] / n_simulations) * 100
    score_analysis = score_analysis.sort_values(by='次数', ascending=False).head(10)  # 只显示前10

    st.header("📈 比分统计")
    st.table(score_analysis)

    # 显示主队和客队进球数统计并列显示
    st.header("🏡 主队与客队进球数统计")

    # 主队进球数统计（前10）
    home_goals_analysis = analyze_data(pd.Series(home_goals_list), '主队进球数').sort_values(by='次数', ascending=False).head(10)

    # 客队进球数统计（前10）
    away_goals_analysis = analyze_data(pd.Series(away_goals_list), '客队进球数').sort_values(by='次数', ascending=False).head(10)

    # 合并主队和客队进球数分析
    combined_goals_analysis = pd.DataFrame({
        '主队进球数': home_goals_analysis['主队进球数'],
        '主队次数': home_goals_analysis['次数'],
        '客队进球数': away_goals_analysis['客队进球数'],
        '客队次数': away_goals_analysis['次数']
    })

    # 填充空值以确保并列格式
    combined_goals_analysis = combined_goals_analysis.fillna(0)

    # 显示主队和客队进球数并列
    st.table(combined_goals_analysis)

    # 显示让球盘口统计
    st.header(f"盘口为 {selected_handicap} 的让球盘口统计")
    st.write(f"主队胜出概率: {handicap_prob_home * 100:.2f}%，凯利指数: {calculate_kelly(handicap_prob_home, handicap_odds_home):.4f}")
    st.write(f"客队胜出概率: {handicap_prob_away * 100:.2f}%，凯利指数: {calculate_kelly(handicap_prob_away, handicap_odds_away):.4f}")

    # 显示大小球盘口统计
    st.header(f"大小球盘口为 {selected_ou_threshold} 的统计")
    st.write(f"大球概率: {ou_prob_over * 100:.2f}%，凯利指数: {calculate_kelly(ou_prob_over, ou_odds_over):.4f}")
    st.write(f"小球概率: {ou_prob_under * 100:.2f}%，凯利指数: {calculate_kelly(ou_prob_under, ou_odds_under):.4f}")

    # 显示凯利下注建议
    st.header("💰 凯利下注建议")
    st.write(f"建议在主队投注金额: {kelly_bet_amount_home:.2f}")
    st.write(f"建议在客队投注金额: {kelly_bet_amount_away:.2f}")
    st.write(f"建议在大球投注金额: {kelly_bet_amount_over:.2f}")
    st.write(f"建议在小球投注金额: {kelly_bet_amount_under:.2f}")

    st.success("模拟完成！请根据结果做出明智的投注决策。")
