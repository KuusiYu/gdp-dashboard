import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os
import pandas as pd

# 检查字体路径是否正确
font_path = 'C:/Windows/Fonts/simsun.ttc'  # 例如，在Windows系统中使用宋体
if not os.path.exists(font_path):
    st.error("字体文件路径不正确，请检查并更改为有效路径")
else:
    font = FontProperties(fname=font_path)  # 加载字体文件

    # Streamlit应用程序标题
    st.title("篮球比赛预测模拟器")

    # 在侧边栏添加用户输入
    st.sidebar.title("输入参数")
    
    # 主队和客队整体平均得分与失分
    home_team_avg_points_for = st.sidebar.number_input("主队近期场均得分", value=105.0, format="%.2f")
    home_team_avg_points_against = st.sidebar.number_input("主队近期场均失分", value=99.0, format="%.2f")
    away_team_avg_points_for = st.sidebar.number_input("客队近期场均得分", value=102.0, format="%.2f")
    away_team_avg_points_against = st.sidebar.number_input("客队近期场均失分", value=101.0, format="%.2f")

    # 增加开关来控制是否使用各节得分和失分进行预测
    use_quarter_scores = st.sidebar.checkbox("使用各节得分和失分进行预测", value=True)

    if use_quarter_scores:
        # 主队和客队各节平均得分与失分
        home_team_q1_avg_points_for = st.sidebar.number_input("主队第一节平均得分", value=26.0, format="%.2f")
        home_team_q1_avg_points_against = st.sidebar.number_input("主队第一节平均失分", value=25.0, format="%.2f")
        home_team_q2_avg_points_for = st.sidebar.number_input("主队第二节平均得分", value=27.0, format="%.2f")
        home_team_q2_avg_points_against = st.sidebar.number_input("主队第二节平均失分", value=26.0, format="%.2f")
        home_team_q3_avg_points_for = st.sidebar.number_input("主队第三节平均得分", value=27.0, format="%.2f")
        home_team_q3_avg_points_against = st.sidebar.number_input("主队第三节平均失分", value=26.0, format="%.2f")
        home_team_q4_avg_points_for = st.sidebar.number_input("主队第四节平均得分", value=25.0, format="%.2f")
        home_team_q4_avg_points_against = st.sidebar.number_input("主队第四节平均失分", value=24.0, format="%.2f")

        away_team_q1_avg_points_for = st.sidebar.number_input("客队第一节平均得分", value=25.0, format="%.2f")
        away_team_q1_avg_points_against = st.sidebar.number_input("客队第一节平均失分", value=26.0, format="%.2f")
        away_team_q2_avg_points_for = st.sidebar.number_input("客队第二节平均得分", value=26.0, format="%.2f")
        away_team_q2_avg_points_against = st.sidebar.number_input("客队第二节平均失分", value=27.0, format="%.2f")
        away_team_q3_avg_points_for = st.sidebar.number_input("客队第三节平均得分", value=26.0, format="%.2f")
        away_team_q3_avg_points_against = st.sidebar.number_input("客队第三节平均失分", value=27.0, format="%.2f")
        away_team_q4_avg_points_for = st.sidebar.number_input("客队第四节平均得分", value=24.0, format="%.2f")
        away_team_q4_avg_points_against = st.sidebar.number_input("客队第四节平均失分", value=25.0, format="%.2f")

    over_under_line = st.sidebar.number_input("大小分", value=210.5, format="%.2f")
    spread = st.sidebar.number_input("让分 (主队让分)", value=-5.5, format="%.2f")
    odds_home_team = st.sidebar.number_input("主队让分赔率", value=1.90, format="%.2f")
    odds_away_team = st.sidebar.number_input("客队让分赔率", value=1.90, format="%.2f")

    # 模拟次数
    num_simulations = 700000

    if use_quarter_scores:
        # 四节比赛的模拟得分
        home_team_scores_q1 = np.random.normal(home_team_q1_avg_points_for, np.sqrt(np.abs(home_team_q1_avg_points_for - home_team_q1_avg_points_against)), num_simulations)
        home_team_scores_q2 = np.random.normal(home_team_q2_avg_points_for, np.sqrt(np.abs(home_team_q2_avg_points_for - home_team_q2_avg_points_against)), num_simulations)
        home_team_scores_q3 = np.random.normal(home_team_q3_avg_points_for, np.sqrt(np.abs(home_team_q3_avg_points_for - home_team_q3_avg_points_against)), num_simulations)
        home_team_scores_q4 = np.random.normal(home_team_q4_avg_points_for, np.sqrt(np.abs(home_team_q4_avg_points_for - home_team_q4_avg_points_against)), num_simulations)
        away_team_scores_q1 = np.random.normal(away_team_q1_avg_points_for, np.sqrt(np.abs(away_team_q1_avg_points_for - away_team_q1_avg_points_against)), num_simulations)
        away_team_scores_q2 = np.random.normal(away_team_q2_avg_points_for, np.sqrt(np.abs(away_team_q2_avg_points_for - away_team_q2_avg_points_against)), num_simulations)
        away_team_scores_q3 = np.random.normal(away_team_q3_avg_points_for, np.sqrt(np.abs(away_team_q3_avg_points_for - away_team_q3_avg_points_against)), num_simulations)
        away_team_scores_q4 = np.random.normal(away_team_q4_avg_points_for, np.sqrt(np.abs(away_team_q4_avg_points_for - away_team_q4_avg_points_against)), num_simulations)

        # 合并四节得分
        home_team_scores = home_team_scores_q1 + home_team_scores_q2 + home_team_scores_q3 + home_team_scores_q4
        away_team_scores = away_team_scores_q1 + away_team_scores_q2 + away_team_scores_q3 + away_team_scores_q4

        # 计算各节总得分
        total_scores_q1 = home_team_scores_q1 + away_team_scores_q1
        total_scores_q2 = home_team_scores_q2 + away_team_scores_q2
        total_scores_q3 = home_team_scores_q3 + away_team_scores_q3
        total_scores_q4 = home_team_scores_q4 + away_team_scores_q4
    else:
        # 使用整体平均得分和失分进行模拟
        home_team_scores = np.random.normal(home_team_avg_points_for, np.sqrt(np.abs(home_team_avg_points_for - home_team_avg_points_against)), num_simulations)
        away_team_scores = np.random.normal(away_team_avg_points_for, np.sqrt(np.abs(away_team_avg_points_for - away_team_avg_points_against)), num_simulations)
    
    total_scores = home_team_scores + away_team_scores

    # 计算胜负
    home_team_wins = np.sum(home_team_scores > away_team_scores)
    away_team_wins = np.sum(home_team_scores < away_team_scores)

    # 计算覆盖情况
    over_hits = np.sum(total_scores > over_under_line)
    under_hits = np.sum(total_scores < over_under_line)
    spread_hits_home_team = np.sum((home_team_scores - away_team_scores) > spread)
    spread_hits_away_team = np.sum((home_team_scores - away_team_scores) < spread)

    # 计算平均得分
    average_home_team_score = np.mean(home_team_scores)
    average_away_team_score = np.mean(away_team_scores)
    average_total_score = np.mean(total_scores)

    # 计算净得分差异
    average_score_diff = average_home_team_score - average_away_team_score

    # 计算投注回报率
    if spread_hits_home_team / num_simulations > 1 / odds_home_team:
        bet_home_roi = (spread_hits_home_team / num_simulations * odds_home_team - 1) * 100
    else:
        bet_home_roi = (spread_hits_home_team / num_simulations * odds_home_team - 1) * 100
    
    if spread_hits_away_team / num_simulations > 1 / odds_away_team:
        bet_away_roi = (spread_hits_away_team / num_simulations * odds_away_team - 1) * 100
    else:
        bet_away_roi = (spread_hits_away_team / num_simulations * odds_away_team - 1) * 100

    # 打印结果
    st.write(f"主队获胜概率: {home_team_wins / num_simulations * 100:.2f}%")
    st.write(f"客队获胜概率: {away_team_wins / num_simulations * 100:.2f}%")
    st.write(f"大于大小分的概率: {over_hits / num_simulations * 100:.2f}%")
    st.write(f"小于大小分的概率: {under_hits / num_simulations * 100:.2f}%")
    st.write(f"主队赢得让分的概率: {spread_hits_home_team / num_simulations * 100:.2f}%")
    st.write(f"客队赢得让分的概率: {spread_hits_away_team / num_simulations * 100:.2f}%")

    st.write(f"\n主队平均得分: {average_home_team_score:.2f}")
    st.write(f"客队平均得分: {average_away_team_score:.2f}")
    st.write(f"总得分平均值: {average_total_score:.2f}")
    st.write(f"主队和客队平均得分差异: {average_score_diff:.2f}")

    st.write(f"\n主队投注回报率: {bet_home_roi:.2f}%")
    st.write(f"客队投注回报率: {bet_away_roi:.2f}%")

    if use_quarter_scores:
        # 显示各节模拟概率最大的平均得分与失分的表格
        quarter_scores_df = pd.DataFrame({
            '节次': ['第一节', '第二节', '第三节', '第四节'],
            '主队得分': [np.mean(home_team_scores_q1), np.mean(home_team_scores_q2), np.mean(home_team_scores_q3), np.mean(home_team_scores_q4)],
            '客队得分': [np.mean(away_team_scores_q1), np.mean(away_team_scores_q2), np.mean(away_team_scores_q3), np.mean(away_team_scores_q4)],
            '总得分': [np.mean(total_scores_q1), np.mean(total_scores_q2), np.mean(total_scores_q3), np.mean(total_scores_q4)]
        })

        st.subheader("各节比分统计")
        st.write(quarter_scores_df)

        # 为每节得分进行直方图可视化
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # 第一节
        axs[0, 0].hist(total_scores_q1, bins=30, alpha=0.5, color='blue', label='第一节总得分')
        axs[0, 0].set_title('第一节总得分分布', fontproperties=font)
        axs[0, 0].set_xlabel('得分', fontproperties=font)
        axs[0, 0].set_ylabel('频率', fontproperties=font)
        axs[0, 0].legend(loc='upper right', prop=font)

        # 第二节
        axs[0, 1].hist(total_scores_q2, bins=30, alpha=0.5, color='green', label='第二节总得分')
        axs[0, 1].set_title('第二节总得分分布', fontproperties=font)
        axs[0, 1].set_xlabel('得分', fontproperties=font)
        axs[0, 1].set_ylabel('频率', fontproperties=font)
        axs[0, 1].legend(loc='upper right', prop=font)

        # 第三节
        axs[1, 0].hist(total_scores_q3, bins=30, alpha=0.5, color='red', label='第三节总得分')
        axs[1, 0].set_title('第三节总得分分布', fontproperties=font)
        axs[1, 0].set_xlabel('得分', fontproperties=font)
        axs[1, 0].set_ylabel('频率', fontproperties=font)
        axs[1, 0].legend(loc='upper right', prop=font)

        # 第四节
        axs[1, 1].hist(total_scores_q4, bins=30, alpha=0.5, color='purple', label='第四节总得分')
        axs[1, 1].set_title('第四节总得分分布', fontproperties=font)
        axs[1, 1].set_xlabel('得分', fontproperties=font)
        axs[1, 1].set_ylabel('频率', fontproperties=font)
        axs[1, 1].legend(loc='upper right', prop=font)

        plt.tight_layout()
        st.pyplot(fig)

    # 总得分的直方图
    fig, ax = plt.subplots()
    ax.hist(total_scores, bins=30, alpha=0.5, label='总得分')
    ax.axvline(x=over_under_line, color='r', linestyle='dashed', linewidth=2, label='大小分线')
    ax.axvline(x=average_total_score, color='g', linestyle='dashed', linewidth=2, label='总得分平均值')
    ax.set_xlabel('总得分', fontproperties=font)
    ax.set_ylabel('频率', fontproperties=font)
    ax.set_title('篮球比赛的蒙特卡洛模拟', fontproperties=font)
    ax.legend(loc='upper right', prop=font)
    st.pyplot(fig)

    # 可视化让分结果
    fig, ax = plt.subplots()
    score_diff = home_team_scores - away_team_scores
    ax.hist(score_diff, bins=30, alpha=0.5, label='得分差异 (主队 - 客队)')
    ax.axvline(x=spread, color='r', linestyle='dashed', linewidth=2, label='让分线')
    ax.axvline(x=average_score_diff, color='g', linestyle='dashed', linewidth=2, label='得分差异平均值')
    ax.set_xlabel('得分差异', fontproperties=font)
    ax.set_ylabel('频率', fontproperties=font)
    ax.set_title('让分分析', fontproperties=font)
    ax.legend(loc='upper right', prop=font)
    st.pyplot(fig)
