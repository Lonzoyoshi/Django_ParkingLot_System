from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

def generate_report():
    """生成测试报告白皮书"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 1. 获取测试数据
    with open('test_results.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 2. 创建统计图表
    plt.figure(figsize=(12, 6))

    # 停车场统计柱状图
    stats = results['parking_stats']
    plt.subplot(121)
    plt.bar(['今日收入', '车流量', '当前停车数', '剩余车位'],
            [stats['today_income'], stats['today_cars'],
             stats['parked_count'], stats['available_spaces']])
    plt.title('停车场实时统计')

    # 停车记录趋势图
    records = pd.DataFrame(results['parking_records'])
    plt.subplot(122)
    records['entry_time'] = pd.to_datetime(records['entry_time'])
    records.set_index('entry_time')['fee'].plot(kind='line')
    plt.title('停车费用趋势')

    # 保存图表
    plt.savefig('parking_analysis.png')

    # 3. 生成HTML报告
    html_content = f"""
    <h1>停车场系统测试报告</h1>
    <h2>测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h2>

    <h3>1. 系统状态概览</h3>
    <table border="1">
        <tr><th>指标</th><th>数值</th></tr>
        <tr><td>今日收入</td><td>{stats['today_income']}元</td></tr>
        <tr><td>今日车流量</td><td>{stats['today_cars']}辆</td></tr>
        <tr><td>当前停车数</td><td>{stats['parked_count']}辆</td></tr>
        <tr><td>剩余车位</td><td>{stats['available_spaces']}个</td></tr>
    </table>

    <h3>2. 数据分析图表</h3>
    <img src="parking_analysis.png" alt="停车场数据分析">

    <h3>3. 停车记录明细</h3>
    {records.to_html()}
    """

    with open('parking_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    generate_report()