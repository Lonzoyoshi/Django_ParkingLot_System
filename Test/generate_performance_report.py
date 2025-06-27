import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import seaborn as sns


def generate_performance_report():
    """生成性能测试报告"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 读取测试结果
    with open('stability_test_report.json', 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 创建图表
    plt.figure(figsize=(15, 10))

    # 1. 性能指标概览
    plt.subplot(221)
    metrics = results['performance_metrics']
    plt.bar(['总请求数', '成功请求数', '失败请求数'],
            [metrics['total_requests'], metrics['successful_requests'], metrics['failed_requests']])
    plt.title('请求统计')
    for i, v in enumerate([metrics['total_requests'], metrics['successful_requests'], metrics['failed_requests']]):
        plt.text(i, v, str(v), ha='center', va='bottom')

    # 2. 响应时间分布
    plt.subplot(222)
    response_times = pd.Series([float(metrics['average_response_time'].replace('s', '')),
                                float(metrics['max_response_time'].replace('s', ''))])
    plt.bar(['平均响应时间', '最大响应时间'], response_times)
    plt.title('响应时间分析(秒)')
    for i, v in enumerate(response_times):
        plt.text(i, v, f'{v:.3f}s', ha='center', va='bottom')

    # 3. 成功率饼图
    plt.subplot(223)
    success_rate = float(metrics['success_rate'].replace('%', ''))
    plt.pie([success_rate, 100 - success_rate],
            labels=['成功', '失败'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%')
    plt.title('请求成功率')

    # 4. 性能趋势图（每秒请求数）
    plt.subplot(224)
    qps = float(metrics['requests_per_second'])
    plt.plot(['0s', '30s', '60s'], [0, qps, qps], marker='o')
    plt.title('每秒请求数(QPS)趋势')
    plt.grid(True)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig('performance_report.png', dpi=300, bbox_inches='tight')

    # 生成HTML报告
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>系统性能测试报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .metric {{ margin: 10px 0; padding: 10px; background-color: #f8f9fa; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>系统性能测试报告</h1>

            <h2>测试概要</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>开始时间</td><td>{results['test_summary']['start_time']}</td></tr>
                <tr><td>结束时间</td><td>{results['test_summary']['end_time']}</td></tr>
                <tr><td>持续时间</td><td>{results['test_summary']['duration']}秒</td></tr>
                <tr><td>并发用户数</td><td>{results['test_summary']['concurrent_users']}</td></tr>
            </table>

            <h2>性能指标</h2>
            <table>
                <tr><th>指标</th><th>数值</th></tr>
                <tr><td>总请求数</td><td>{metrics['total_requests']}</td></tr>
                <tr><td>成功请求数</td><td>{metrics['successful_requests']}</td></tr>
                <tr><td>失败请求数</td><td>{metrics['failed_requests']}</td></tr>
                <tr><td>成功率</td><td>{metrics['success_rate']}</td></tr>
                <tr><td>平均响应时间</td><td>{metrics['average_response_time']}</td></tr>
                <tr><td>最大响应时间</td><td>{metrics['max_response_time']}</td></tr>
                <tr><td>每秒请求数</td><td>{metrics['requests_per_second']}</td></tr>
            </table>

            <h2>性能分析图表</h2>
            <img src="performance_report.png" alt="性能分析图表" style="width: 100%;">
        </div>
    </body>
    </html>
    """

    # 保存HTML报告
    with open('performance_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    generate_performance_report()