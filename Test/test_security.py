import unittest
import requests
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from bs4 import BeautifulSoup


class TestSystemStability(unittest.TestCase):
    BASE_URL = 'http://127.0.0.1:8000'
    CONCURRENT_USERS = 50  # 并发用户数
    TEST_DURATION = 60  # 测试持续时间(秒)

    def setUp(self):
        self.session = requests.Session()
        self.login()
        self.results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': [],
            'start_time': None,
            'end_time': None
        }

    def login(self):
        """管理员登录"""
        try:
            r = self.session.get(f'{self.BASE_URL}/parking/login/')
            soup = BeautifulSoup(r.text, 'html.parser')
            csrf_token = soup.select_one('#admin-login form input[name="csrfmiddlewaretoken"]')['value']

            r = self.session.post(
                f'{self.BASE_URL}/parking/admin/login/',
                data={
                    'username': 'admin',
                    'password': 'hangzi0.',
                    'csrfmiddlewaretoken': csrf_token
                },
                headers={'Referer': f'{self.BASE_URL}/parking/login/'}
            )
            self.assertTrue(r.ok)
        except Exception as e:
            print(f"登录失败: {str(e)}")
            raise

    def simulate_user_actions(self):
        """模拟用户操作"""
        endpoints = [
            '/parking/get_stats/',
            '/parking/get_records/',
            '/parking/manage/'
        ]

        try:
            for endpoint in endpoints:
                start_time = time.time()
                r = self.session.get(f'{self.BASE_URL}{endpoint}')
                response_time = time.time() - start_time

                with threading.Lock():
                    self.results['total_requests'] += 1
                    self.results['response_times'].append(response_time)

                    if r.ok:
                        self.results['successful_requests'] += 1
                    else:
                        self.results['failed_requests'] += 1
                        self.results['errors'].append({
                            'endpoint': endpoint,
                            'status_code': r.status_code,
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'error': r.text[:200]
                        })

        except Exception as e:
            with threading.Lock():
                self.results['failed_requests'] += 1
                self.results['errors'].append({
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })

    def test_system_stability(self):
        """系统稳定性测试"""
        self.results['start_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.CONCURRENT_USERS) as executor:
            # 计算需要执行的次数
            iterations = int((self.TEST_DURATION * self.CONCURRENT_USERS) / 3)  # 每个用户平均3秒一次操作
            # 提交任务
            futures = [executor.submit(self.simulate_user_actions)
                       for _ in range(iterations)]

            # 等待所有任务完成
            for future in futures:
                future.result()

        self.results['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 生成测试报告
        self.generate_report()

        # 验证测试结果
        self.assertGreater(self.results['successful_requests'], 0)
        self.assertLess(
            self.results['failed_requests'] / self.results['total_requests'],
            0.05  # 允许5%的失败率
        )

    def generate_report(self):
        """生成测试报告"""
        if self.results['response_times']:
            avg_response_time = sum(self.results['response_times']) / len(self.results['response_times'])
            max_response_time = max(self.results['response_times'])
        else:
            avg_response_time = max_response_time = 0

        report = {
            'test_summary': {
                'start_time': self.results['start_time'],
                'end_time': self.results['end_time'],
                'duration': self.TEST_DURATION,
                'concurrent_users': self.CONCURRENT_USERS
            },
            'performance_metrics': {
                'total_requests': self.results['total_requests'],
                'successful_requests': self.results['successful_requests'],
                'failed_requests': self.results['failed_requests'],
                'success_rate': f"{(self.results['successful_requests'] / self.results['total_requests'] * 100):.2f}%",
                'average_response_time': f"{avg_response_time:.3f}s",
                'max_response_time': f"{max_response_time:.3f}s",
                'requests_per_second': f"{self.results['total_requests'] / self.TEST_DURATION:.2f}"
            },
            'errors': self.results['errors']
        }

        # 保存JSON报告
        with open('stability_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 生成HTML报告
        html_report = f"""
        <html>
        <head>
            <title>系统稳定性测试报告</title>
            <style>
                body {{ font-family: Arial; margin: 20px; }}
                .metric {{ margin: 10px 0; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>系统稳定性测试报告</h1>
            <h2>测试概要</h2>
            <div class="metric">开始时间: {report['test_summary']['start_time']}</div>
            <div class="metric">结束时间: {report['test_summary']['end_time']}</div>
            <div class="metric">持续时间: {report['test_summary']['duration']}秒</div>
            <div class="metric">并发用户数: {report['test_summary']['concurrent_users']}</div>

            <h2>性能指标</h2>
            <div class="metric">总请求数: {report['performance_metrics']['total_requests']}</div>
            <div class="metric">成功请求数: {report['performance_metrics']['successful_requests']}</div>
            <div class="metric">失败请求数: {report['performance_metrics']['failed_requests']}</div>
            <div class="metric">成功率: {report['performance_metrics']['success_rate']}</div>
            <div class="metric">平均响应时间: {report['performance_metrics']['average_response_time']}</div>
            <div class="metric">最大响应时间: {report['performance_metrics']['max_response_time']}</div>
            <div class="metric">每秒请求数: {report['performance_metrics']['requests_per_second']}</div>

            <h2>错误记录</h2>
            {''.join(f'<div class="error">{error}</div>' for error in report['errors'])}
        </body>
        </html>
        """

        with open('stability_test_report.html', 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"测试报告已生成：stability_test_report.json 和 stability_test_report.html")


if __name__ == '__main__':
    unittest.main(verbosity=2)