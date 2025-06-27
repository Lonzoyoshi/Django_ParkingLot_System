import unittest
import requests
import json
import time
from bs4 import BeautifulSoup
from datetime import datetime

class TestSystemOperation(unittest.TestCase):
    BASE_URL = 'http://127.0.0.1:8000'

    def setUp(self):
        """初始化测试环境"""
        self.session = requests.Session()
        self.results = {
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_tests': [],
            'output_tests': [],
            'analysis': {}
        }
        if not self.login():
            raise Exception("登录失败")

    def login(self):
        """管理员登录"""
        try:
            # 1. 访问登录页面获取CSRF token
            r = self.session.get(f'{self.BASE_URL}/parking/login/')
            soup = BeautifulSoup(r.text, 'html.parser')
            csrf_token = soup.select_one('#admin-login form input[name="csrfmiddlewaretoken"]')

            if not csrf_token:
                print("未找到CSRF token")
                return False

            # 2. 设置登录数据
            login_data = {
                'username': 'admin',
                'password': 'hangzi0.',
                'csrfmiddlewaretoken': csrf_token['value']
            }

            # 3. 发送登录请求
            r = self.session.post(
                f'{self.BASE_URL}/parking/admin/login/',
                data=login_data,
                headers={
                    'Referer': f'{self.BASE_URL}/parking/login/',
                }
            )

            if not r.ok:
                print(f"登录请求失败: {r.status_code}")
                return False

            # 4. 验证登录状态
            r = self.session.get(f'{self.BASE_URL}/parking/manage/')
            return r.ok

        except Exception as e:
            print(f"登录过程发生异常: {str(e)}")
            return False

    def test_data_input(self):
        """测试数据输入操作"""
        # 1. 测试车辆入场记录
        entry_data = {
            'plate_no': '京A12345',
            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': '在场'
        }

        r = self.session.post(
            f'{self.BASE_URL}/parking/update_record/',
            json=entry_data,
            headers={
                'X-CSRFToken': self.session.cookies.get('csrftoken'),
                'Content-Type': 'application/json'
            }
        )

        self.results['input_tests'].append({
            'test_name': '车辆入场记录',
            'input_data': entry_data,
            'result': r.ok,
            'response': r.json() if r.ok else None
        })

        # 2. 测试车辆出场记录
        exit_data = {
            'plate_no': '京A12345',
            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fee': 25.00,
            'status': '已离场'
        }

        r = self.session.post(
            f'{self.BASE_URL}/parking/update_record/',
            json=exit_data,
            headers={
                'X-CSRFToken': self.session.cookies.get('csrftoken'),
                'Content-Type': 'application/json'
            }
        )

        self.results['input_tests'].append({
            'test_name': '车辆出场记录',
            'input_data': exit_data,
            'result': r.ok,
            'response': r.json() if r.ok else None
        })

    def test_data_output(self):
        """测试数据输出和可视化效果"""
        # 1. 测试停车记录列表
        r = self.session.get(f'{self.BASE_URL}/parking/get_records/')
        self.results['output_tests'].append({
            'test_name': '停车记录列表',
            'result': r.ok,
            'data_count': len(r.json()['records']) if r.ok else 0
        })

        # 2. 测试统计数据
        r = self.session.get(f'{self.BASE_URL}/parking/get_stats/')
        self.results['output_tests'].append({
            'test_name': '统计数据显示',
            'result': r.ok,
            'stats': r.json() if r.ok else None
        })

    def analyze_results(self):
        """分析测试结果"""
        # 计算成功率
        input_success = sum(1 for test in self.results['input_tests'] if test['result'])
        output_success = sum(1 for test in self.results['output_tests'] if test['result'])

        total_tests = len(self.results['input_tests']) + len(self.results['output_tests'])
        success_rate = ((input_success + output_success) / total_tests * 100) if total_tests > 0 else 0

        self.results['analysis'] = {
            'total_tests': total_tests,
            'success_rate': f"{success_rate:.2f}%",
            'input_success_rate': f"{(input_success / len(self.results['input_tests']) * 100):.2f}%" if self.results['input_tests'] else "0%",
            'output_success_rate': f"{(output_success / len(self.results['output_tests']) * 100):.2f}%" if self.results['output_tests'] else "0%"
        }

    def generate_report(self):
        """生成测试报告"""
        self.results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

        report = {
            'summary': {
                'start_time': self.results['start_time'],
                'end_time': self.results['end_time'],
                'success_rate': self.results['analysis']['success_rate'],
                'total_tests': self.results['analysis']['total_tests']
            },
            'input_operations': self.results['input_tests'],
            'output_operations': self.results['output_tests'],
            'analysis': self.results['analysis']
        }

        # 保存报告
        with open('system_operation_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print("\n系统操作演示测试报告:")
        print(f"开始时间: {report['summary']['start_time']}")
        print(f"结束时间: {report['summary']['end_time']}")
        print(f"测试成功率: {report['summary']['success_rate']}")
        print("\n详细报告已保存至 system_operation_report.json")

    def tearDown(self):
        """清理测试环境"""
        if hasattr(self, 'session'):
            self.session.close()

if __name__ == '__main__':
    # 创建测试实例
    test = TestSystemOperation()

    print("开始系统操作演示测试...\n")

    try:
        # 运行数据输入测试
        print("1. 测试数据输入操作...")
        test.test_data_input()

        # 运行数据输出测试
        print("2. 测试数据输出操作...")
        test.test_data_output()

        # 分析结果
        print("3. 分析测试结果...")
        test.analyze_results()

        # 生成报告
        test.generate_report()

    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
    finally:
        test.tearDown()