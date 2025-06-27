import unittest
import requests
import json
import time
from bs4 import BeautifulSoup


class TestSecurity(unittest.TestCase):
    BASE_URL = 'http://127.0.0.1:8000'

    def setUp(self):
        self.session = requests.Session()
        self.results = {
            'tests': {
                'authentication': [],
                'authorization': [],
                'input_validation': [],
                'session_security': [],
                'data_protection': []
            },
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def test_authentication_security(self):
        """身份认证安全测试"""
        # 测试密码策略
        weak_passwords = ['123456', 'password', 'admin123']
        for pwd in weak_passwords:
            r = self.login_attempt('admin', pwd)
            self.results['tests']['authentication'].append({
                'test_name': '弱密码防护',
                'input': pwd,
                'result': r.status_code != 200,
                'details': '系统拒绝弱密码登录'
            })

        # 测试账户锁定
        for _ in range(5):
            r = self.login_attempt('admin', 'wrong_pwd')
        r = self.login_attempt('admin', 'wrong_pwd')
        self.results['tests']['authentication'].append({
            'test_name': '账户锁定机制',
            'result': r.status_code == 403,
            'details': '多次失败后账户锁定'
        })

    def test_authorization_security(self):
        """授权访问安全测试"""
        protected_endpoints = [
            '/parking/manage/',
            '/parking/update_record/',
            '/parking/delete_record/1/'
        ]

        # 未登录访问测试
        for endpoint in protected_endpoints:
            r = requests.get(f'{self.BASE_URL}{endpoint}')
            self.results['tests']['authorization'].append({
                'test_name': f'未授权访问 {endpoint}',
                'result': r.status_code in [302, 403],
                'details': '未登录重定向到登录页面'
            })

        # 越权访问测试
        self.login_as_normal_user()
        for endpoint in protected_endpoints:
            r = self.session.get(f'{self.BASE_URL}{endpoint}')
            self.results['tests']['authorization'].append({
                'test_name': f'越权访问 {endpoint}',
                'result': r.status_code == 403,
                'details': '普通用户无法访问管理功能'
            })

    def test_input_validation(self):
        """输入验证安全测试"""
        # XSS测试
        xss_payloads = [
            '<script>alert(1)</script>',
            'javascript:alert(1)'
        ]

        for payload in xss_payloads:
            data = {'plate_no': payload}
            r = self.session.post(f'{self.BASE_URL}/parking/update_record/', json=data)
            self.results['tests']['input_validation'].append({
                'test_name': 'XSS防护',
                'input': payload,
                'result': payload not in r.text,
                'details': '系统过滤XSS攻击代码'
            })

        # SQL注入测试
        sql_payloads = [
            "' OR '1'='1",
            "admin'--"
        ]

        for payload in sql_payloads:
            r = self.login_attempt(payload, payload)
            self.results['tests']['input_validation'].append({
                'test_name': 'SQL注入防护',
                'input': payload,
                'result': r.status_code != 200,
                'details': '系统防止SQL注入攻击'
            })

    def login_attempt(self, username, password):
        """尝试登录"""
        try:
            # 获取登录页面和CSRF token
            r = self.session.get(f'{self.BASE_URL}/parking/login/')
            soup = BeautifulSoup(r.text, 'html.parser')
            csrf_token = soup.select_one('#admin-login form input[name="csrfmiddlewaretoken"]')

            if not csrf_token:
                return requests.Response()  # 返回一个空响应表示失败

            # 发送登录请求
            login_data = {
                'username': username,
                'password': password,
                'csrfmiddlewaretoken': csrf_token['value']
            }

            return self.session.post(
                f'{self.BASE_URL}/parking/admin/login/',
                data=login_data,
                headers={'Referer': f'{self.BASE_URL}/parking/login/'}
            )
        except Exception as e:
            print(f"登录尝试失败: {str(e)}")
            response = requests.Response()
            response.status_code = 500
            return response

    def login_as_normal_user(self):
        """以普通用户身份登录"""
        try:
            # 获取登录页面和CSRF token
            r = self.session.get(f'{self.BASE_URL}/parking/login/')
            soup = BeautifulSoup(r.text, 'html.parser')
            csrf_token = soup.select_one('#user-login form input[name="csrfmiddlewaretoken"]')

            if not csrf_token:
                return False

            # 发送登录请求
            login_data = {
                'username': 'normal_user',  # 使用普通用户账号
                'password': 'normal123',  # 使用普通用户密码
                'csrfmiddlewaretoken': csrf_token['value']
            }

            r = self.session.post(
                f'{self.BASE_URL}/parking/customer/login/',
                data=login_data,
                headers={'Referer': f'{self.BASE_URL}/parking/login/'}
            )

            return r.ok
        except Exception as e:
            print(f"普通用户登录失败: {str(e)}")
            return False

    def calculate_security_score(self):
        """计算安全得分"""
        total_tests = 0
        passed_tests = 0

        for category in self.results['tests'].values():
            for test in category:
                total_tests += 1
                if test['result']:
                    passed_tests += 1

        return (passed_tests / total_tests * 100) if total_tests > 0 else 0

    def generate_security_report(self):
        """生成安全测试报告"""
        self.results['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.results['security_score'] = self.calculate_security_score()

        report = {
            'summary': {
                'start_time': self.results['start_time'],
                'end_time': self.results['end_time'],
                'security_score': f"{self.results['security_score']:.2f}%"
            },
            'test_results': self.results['tests'],
            'recommendations': self.generate_recommendations()
        }

        # 保存JSON报告
        with open('security_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"安全测试得分: {report['summary']['security_score']}")
        print("完整报告已保存到 security_test_report.json")

    def generate_recommendations(self):
        """生成安全建议"""
        return {
            'authentication': [
                '实施强密码策略',
                '添加双因素认证',
                '实现账户锁定机制'
            ],
            'authorization': [
                '细化权限控制',
                '实现角色基础访问控制(RBAC)',
                '添加操作审计日志'
            ],
            'input_validation': [
                '使用参数化查询防止SQL注入',
                'HTML转义防止XSS攻击',
                '验证所有用户输入'
            ],
            'session_security': [
                '使用安全的会话ID',
                '设置会话超时',
                '防止会话固定攻击'
            ]
        }


if __name__ == '__main__':
    test = TestSecurity()
    test.setUp()
    test.test_authentication_security()
    test.test_authorization_security()
    test.test_input_validation()
    test.generate_security_report()
