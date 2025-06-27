import requests
import unittest
from bs4 import BeautifulSoup
import json
import random
from datetime import datetime
from datetime import timedelta


class TestParkingUpdate(unittest.TestCase):
    BASE_URL = 'http://127.0.0.1:8000'

    def setUp(self):
        self.session = requests.Session()
        self.login()

    def login(self):
        """管理员登录"""
        try:
            # 1. 访问登录页面获取CSRF token
            r = self.session.get(f'{self.BASE_URL}/parking/login/')
            self.assertTrue(r.ok, "访问登录页面失败")

            # 2. 解析页面内容
            soup = BeautifulSoup(r.text, 'html.parser')

            # 3. 找到管理员登录表单（使用正确的选择器）
            admin_form = soup.select_one('#admin-login form')
            if not admin_form:
                print("未找到管理员登录表单")
                print(f"页面内容: {r.text[:500]}")  # 打印页面前500个字符用于调试
                self.fail("未找到管理员登录表单")

            # 4. 获取CSRF token
            csrf_token = admin_form.find('input', {'name': 'csrfmiddlewaretoken'})
            if not csrf_token:
                print("未找到CSRF token")
                self.fail("未找到CSRF token")

            # 5. 设置登录数据
            login_data = {
                'username': 'admin',
                'password': 'hangzi0.',
                'csrfmiddlewaretoken': csrf_token['value']
            }

            # 6. 从表单获取实际的提交URL
            action_url = admin_form.get('action')
            login_url = f"{self.BASE_URL}{action_url}"

            # 7. 发送登录请求
            r = self.session.post(
                login_url,
                data=login_data,
                headers={
                    'Referer': f'{self.BASE_URL}/parking/login/',
                }
            )

            self.assertTrue(r.ok, "登录请求失败")

            # 8. 验证登录状态
            r = self.session.get(f'{self.BASE_URL}/parking/manage/')
            self.assertTrue(r.ok, "登录验证失败")

        except Exception as e:
            print(f"登录过程异常: {str(e)}")
            raise

    def generate_test_data(self, num_records=10):
        """生成测试数据"""
        plates = ['京A' + str(random.randint(10000, 99999)) for _ in range(num_records)]
        current_time = datetime.now()
        test_records = []

        for i, plate in enumerate(plates):
            # 生成随机的入场时间（过去24小时内）
            entry_time = current_time - timedelta(hours=random.uniform(0, 24))

            # 部分车辆已经离场，部分仍在停车场
            if random.random() < 0.7:  # 70%的车辆已离场
                exit_time = entry_time + timedelta(hours=random.uniform(1, 5))
                fee = random.uniform(10, 50)
                status = "已离场"
            else:
                exit_time = None
                fee = 0
                status = "在场"

            record = {
                'plate_no': plate,
                'entry_time': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'exit_time': exit_time.strftime('%Y-%m-%d %H:%M:%S') if exit_time else None,
                'fee': round(fee, 2),
                'status': status
            }
            test_records.append(record)

        return test_records

    def insert_test_records(self, records):
        """插入测试记录"""
        for record in records:
            try:
                r = self.session.post(
                    f'{self.BASE_URL}/parking/update_record/',
                    json=record,
                    headers={
                        'X-CSRFToken': self.session.cookies.get('csrftoken'),
                        'Content-Type': 'application/json'
                    }
                )
                self.assertTrue(r.ok, f"插入记录失败: {record['plate_no']}")
                print(f"成功插入记录: {record['plate_no']}")
            except Exception as e:
                print(f"插入记录时出错: {str(e)}")

    def test_parking_data_generation(self):
        """测试数据生成和灌入"""
        # 生成测试数据
        test_records = self.generate_test_data(num_records=20)

        # 插入测试数据
        self.insert_test_records(test_records)

        # 验证数据插入是否成功
        r = self.session.get(f'{self.BASE_URL}/parking/get_records/')
        self.assertTrue(r.ok)
        data = r.json()
        self.assertIn('records', data)

        # 验证统计数据是否更新
        r = self.session.get(f'{self.BASE_URL}/parking/get_stats/')
        self.assertTrue(r.ok)
        stats = r.json()

        # 保存测试结果
        test_results = {
            'parking_stats': stats,
            'parking_records': data.get('records', [])
        }

        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print("测试数据已保存到 test_results.json")

    def simulate_parking_updates(self):
        """模拟停车场实时更新"""
        # 模拟新车入场
        new_car = {
            'plate_no': '京A' + str(random.randint(10000, 99999)),
            'entry_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': None,
            'fee': 0,
            'status': '在场'
        }
        self.insert_test_records([new_car])
        print(f"新车入场: {new_car['plate_no']}")

        # 模拟车辆离场
        r = self.session.get(f'{self.BASE_URL}/parking/get_records/')
        if r.ok:
            records = r.json().get('records', [])
            parked_cars = [r for r in records if r['status'] == '在场']
            if parked_cars:
                car_to_exit = random.choice(parked_cars)
                car_to_exit['exit_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                car_to_exit['fee'] = round(random.uniform(10, 50), 2)
                car_to_exit['status'] = '已离场'
                self.insert_test_records([car_to_exit])
                print(f"车辆离场: {car_to_exit['plate_no']}")

    def test_parking_stats(self):
        """测试停车场统计数据"""
        r = self.session.get(f'{self.BASE_URL}/parking/get_stats/')
        self.assertTrue(r.ok)
        stats = r.json()

        required_fields = ['today_income', 'today_cars', 'parked_count', 'available_spaces']
        for field in required_fields:
            self.assertIn(field, stats)

        # 获取停车记录数据
        records_response = self.session.get(f'{self.BASE_URL}/parking/get_records/')
        self.assertTrue(records_response.ok)
        records_data = records_response.json()

        # 保存测试结果
        test_results = {
            'parking_stats': stats,
            'parking_records': records_data.get('records', [])
        }

        # 保存到文件
        try:
            with open('test_results.json', 'w', encoding='utf-8') as f:
                json.dump(test_results, f, ensure_ascii=False, indent=2)
            print("测试结果已保存到 test_results.json")
        except Exception as e:
            print(f"保存测试结果时出错: {str(e)}")

    def test_parking_record(self):
        """测试停车记录管理"""
        r = self.session.get(f'{self.BASE_URL}/parking/get_records/')
        self.assertTrue(r.ok)
        data = r.json()
        self.assertIn('records', data)

    def tearDown(self):
        if hasattr(self, 'session'):
            try:
                self.session.get(f'{self.BASE_URL}/parking/logout/')
            finally:
                self.session.close()


if __name__ == '__main__':
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParkingUpdate)
    # 运行测试
    unittest.TextTestRunner(verbosity=2).run(suite)