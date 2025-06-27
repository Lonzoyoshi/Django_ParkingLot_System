from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import os


class TestSystemScreenshots:
    def __init__(self):
        # 创建Chrome浏览器实例
        self.driver = webdriver.Chrome()
        self.driver.maximize_window()
        self.BASE_URL = 'http://127.0.0.1:8000'
        # 创建screenshots文件夹
        self.screenshot_dir = 'screenshots'
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)

    def login(self):
        """管理员登录"""
        try:
            self.driver.get(f'{self.BASE_URL}/parking/login/')

            # 等待登录表单加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "admin-login"))
            )

            # 切换到管理员登录tab
            admin_tab = self.driver.find_element(By.CSS_SELECTOR, 'a[href="#admin-login"]')
            admin_tab.click()

            # 获取登录表单元素
            username = self.driver.find_element(By.NAME, 'username')
            password = self.driver.find_element(By.NAME, 'password')
            submit = self.driver.find_element(By.CSS_SELECTOR, '#admin-login button[type="submit"]')

            # 输入登录信息
            username.send_keys('admin')
            password.send_keys('hangzi0.')

            # 保存登录页面截图
            self.take_screenshot('1_login_page')

            # 提交登录表单
            submit.click()

            # 等待跳转完成
            WebDriverWait(self.driver, 10).until(
                EC.url_contains('/parking/manage/')
            )

            return True
        except Exception as e:
            print(f"登录失败: {str(e)}")
            return False

    def capture_dashboard(self):
        """捕获仪表盘页面"""
        try:
            # 等待仪表盘加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "dashboard-section"))
            )
            time.sleep(2)  # 等待数据加载
            self.take_screenshot('2_dashboard')
        except Exception as e:
            print(f"仪表盘截图失败: {str(e)}")

    def capture_camera_view(self):
        """捕获摄像头监控页面"""
        try:
            # 点击摄像头监控标签
            camera_tab = self.driver.find_element(By.CSS_SELECTOR, 'a[href="#camera-section"]')
            camera_tab.click()

            # 等待摄像头视图加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "camera-section"))
            )
            time.sleep(2)
            self.take_screenshot('3_camera_view')
        except Exception as e:
            print(f"摄像头视图截图失败: {str(e)}")

    def capture_records(self):
        """捕获停车记录页面"""
        try:
            # 点击停车记录标签
            records_tab = self.driver.find_element(By.CSS_SELECTOR, 'a[href="#records-section"]')
            records_tab.click()

            # 等待记录表格加载
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "records-table"))
            )
            time.sleep(2)
            self.take_screenshot('4_parking_records')
        except Exception as e:
            print(f"停车记录截图失败: {str(e)}")

    def take_screenshot(self, name):
        """保存截图"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'{self.screenshot_dir}/{name}_{timestamp}.png'
        self.driver.save_screenshot(filename)
        print(f"截图已保存: {filename}")

    def run_tests(self):
        """运行所有测试并捕获截图"""
        try:
            print("开始系统界面截图...")

            # 登录并截图
            if self.login():
                print("登录成功，开始捕获各个界面...")

                # 捕获仪表盘
                print("正在捕获仪表盘...")
                self.capture_dashboard()

                # 捕获摄像头监控
                print("正在捕获摄像头监控...")
                self.capture_camera_view()

                # 捕获停车记录
                print("正在捕获停车记录...")
                self.capture_records()

                print("\n所有截图已完成！")
                print(f"截图保存在 {self.screenshot_dir} 目录下")
            else:
                print("登录失败，无法继续测试")
        except Exception as e:
            print(f"测试过程出错: {str(e)}")
        finally:
            self.driver.quit()


if __name__ == '__main__':
    # 创建测试实例并运行
    screenshot_test = TestSystemScreenshots()
    screenshot_test.run_tests()