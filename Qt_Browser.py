import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWebEngineWidgets import *

class PlateRecognitionBrowser(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle('车牌识别系统浏览器')
        self.setGeometry(100, 100, 1280, 768)

        # 创建工具栏
        navbar = QToolBar()
        self.addToolBar(navbar)

        # 后退按钮
        back_btn = QAction('后退', self)
        back_btn.setStatusTip('返回上一页')
        back_btn.triggered.connect(lambda: self.browser.back())
        navbar.addAction(back_btn)

        # 前进按钮
        forward_btn = QAction('前进', self)
        forward_btn.setStatusTip('前进到下一页')
        forward_btn.triggered.connect(lambda: self.browser.forward())
        navbar.addAction(forward_btn)

        # 刷新按钮
        reload_btn = QAction('刷新', self)
        reload_btn.setStatusTip('刷新当前页面')
        reload_btn.triggered.connect(lambda: self.browser.reload())
        navbar.addAction(reload_btn)

        # 地址栏
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.navigate_to_url)
        navbar.addWidget(self.url_bar)

        # 创建浏览器窗口
        self.browser = QWebEngineView()
        self.browser.setUrl(QUrl('http://127.0.0.1:8000'))
        self.setCentralWidget(self.browser)

        # 更新URL显示
        self.browser.urlChanged.connect(self.update_url)

        # 显示状态栏
        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # 显示窗口
        self.show()

    def navigate_to_url(self):
        """导航到输入的URL"""
        url = self.url_bar.text()
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        self.browser.setUrl(QUrl(url))

    def update_url(self, q):
        """更新地址栏显示的URL"""
        self.url_bar.setText(q.toString())
        self.url_bar.setCursorPosition(0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置应用程序名称
    app.setApplicationName('车牌识别系统浏览器')
    # 创建浏览器窗口
    window = PlateRecognitionBrowser()
    # 运行应用
    sys.exit(app.exec_())