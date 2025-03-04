from time import sleep
from lxml import html
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
'''
爬取电影code
类似于https://movie.douban.com/subject/34858077/中的34858077
'''
def get_films_code():
    edge_options = Options()
    edge_options.add_argument("--headless")
    edge_options.add_argument("--disable-gpu")
    edge_options.add_argument("--no-sandbox")

    driver = webdriver.Edge(service=Service(), options=edge_options)

    try:
        # 打开目标网页
        driver.get('https://movie.douban.com/explore')
        sleep(2)
        # 等待并点击“排序”按钮
        try:
            sort_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '排序')]"))
            )
            ActionChains(driver).move_to_element(sort_button).click().perform()
            sleep(0.9)  # 等待排序下拉菜单出现
        except Exception as e:
            print(e)

        # 等待并点击“近期热度”按钮
        try:
            recent_heat_button = WebDriverWait(driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '近期热度')]"))
            )
            ActionChains(driver).move_to_element(recent_heat_button).click().perform()
            sleep(1)  # 确保点击后让页面有时间加载
        except Exception as e:
            print(e)

        # 模拟点击“加载更多”按钮
        for i in range(25):
            print('Round:', i + 1)

            try:
                # 等待“加载更多”按钮可被点击
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (By.CSS_SELECTOR, 'button.drc-button.button.l.default.default.blue.secondary.block'))
                )
                # 点击“加载更多”按钮
                ActionChains(driver).move_to_element(load_more_button).click().perform()
                sleep(1.2)  # 等待1秒以允许页面加载新内容
            except Exception as e:
                print(e)
                break  # 如果遇到问题，退出循环

        # 等待数据加载
        sleep(1)
        urls = []
        # 获取更新后的 HTML 源内容
        html_ = driver.page_source
        tree = html.fromstring(html_)
        for i in tree.xpath('//a/@href'):
            if '/movie/' in i:
                urls.append(i)
        with open('urls.txt', 'w') as f:
            for url in urls:
                f.write(url + '\n')
    finally:
        # 关闭浏览器
        driver.quit()
        return urls
