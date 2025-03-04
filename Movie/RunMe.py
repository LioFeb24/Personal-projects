import csv
import json
import os
from datetime import datetime
from time import sleep

from lxml import html
from selenium import webdriver
from selenium.webdriver.edge.options import Options

from get_films_code import get_films_code


def get_films_datas(url) -> list:
    '''
    爬取电影信息，若某项信息不存在则使用 None 替代
    :param url: 所爬取的电影的豆瓣url
    :return: list:[title, country, director, main_actor, actor, type, films_list, date, long, want_watch, watched, p]
    title 标题
    country 地区
    director 导演
    main_actor 主演
    actor 演员
    type 电影类型
    films_list 所属电影榜单
    date 初映日期
    long 电影时长
    want_watch 想看人数
    watched 观看了的人数
    p 电影评分
    '''
    edge_options = Options()
    edge_options.add_argument("--headless")  # 启用无头模式
    driver = webdriver.Edge(options=edge_options)

    try:
        driver.get(url)
        sleep(1)  # 给 JavaScript 充分加载时间
        page_source = driver.page_source
        tree = html.fromstring(page_source)

        # 获取数据，若不存在则使用 None
        def safe_xpath(xpath_expr, is_list=False, transform=str):
            try:
                result = tree.xpath(xpath_expr)
                if is_list:
                    return result if result else None
                return transform(result[0]) if result else None
            except Exception:
                return None

        title = safe_xpath('//meta[@property="og:title"]/@content')
        main_actor = safe_xpath('//a[@rel="v:starring"]/text()')
        actor = safe_xpath('//meta[@property="video:actor"]/@content', is_list=True)
        director = safe_xpath('//a[@rel="v:directedBy"]/text()')
        type = safe_xpath('//span[@property="v:genre"]/text()', is_list=True)

        date_str = safe_xpath('//span[@property="v:initialReleaseDate"]/text()')
        if date_str:
            date_str = ''.join(filter(lambda x: x.isdigit() or x in '-', date_str))
            if len(date_str) == 7:
                date_str += '01'
            date = datetime.strptime(date_str, '%Y-%m-%d').timestamp()
        else:
            date = None

        long = safe_xpath('//span[@property="v:runtime"]/@content', transform=int)

        # 解析 JSON 评分
        rating_json = safe_xpath('//script[@type="application/ld+json"]/text()')
        if rating_json:
            try:
                rating_data = json.loads(rating_json.replace('\n', '').replace(' ', ''))
                p = float(rating_data.get('aggregateRating', {}).get('ratingValue', None))
            except json.JSONDecodeError:
                p = None
        else:
            p = None

        country = safe_xpath("//div[@id='info']/span[contains(., '制片国家/地区:')]/following-sibling::text()[1]",
                             transform=lambda x: x.strip() if x else None)
        films_list = safe_xpath("//div[@id='subject-doulist']//a/text()", is_list=True)[1:]

        # 获取想看 & 已看人数
        watch = safe_xpath("//div[@class='subject-others-interests-ft']//a/text()", is_list=True)
        want_watch = int(watch[0][:-3]) if watch and len(watch) > 0 else None
        watched = int(watch[1][:-3]) if watch and len(watch) > 1 else None

        info = [title, country, director, main_actor, actor, type, films_list, date, long, want_watch, watched, p]
        print(info)
        return info
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return [None] * 12  # 如果爬取失败，返回全部 None
    finally:
        driver.quit()


def main(list_form):
    '''
    :param list_form:
    1：从本地获取所爬取的电影列表urls以此爬取电影信息
    0：先爬取电影列表再由获取的电影列表urls爬取电影信息
    :return:None
    保存为本地datas.csv
    '''
    if list_form == 1:
        print('正在由本地urls.txt爬取')
        with open('urls.txt', 'r') as f:
            urls = f.readlines()
        print(urls)
        datas = []
        title_colums = ['title', 'country', 'director', 'main_actor', 'actor', 'type', 'films_list', 'date', 'long',
                        'want_watch', 'watched', 'p']

        for i in range(len(urls)):
            datas.append(get_films_datas(urls[i]))
            print(f'目前进度:{i + 1},共计{len(urls)}个任务', )
        # 写入 CSV 文件
        with open("datas111.csv", "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(title_colums)
            # 写入电影数据
            for row in datas:
                formatted_row = [", ".join(map(str, item)) if isinstance(item, list) else item for item in row]
                writer.writerow(formatted_row)
        print("CSV 文件写入完成！")
    if list_form == 0:
        print('正在爬取电影列表urls共计25rounds')
        urls = get_films_code()
        datas = []
        title_colums = ['title', 'country', 'director', 'main_actor', 'actor', 'type', 'films_list', 'date', 'long',
                        'want_watch', 'watched', 'p']

        for i in range(len(urls)):
            datas.append(get_films_datas(urls[i]))
            print(f'目前进度:{i + 1},共计{len(urls)}个任务', )
        # 写入 CSV 文件
        with open("datas.csv", "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow(title_colums)
            # 写入电影数据
            for row in datas:
                formatted_row = [", ".join(map(str, item)) if isinstance(item, list) else item for item in row]
                writer.writerow(formatted_row)
        print("CSV 文件写入完成！")


while True:
    a = input(
        '输入电影列表获取方式\n1：从本地加载(已经预先爬取电影列表)\n2：从豆瓣电影加载(先爬取电影列表再按照电影列表爬取电影信息)\n')
    if a == '1':
        if os.path.isfile('urls.txt'):
            main(1)
        else:
            print('urls.txt不存在\n请先爬取电影列表')
            continue
    elif a == '2':
        main(0)
