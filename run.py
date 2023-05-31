import json
from math import log, sqrt
import random
import threading
import numpy as np
import pytz
import requests
import time
import re
import datetime
import pandas as pd
from tqdm import tqdm
import os
import concurrent.futures
import statistics
import argparse
from bs4 import BeautifulSoup

class PixivDataAnalysis():

    def __init__(
                    self,
                    *, 
                    uid : str, 
                    cookies : list,
                    output_path : str, 
                    proxies = None,
                    sleep_time,
                    quick_mode = False,
                ):
        self.quick_mode = quick_mode
        self.sleep_time = sleep_time
        self.start_time = time.time()
        self.uid = uid
        self.session = requests.Session()
        self.path = output_path
        self.headers_list = [{
            "referer": "https://www.pixiv.net/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36 Edg/112.0.1722.48",
            'baggage' : 'sentry-environment=production,sentry-release=6560f6c82b0cc2220a31afe4ca436a083210e8c7,sentry-public_key=7b15ebdd9cf64efb88cfab93783df02a,sentry-trace_id=4c3c537d81b4413ba200ce44f4f7e51d,sentry-sample_rate=0.0001',
        }]
        self.headers_list = self.headers_list * len(cookies)
        for index, headers in enumerate(self.headers_list):
            headers['cookie'] = cookies[index]
        
        self.proxies = proxies
        self.data = {}
        if not self.path.endswith('/') and not self.path.endswith('\\'):
            self.path += '/'

        text = self.session.get(f'https://www.pixiv.net/users/{self.uid}', headers=self.get_headers(), proxies=self.proxies).text
        self.name = re.findall('meta property="og:title" content="(.*?)"', text)[0]

        print(f'Uid:{self.uid}, Name:{self.name}')
    def get_headers(self):
        return random.choice(self.headers_list)
    def get_pages(self) -> list:
        illustration_url = f'https://www.pixiv.net/ajax/user/{self.uid}/profile/all'
        response = self.session.get(illustration_url, headers=self.get_headers(), proxies=self.proxies)
        response.raise_for_status()
        img_urls = list(response.json()['body']['illusts'].keys())
        return img_urls
    def read_data_from_url(self, url_id: str, index: int, pbar: tqdm, numbe_of_calls=1) -> dict:
        try:
            url = f'https://www.pixiv.net/artworks/{url_id}'
            response = requests.get(url, headers=self.get_headers(), proxies=self.proxies)
            pbar.set_description(f'Pid:{url_id}')
            response.raise_for_status()
            text = response.text

            view = re.findall('"viewCount":(.*?),', text)[0]
            bookmarkCount = re.findall('"bookmarkCount":(.*?),"', text)[0]
            r18 = re.findall('"tag":"(.*?)","locked"', text)
            r18 = r18[0] if r18 else ''
            
            soup = BeautifulSoup(response.text, "html.parser")
            total_url = f'https://www.pixiv.net/ajax/illust/{url_id}/pages'
            total = len(requests.get(total_url, headers=self.get_headers(), proxies=self.proxies).json()['body'])
            
            content = soup.find("meta", {"id": "meta-preload-data"})["content"]
            json_data = json.loads(content)
            unformatted_time = json_data['illust'][url_id]['createDate']
            
            pbar.set_description(f'Pid:{url_id}')
            delta_time = self.get_utc_timestamp() - self.convert_to_utc_timestamp(unformatted_time)
            delta_time += time.time() - self.get_utc_timestamp()
            
            view_per_day = int(view) / delta_time * 3600 * 24
            bookmark_ratio = float(bookmarkCount) / float(view) * 100
            is_r18 = r18 == 'R-18'
            view = int(view)
            point = log(view / sqrt(delta_time) * view / log(view, 2), 2)
            self.data[index] = [
                url_id,
                delta_time,
                view,
                bookmarkCount,
                view_per_day,
                bookmark_ratio,
                is_r18,
                total,
                point
            ]
            pbar.update(1)
        except:
            if numbe_of_calls != 1:
                with tqdm(range(120), leave=False) as sleep_pbar:
                    for i in sleep_pbar:
                        sleep_pbar.set_description(f'Pid:{url_id} Number of calls:{numbe_of_calls}')
                        time.sleep(0.25)
            self.read_data_from_url(url_id, index, pbar, numbe_of_calls + 1)

    def tqdm_keep_alive(self, pbar : tqdm, t : float):
        while True:
            time.sleep(0.25)
            pbar.set_postfix_str(f' {pbar.n/(time.time() - t):.2f}it/s')
            pbar.refresh()
    def get_utc_timestamp(self):
        utc_time = datetime.datetime.utcnow()
        utc_timestamp = int(utc_time.timestamp())
        return utc_timestamp
    def convert_to_utc_timestamp(self,time_str):
        time_obj = datetime.datetime.fromisoformat(time_str)
        tz_offset = time_obj.utcoffset()
        time_zone = pytz.FixedOffset(tz_offset.seconds // 60)
        utc_time = time_zone.normalize(time_obj.astimezone(pytz.utc))
        return int(utc_time.timestamp())
    def read_data_from_urls(self, urls : list):
        try:
            with tqdm(total=len(urls), leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]') as pbar, \
                concurrent.futures.ThreadPoolExecutor(max_workers=256) as executor:
                t1 = threading.Thread(target=self.tqdm_keep_alive, args=(pbar, time.time()))
                t1.daemon = True
                t1.start()
                futures = [executor.submit(self.read_data_from_url, img_id, index, pbar) 
                           for index, img_id in enumerate(urls)]
                for future in concurrent.futures.as_completed(futures):
                        future.result()
        finally:
            self.data = {
                index: item[1] \
                for index, item in enumerate(
                    sorted(self.data.items(),
                    key=lambda item: int(item[1][0]),
                    reverse=True)
                )
            }
            return self.data
    
    def save_data(self, data : dict, log):
        a_data=pd.DataFrame(columns=('pid', 'delta_time', 'view', 'bookmark', 'view_per_day', 'bookmark_ratio', 'is_r18', 'num_of_pages', 'point'))
        for i, (_, value) in enumerate(data.items()):
            a_data.loc[i + 1] = value
        t = time.localtime(time.time())
        rsu = '{}-{}-{}_{}-{}-{}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        name = '{}-{}'.format(self.name, rsu)
        name = self.replace_invalid_chars(name)
        a_data.to_excel(f"{self.path}{name}.xlsx")
        file = open(f'{self.path}{name}.txt', 'w', encoding='utf-8')
        file.write(log)
        output = os.path.join(self.path, name)
        print(f'已保存至 "{output}.txt"')
    def print_data(self, data: dict, max_num : int):
        num_of_pages_list = [int(value[7]) for value in data.values()]
        view_list = [int(value[2]) for value in data.values()]
        bookmark_list = [int(value[3]) for value in data.values()]
        R18_num = sum(1 for value in data.values() if value[6])
        delta_time_list = [value[1] for value in data.values()]
        delta_time_list = sorted(delta_time_list)
        last_time = max(delta_time_list)
        first_time = min(delta_time_list)
        delta_time_list = [delta_time_list[i+1] - delta_time_list[i] for i in range(len(delta_time_list)-1)]
        avg_delta_time = sum(delta_time_list) / len(delta_time_list)
        max_view_pid = max(data.items(), key=lambda x: int(x[1][2]))[1][0]
        max_bookmark_pid = max(data.items(), key=lambda x: int(x[1][3]))[1][0]
        min_view_pid = min(data.items(), key=lambda x: int(x[1][2]))[1][0]
        min_bookmark_pid = min(data.items(), key=lambda x: int(x[1][3]))[1][0]
        total_page, total_view, total_bookmark, total_num_of_pages = len(data), sum(view_list), sum(bookmark_list), sum(num_of_pages_list)
        bookmark_dev_view_list = [float(bookmark_list[i]) / view_list[i] for i in range(len(bookmark_list))]
        view_lower_limit, view_upper_limit = self.get_outlier_limit(view_list)
        bookmark_lower_limit, bookmark_upper_limit = self.get_outlier_limit(bookmark_list)
        IQR_view = self.quartiles(view_list)
        IQR_bookmark = self.quartiles(bookmark_list)
        if max_num is None:
            log = f'投稿数:{total_page}\t'
        else:
            log = f'节选投稿数:{total_page}[{max_num * 100:.0f}%]\t'
        if R18_num > 0:
            log += f'R18投稿数:{R18_num}\t总阅读:{total_view}\t总收藏:{total_bookmark}\t总图片数:{total_num_of_pages}\t'
            log += f'R18投稿占比:{float(R18_num) / total_page * 100:.3f}%\n'
        else:
            log = f'投稿数:{total_page}\t总阅读:{total_view}\t总收藏:{total_bookmark}\t总图片数:{total_num_of_pages}\n'
        log += f'平均阅读:{float(total_view) / total_page:8.3f}\n平均收藏:{float(total_bookmark) / total_page:8.3f}\n平均图片数:{float(total_num_of_pages) / total_page:.3f}\n'
        log += f'收藏/阅读(整体):{float(total_bookmark) / total_view * 100:.3f}%\n收藏/阅读(平权):{float(sum(bookmark_dev_view_list)) / total_page * 100:.3f}%\n'
        log += f'最高阅读:{max(view_list)}\tpid:{max_view_pid}\t最低阅读:{min(view_list)}\tpid:{min_view_pid}\n'
        log += f'最高收藏:{max(bookmark_list)}\tpid:{max_bookmark_pid}\t最低收藏:{min(bookmark_list)}\tpid:{min_bookmark_pid}\n'
        log += f'阅读量(每投稿) mean:{statistics.mean(view_list):.3f}\tmed:{IQR_view[1]}\tsk:{self.get_skewness(view_list):.3f}\tcv:{statistics.stdev(view_list) / statistics.mean(view_list) * 100 :.3f}%\t95%CI:[{view_lower_limit:.0f}, {view_upper_limit:.0f}]\tIQR:{IQR_view}\n'
        log += f'收藏量(每投稿) mean:{statistics.mean(bookmark_list):.3f}\tmed:{IQR_bookmark[1]}\tsk:{self.get_skewness(bookmark_list):.3f}\tcv:{statistics.stdev(bookmark_list) / statistics.mean(bookmark_list) * 100 :.3f}%\t95%CI:[{bookmark_lower_limit:.0f}, {bookmark_upper_limit:.0f}]\tIQR:{IQR_bookmark}\n'
        log += f'平均投稿间隔:{self.timestamp2str(avg_delta_time)}\t中位投稿间隔:{self.timestamp2str(statistics.median(delta_time_list))}\t距离最近投稿:{self.timestamp2str(first_time)}\t距离首次投稿:{self.timestamp2str(last_time)}'
        print(log)
        return log
    def quartiles(self, data):
        """
        计算数据集的四分位数
        :param data: 数据集，类型为list
        :return: 四分位数列表，类型为list
        """
        q1 = np.percentile(data, 25)
        q2 = np.percentile(data, 50)
        q3 = np.percentile(data, 75)
        return [round(q1, 3), round(q2, 3), round(q3, 3)]
    def get_skewness(self, data):
        data = np.array(data)
        skewness = pd.Series(data).skew()
        return skewness
    def timestamp2str(self, timestamp: float) -> str:
        timestamp = int(timestamp)
        time_units = [(31536000, "年"), (86400, "日"), (3600, "时"), (60, "分"), (1, "秒")]
        output = ""
        for unit, unit_str in time_units:
            if timestamp >= unit:
                num_units = int(timestamp // unit)
                timestamp %= unit
                if num_units >= 1:
                    output += str(num_units) + unit_str
        if output == "":
            output += "0秒"
        return output
    def replace_invalid_chars(self, s: str) -> str:
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        new_s = ''
        for c in s:
            if c in invalid_chars:
                new_s += '_'
            else:
                new_s += c

        return new_s
    def get_outlier_limit(self, data: list) -> tuple:
        data = np.array(data)
        n = len(data)
        bootstraps = 10000
        conf_level = 0.95
        means = []
        for _ in range(bootstraps):
            resample = np.random.choice(data, size=n, replace=True)
            mean = resample.mean()
            means.append(mean)
        alpha = (1 - conf_level) / 2
        lower_limit = np.percentile(means, alpha * 100)
        upper_limit = np.percentile(means, (1 - alpha) * 100)
        return lower_limit, upper_limit
    def show_data(self, data : dict):
        import matplotlib.pyplot as plt
        point = [(index + 1, item[8]) for index, item in enumerate(data.values())]
        x_point = [x[0] for x in point]
        y_poiny = [x[1] for x in point]
        plt.plot(x_point, y_poiny)
        plt.title('Point')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


def main(uid, output_dir, cookie, sleep_time, quick_mode, max_num):
    with open(cookie, 'r') as f:
        cookies = f.read().split('\n')
    os.makedirs(output_dir, exist_ok=True)
    p = PixivDataAnalysis(  
                            uid=uid,
                            cookies=cookies,
                            output_path=output_dir,
                            proxies=PROXIES,
                            sleep_time=sleep_time,
                            quick_mode=quick_mode)

    img_urls = p.get_pages()
    if max_num is not None:
        if max_num > len(img_urls):
            max_num = len(img_urls)
        if max_num < 1:
            max_num = len(img_urls) * max_num
        max_num = int(max_num)
        img_urls = img_urls[:max_num]
    data = p.read_data_from_urls(img_urls)
    log = p.print_data(data, args.max_num)
    p.show_data(data)
    p.save_data(data, log)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--uid', type=str, required=True)
    parser.add_argument('-c', '--cookie', type=str, default='cookies.txt')
    parser.add_argument('-o', '--output_dir', type=str, default='output/')
    parser.add_argument('-t', '--sleep_time', type=float, default=1.1)
    parser.add_argument('-m', '--max_num', type=float, default=None)
    args = parser.parse_args()
    PROXIES = {
        'http': 'http://127.0.0.1:7890',
        'https': 'http://127.0.0.1:7890'
    }   
    main(args.uid, args.output_dir, args.cookie, args.sleep_time, True, args.max_num)