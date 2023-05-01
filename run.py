import requests
import time
import re
import datetime
import pandas as pd
from tqdm import tqdm
import config
import os

proxies = {
    'http': 'http://127.0.0.1:7890',
    'https': 'http://127.0.0.1:7890'
}

class PixivDataAnalysis():

    def __init__(
                    self,
                    *, 
                    uid : str, 
                    headers_list : list,
                    output_path : str, 
                    default_name : str = None,
                ):

        self.uid = uid
        self.session = requests.Session()
        self.path = output_path
        self.headers_list = headers_list
        self.global_index = 0

        if not self.path.endswith('/') and not self.path.endswith('\\'):
            self.path += '/'

        if default_name == None:
            text = self.session.get(f'https://www.pixiv.net/users/{self.uid}', headers=self.getHeaders(1), proxies=proxies).text
            self.name = re.findall('meta property="og:title" content="(.*?)"', text)[0]
        else:
            self.name = default_name

        print(f'Uid:{self.uid}, Name:{self.name}')

    def getHeaders(self, index = None) -> dict:
        if index != None:
            return self.headers_list[index]
        return self.headers_list[self.global_index]
    def getUrls(self) -> list:
        illustration_url = f'https://www.pixiv.net/ajax/user/{self.uid}/profile/all'
        response = self.session.get(illustration_url, headers=self.getHeaders(1), proxies=proxies)
        response.raise_for_status()
        img_urls = list(response.json()['body']['illusts'].keys())
        return img_urls
    def readData(self, urls : list):
        data = {}
        i = 0
        try:
            pbar = tqdm(urls)
            for img_url_num in pbar:
                pbar.set_description(f'Crawling Page {img_url_num} Using headers index {self.global_index}')

                while True:
                    try:
                        text = self.session.get(f'https://www.pixiv.net/artworks/{img_url_num}', headers=self.getHeaders(), proxies=proxies).text
                        view = re.findall('"viewCount":(.*?),', text)[0]
                        bookmarkCount = re.findall('"bookmarkCount":(.*?),"', text)[0]
                        r18 = re.findall('"tag":"(.*?)","locked"', text)[0] if len(re.findall('"tag":"(.*?)","locked"', text)) > 0 else ''
                        unformatted_time = re.findall('"mini":"(.*?)",', text)[0]
                    except:
                        if self.global_index < len(self.headers_list) - 1:
                            self.global_index += 1
                        else:
                            raise
                    else:
                        break
                format_time = self.formatTime(unformatted_time)
                delta_time = self.creationTimestamp(format_time)

                data[i] = [img_url_num, \
                            delta_time, \
                            view, \
                            bookmarkCount, \
                            int(view) / delta_time * 3600 * 24, \
                            float(bookmarkCount) / float(view) * 100, \
                            r18 == 'R-18']

                i += 1
        finally:
            return data
    def saveData(self, data : dict):
        a_data=pd.DataFrame(columns=('pid', 'delta_time', 'view', 'like', 'view/day', 'like/view', 'isR18'))
        for i in tqdm(range(1, len(data) + 1), desc = 'Writing Document'):
            a_data.loc[i] = data[i - 1]
        t = time.localtime(time.time())
        rsu = '{}-{}-{} {}-{}-{}'.format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)
        name = '{}-{}'.format(self.name, rsu)
        a_data.to_excel(f"{self.path}{name}.xlsx")
        print(f'Saved {name}.xlsx to {self.path[0:-1]}')

    def formatTime(self, unformatted_time : str):
        unformatted_time : list = unformatted_time.split('/')
        format_time = ''
        for i in reversed(range(2, 8)):
            format_time += unformatted_time[-i] + '-'
        return format_time[0:-1]
    
    def creationTimestamp(self, format_time : str):
        offset = int(time.timezone / 60 / 60 * -1)
        now_time = (datetime.datetime.now()+datetime.timedelta(hours= 9 - offset)).strftime("%Y-%m-%d-%H-%M-%S")
        return time.mktime(time.strptime(now_time, '%Y-%m-%d-%H-%M-%S')) \
            - time.mktime(time.strptime(format_time, '%Y-%m-%d-%H-%M-%S'))

def main():
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    p = PixivDataAnalysis(  
                            uid=config.UID,
                            headers_list=config.HEADERS_LIST,
                            output_path=config.SAVE_PATH,
                            default_name=None)

    img_urls = p.getUrls()
    data = p.readData(img_urls)
    p.saveData(data)

if __name__ == '__main__':
    main()
