# -*- coding: utf-8 -*-

import urllib
import urllib2
import requests
from bs4 import BeautifulSoup
import hashlib
import os

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/66.0.3359.181 Chrome/66.0.3359.181 Safari/537.36"
SEARCH_PAGE_URL = 'https://www.zimuku.cn/search?q='
DOWNLOAD_PAGE_URL = 'http://www.subku.net/dld/'
g_dst_md5 = None
g_zimu_dir = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) + '/zimuku'


def get_search_page(dst):
    url = SEARCH_PAGE_URL+dst
    resp = urllib.urlopen(url)
    raw = resp.read()
    page_id = get_download_page_id(raw)
    if page_id == '':
        print '没有找到《'+dst+'》的字幕。'
    else:
        get_download_page(page_id, dst)


def get_download_page_id(raw):
    soup = BeautifulSoup(raw, 'html.parser', from_encoding='utf-8')
    nodes = soup.select('div.sublist tr')
    node_info = {'url': '', 'num': 0}
    for node in nodes:
        if node.select('td.last'):
            num = node.select('td.last')[0]
        else:
            continue

        if num:
            num = int(num.get_text())
            if num >= node_info['num']:
                node_info['url'] = node.select('td.first a')[0]['href']
                node_info['num'] = num
    # print node_info['url']
    # print node_info['num']
    page_id = node_info['url'].split('/')[-1]
    return page_id


def get_download_page(page_id, dst):
    url = DOWNLOAD_PAGE_URL + page_id
    resp = urllib.urlopen(url)
    raw = resp.read()
    soup = BeautifulSoup(raw, 'html.parser', from_encoding='utf-8')
    nodes = soup.select('div.down ul li a')
    if nodes:
        get_zimu_file_static_path(nodes[0]['href'], url, dst)


def get_zimu_file_static_path(url, download_page_url, dst):
    print 'getting ' + url
    if str(url).find('static.zimuku.cn/download') != -1:
        get_zimu_file(url, dst)
        return

    urls = url.split('://')
    host = urls[1].split('/')[0]
    cookie = 'PHPSESSID=tje1levp4aa3j65uvalrs1m3b2;'
    if host == 'www.zimuku.cn':
        cookie = 'PHPSESSID=m5tj7eah61e4cho0599pdpmt02;'
        subid = download_page_url.split('/')[-1]
        subid = subid.split('.')[0]
        cookie += 'zmk_home_view_subid='+str(subid)+';'
    cookie += 'yunsuo_session_verify=' + g_dst_md5 + ';'
    print host
    print cookie

    try:
        # req = urllib2.Request(url,
        #                       headers={
        #                           'User-Agent': UA,
        #                           'Host': host,
        #                           'Connection': 'keep-alive',
        #                           'Referer': download_page_url,
        #                           'Cookie': cookie,
        #                           'Upgrade-Insecure-Requests': 1
        #                       })
        # resp = urllib2.urlopen(req)
        resp = requests.get(url, headers={
                                  'User-Agent': UA,
                                  'Host': host,
                                  'Connection': 'keep-alive',
                                  'Referer': download_page_url,
                                  'Cookie': cookie,
                                  'Upgrade-Insecure-Requests': '1'
                              }, allow_redirects=False)
        headers = resp.headers
        print headers
        if headers.get('Location', None):
            get_zimu_file_static_path(headers['Location'], download_page_url, dst)
    except urllib2.HTTPError, e:
        print e.headers
        print e
    # raw = resp.read()
    # print raw


def get_zimu_file(url, dst):
    print 'getting zimu file...'
    resp = urllib2.urlopen(url)
    headers = resp.headers
    print headers
    filename = headers.get('Content-Disposition','').split('filename=')[1]
    file_type = filename.split('.')[-1]
    file_type = file_type.split('\"')[0]
    file_type = file_type.split('\'')[0]
    file_name = os.path.join(g_zimu_dir, dst+'_zimu.'+file_type)

    raw = resp.read()
    with open(file_name, 'w') as fp:
        fp.write(raw)


def get_md5_str(dst):
    m5 = hashlib.md5()
    m5.update(dst)
    return m5.hexdigest()


def get_zimu(dst):
    dst = dst if dst else '因与缘'
    global g_dst_md5
    g_dst_md5 = get_md5_str(dst)
    print g_dst_md5
    get_search_page(dst)


if __name__ == '__main__':
    file_test = '大话西游'#'时空恋旅人'  # '海边的卡夫卡'
    get_zimu(file_test)