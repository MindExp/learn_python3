###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

try:
    html = urlopen('https://cn.bing.com')
    if html is None:  # 服务器访问结果检查
        print('URL is not found.')
        exit()
except HTTPError as e:
    print(e)
else:
    bsObj = BeautifulSoup(html.read(), 'html5lib')
    if bsObj.title is None:  # 标签检查
        print('Tag title is not found')
    else:
        print(bsObj.title)

###############################
from urllib.request import urlopen
from urllib.request import HTTPError
from bs4 import BeautifulSoup

def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
        return None
    try:
        bs_obj = BeautifulSoup(html.read(), 'html5lib')
        title = bs_obj.title
    except AttributeError as e:
        print(e)
        return None
    return title


url = 'https://cn.bing.com'
title = getTitle(url)
if title is None:
    print('Title could not be found')
else:
    print(title)
    
###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

def main():
    url = 'http://www.pythonscraping.com/pages/warandpeace.html'
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    else:
        try:
            bs_obj = BeautifulSoup(html.read(), 'html5lib')
            # bs_obj.findAll(tag, attributes), the value of attributes is a key-value pair.
            tag_list = bs_obj.findAll('span', {'class': ['green', 'red']})
        except AttributeError as e:
            print(e)
        else:
            for tag in tag_list:
                # .get_text() 会把正在处理的 HTML 文档中所有的标签都清除，然后返回一个只包含文字的字符串
                print(tag.get_text())


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

def main():
    url = 'http://www.pythonscraping.com/pages/page3.html'
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    else:
        try:
            bs_obj = BeautifulSoup(html.read(), 'html5lib')
            for child in bs_obj.find('table', {'id': 'giftList'}).children:
                print(child)
        except AttributeError as e:
            print(e)


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup

def main():
    url = 'http://www.pythonscraping.com/pages/page3.html'
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    else:
        try:
            bs_obj = BeautifulSoup(html.read(), 'html5lib')
            for child in bs_obj.find('table', {'id': 'giftList'}).tr.next_siblings:
                print(child)
        except AttributeError as e:
            print(e)


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup


def main():
    url = 'http://www.pythonscraping.com/pages/page3.html'
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    else:
        try:
            bs_obj = BeautifulSoup(html.read(), 'html5lib')
            print(bs_obj.find('img', {'src': '../img/gifts/img2.jpg'}).parent.previous_sibling.get_text())
        except AttributeError as e:
            print(e)


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import re

def main():
    url = 'http://www.pythonscraping.com/pages/page3.html'
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
    else:
        try:
            bs_obj = BeautifulSoup(html.read(), 'html5lib')
            print(bs_obj.img.attrs)  # 以字典形式返回第一个 img 标签的所有属性值
            images = bs_obj.findAll('img', {'src': re.compile('\.\./img/gifts/img.*.jpg')})
            for image in images:
                print(image, image['src'])
        except AttributeError as e:
            print(e)


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

def main():
    url = 'http://en.wikipedia.org/wiki/Kevin_Bacon'
    html = urlopen(url)
    bs_obj = BeautifulSoup(html.read(), 'html5lib')
    for link in bs_obj.findAll('a'):
        if 'href' in link.attrs:
            print(link.attrs['href'])


if __name__ == '__main__':
    main()
    
###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

def main():
    url = 'http://en.wikipedia.org/wiki/Kevin_Bacon'
    html = urlopen(url)
    bs_obj = BeautifulSoup(html.read(), 'html5lib')
    # 过滤出词条链接
    for link in bs_obj.find('div', {'id': 'bodyContent'}).findAll('a', href=re.compile('^(/wiki/)((?!:).)*$')):
        if 'href' in link.attrs:
            print(link.attrs['href'])


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import re

def getLink(article_url):
    try:
        html = urlopen(article_url)
    except HTTPError as e:
        print(e)
    else:
        bs_obj = BeautifulSoup(html.read(), 'html5lib')
        try:
            url_list = bs_obj.find('div', {'id': 'bodyContent'}).findAll('a',
                                                                         {'href': re.compile('^(/wiki/)((?!:).)*$')})
        except AttributeError as e:
            print(e)
        else:
            return url_list


def main():
    url = 'https://en.wikipedia.org/wiki/Kevin_Bacon'
    for link in getLink(url):
        print('{0}\t{1}{2}'.format(link.get_text(), url, link.attrs['href']))


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

pages = set()
def getLink(page_url):
    global pages
    # 单域名数据采集
    html = urlopen('https://en.wikipedia.org' + page_url)
    bs_obj = BeautifulSoup(html, 'html5lib')
    for link in bs_obj.findAll('a', {'href': re.compile('^(/wiki/)')}):
        if link.attrs['href'] not in pages:
            new_page = link.attrs['href']
            print(new_page)
            pages.add(new_page)
            getLink(new_page)

def main():
    getLink('/wiki/Kevin_Bacon')


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

pages = set()
def getLink(page_url):
    global pages
    html = urlopen('https://en.wikipedia.org' + page_url)
    bs_obj = BeautifulSoup(html, 'html.parser')
    try:
        print(bs_obj.h1.get_text())
        print(bs_obj.find({'id': 'mw-content-text'}).findAll('p')[0])
        print(bs_obj.find(id='ca-edit').find('span').find('a').attrs['href'])
    except AttributeError:
        print('This page is missing something! Continuing.')
    for link in bs_obj.findAll('a', {'href': re.compile('^(/wiki/)')}):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                new_page = link.attrs['href']
                print('-------------------\n' + new_page)
                pages.add(new_page)
                getLink(new_page)

def main():
    getLink('')


if __name__ == '__main__':
    main()

###############################
from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re
import datetime
import random

pages = set()
random.seed(datetime.datetime.now())

# Retrieves a list of all Internal links found on a page
def getInternalLinks(bs, include_url):
    include_url = '{}://{}'.format(urlparse(include_url).scheme, urlparse(include_url).netloc)
    internal_links = []
    # Finds all links that begin with a "/"
    for link in bs.findAll('a', {'href': re.compile('^(/|.*' + include_url + ')')}):
        internal_link = link.attrs['href']
        if internal_link is not None:
            if internal_link not in internal_links:
                if link.attrs['href'].starstwith('/'):
                    internal_links.append(include_url + internal_link)
                else:
                    internal_links.append(internal_link)
    return internal_links

# Retrieves a list of all external links found on a page
def getExternalLinks(bs, exclude_url):
    external_links = []
    # Finds all links that start with "http" that do
    # not contain the current URL
    for link in bs.findAll('a', {'href': re.compile('^(http|www)((?!' + exclude_url + ').)*$')}):
        external_link = link.attrs['href']
        if external_link is not None:
            if external_link not in external_links:
                external_links.append(external_link)
    return external_links

def splitAddress(address):
    address_parts = address.replace('http://', '').split('/')
    return address_parts

def getRandomExternalLink(starting_page):
    html = urlopen(starting_page)
    bs = BeautifulSoup(html, 'html.parser')
    # urlparse(starting_page).scheme：获取网络协议名称
    # urlparse(starting_page).netloc：获取 url 域名
    external_links = getExternalLinks(bs, urlparse(starting_page).netloc)
    if len(external_links) == 0:
        print('No external links, looking around the site for one')
        domain = '{}://{}'.format(urlparse(starting_page).scheme, urlparse(starting_page).netloc)
        internal_links = getInternalLinks(bs, domain)
        return getRandomExternalLink(internal_links[random.randint(0, len(internal_links)-1)])
    else:
        return external_links[random.randint(0, len(external_links)-1)]

def followExternalOnly(starting_site):
    external_link = getRandomExternalLink(starting_site)
    print('Random external link is: {}'.format(external_link))
    followExternalOnly(external_link)

def main():
    followExternalOnly('http://oreilly.com')


if __name__ == '__main__':
    main()

###############################