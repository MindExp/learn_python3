###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError

try:
    html = urlopen('https://cn.bing.com')
except HTTPError as e:
    print(e)
else:
    try:
    	bs = BeautifulSoup(html.read(), 'html.parser')
        bad_content = bs.non_existing_tag.another_tag
    except AttributeError as e:
        print('non_existing_tag was not found!')
    else:
        if bad_content is None:
            print('another_tag was not found!')
        else:
            print(bad_content)

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
from bs4 import BeautifulSoup
import requests

class Content(object):
    """
        Common base class for all articles/pages
    """
    def __init__(self, url=None, title=None, body=None):
        self.url = url
        self.title = title
        self.body = body

    def print(self):
        print('Title: {}'.format(self.title))
        print('URL: {}'.format(self.url))
        print('Body: {}'.format(self.body))

def getPage(url):
    req = requests.get(url)
    return BeautifulSoup(req.content, 'html.parser')

def scrapePeople(url):
    bs = getPage(url)
    title = bs.find('h1').get_text()
    body = bs.find('div', {'class': 'box_con', 'id': 'rwb_zw'}).text
    return Content(url, title, body)

def scrapeBrookings(url):
    bs = getPage(url)
    title = bs.find('h1').text
    body = bs.find('div', {'class': ['post-body', 'post-body-enhanced']}).text
    return Content(url, title, body)


url = 'http://tw.people.com.cn/n1/2018/0809/c14657-30217760.html'
scrapePeople(url).print()
url = 'https://www.brookings.edu/blog/future-development/2018/01/26/delivering-inclusive-urban-access-3-uncomfortable-truths/'
scrapeBrookings(url).print()

###############################
from bs4 import BeautifulSoup
import requests

class Content:
    """
    Common base class for all articles/pages
    """

    def __init__(self, url, title, body):
        self.url = url
        self.title = title
        self.body = body

    def print(self):
        """
        Flexible printing function controls output
        """
        print("URL: {}".format(self.url))
        print("TITLE: {}".format(self.title))
        print("BODY:\n{}".format(self.body))


class Website:
    """
    Contains information about website structure
    """

    def __init__(self, name, url, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.titleTag = titleTag
        self.bodyTag = bodyTag

class Crawler:

    def getPage(self, url):
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException as e:
            print('{}\n{}'.format(url, e))
            return None
        return BeautifulSoup(req.text, 'html.parser')

    def safeGet(self, pageObj, selector):
        """
        BeautifulSoup select function with a single string CSS selector for each
        piece of information you want to collect and put all of these selectors in a
        dictionary object

        Utilty function used to get a content string from a Beautiful Soup
        object and a selector. Returns an empty string if no object
        is found for the given selector
        """
        selectedElems = pageObj.select(selector)
        if selectedElems is not None and len(selectedElems) > 0:
        	# return selectedElems[0].get_text()
            return '\n'.join([elem.get_text() for elem in selectedElems])
        return ''

    def parse(self, site, url):
        """
        Extract content from a given page URL
        """
        bs = self.getPage(url)
        try:
            title = self.safeGet(bs, site.titleTag)
            body = self.safeGet(bs, site.bodyTag)
            if title != '' and body != '':
                content = Content(url, title, body)
                content.print()
        except AttributeError as e:
            print('{}\n{}'.format(url, e))
            return None


crawler = Crawler()

siteData = [
    # <section id="product-description">...</section>
    ['O\'Reilly Media', 'http://oreilly.com', 'h1', 'section#product-description'],
    ['Reuters', 'http://reuters.com', 'h1', 'div.StandardArticleBody_body_1gnLA'],
    # <div class="post-body post-body-enhanced" itemprop="articleBody">...</div>
    ['Brookings', 'http://www.brookings.edu', 'h1', 'div.post-body'],
    ['New York Times', 'http://nytimes.com', 'h1', 'p.story-content']
]

websites = []

for row in siteData:
    websites.append(Website(row[0], row[1], row[2], row[3]))

crawler.parse(websites[0],
              'http://shop.oreilly.com/product/0636920028154.do')
crawler.parse(
    websites[1], 'http://www.reuters.com/article/us-usa-epa-pruitt-idUSKBN19W2D0')
crawler.parse(
    websites[2],
    'https://www.brookings.edu/blog/techtank/2016/03/01/idea-to-retire-old-methods-of-policy-education/')
crawler.parse(
    websites[3],
    'https://www.nytimes.com/2018/01/28/business/energy-environment/oil-boom.html')

###############################
"""
	Crawling Sites Through Search
	应用：通过搜索关键词，在搜索结果中爬取需求内容
"""
from bs4 import BeautifulSoup
import requests

class Content:
    """
    Common base class for all articles/pages
    """

    def __init__(self, topic, url, title, body):
        self.topic = topic
        self.title = title
        self.body = body
        self.url = url

    def print(self):
        """
        Flexible printing function controls output
        """
        print("New article found for topic: {}".format(self.topic))
        print("URL: {}".format(self.url))
        print("TITLE: {}".format(self.title))
        print("BODY:\n{}".format(self.body))


class Website:
    """
    Contains information about website structure
    """

    def __init__(self, name, url, searchUrl, resultListing, resultUrl, absoluteUrl, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.searchUrl = searchUrl
        self.resultListing = resultListing
        self.resultUrl = resultUrl
        self.absoluteUrl = absoluteUrl
        self.titleTag = titleTag
        self.bodyTag = bodyTag

class Crawler:

    def getPage(self, url):
        """
        :param url:
        :return:页面 bs 对象
        """
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException as e:
            print('{}\n{}'.format(url, e))
            return None
        return BeautifulSoup(req.content, 'html.parser')

    def safeGet(self, pageObj, selector):
        """
        BeautifulSoup select function with a single string CSS selector for each
        piece of information you want to collect and put all of these selectors in a
        dictionary object

        Utilty function used to get a content string from a Beautiful Soup
        object and a selector. Returns an empty string if no object
        is found for the given selector
        """
        selectedElems = pageObj.select(selector)
        if selectedElems is not None and len(selectedElems) > 0:
            return selectedElems[0].get_text()
        return ''

    def search(self, topic, site):
        """
        Searches a given website for a given topic and records all pages found
        """
        # 获取搜索结果页面 bs 对象
        bs = self.getPage(site.searchUrl + topic)
        try:
            # 通过 CSS 选择器获取搜索结果对象
            searchResults = bs.select(site.resultListing)
        except AttributeError as e:
            print('{}\n{}'.format(site, e))
            return None
        for result in searchResults:
            # 提取搜素结果对象 URL
            url = result.select(site.resultUrl)[0].attrs["href"]
            # Check to see whether it's a relative or an absolute URL
            if site.absoluteUrl:
                bs = self.getPage(url)
            else:
                bs = self.getPage(site.url + url)
            if bs is None:
                print("Something was wrong with that page or URL. Skipping!")
                return
            # 通过 CSS 选择器爬取需求内容
            title = self.safeGet(bs, site.titleTag)
            body = self.safeGet(bs, site.bodyTag)
            if title != '' and body != '':
                content = Content(topic, title, body, url)
                content.print()


crawler = Crawler()

siteData = [
    ['O\'Reilly Media', 'http://oreilly.com', 'https://ssearch.oreilly.com/?q=',
        'article.product-result', 'p.title a', True, 'h1', 'section#product-description'],
    ['Reuters', 'http://reuters.com', 'http://www.reuters.com/search/news?blob=',
     'div.search-result-content', 'h3.search-result-title a', False, 'h1', 'div.StandardArticleBody_body_1gnLA'],
    ['Brookings', 'http://www.brookings.edu', 'https://www.brookings.edu/search/?s=',
        'div.list-content article', 'h4.title a', True, 'h1', 'div.post-body']
]
sites = []
for row in siteData:
    sites.append(Website(row[0], row[1], row[2],
                         row[3], row[4], row[5], row[6], row[7]))

topics = ['python', 'data science']
for topic in topics:
    print("GETTING INFO ABOUT: " + topic)
    for targetSite in sites:
        crawler.search(topic, targetSite)

###############################
"""
	Crawling Sites Through Links
	应用：通过门户网站爬取内容
"""
from bs4 import BeautifulSoup
import requests, re

class Content:
    """Common base class for all articles/pages"""

    def __init__(self, topic, url, title, body):
        self.topic = topic
        self.title = title
        self.body = body
        self.url = url

    def print(self):
        """
        Flexible printing function controls output
        """
        print("New article found for topic: {}".format(self.topic))
        print("URL: {}".format(self.url))
        print("TITLE: {}".format(self.title))
        print("BODY:\n{}".format(self.body))


class Website:

    def __init__(self, name, url, targetPattern, absoluteUrl, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.targetPattern = targetPattern
        self.absoluteUrl = absoluteUrl
        self.titleTag = titleTag
        self.bodyTag = bodyTag

class Crawler:

    def __init__(self, site):
        self.site = site
        self.visited = []

    def getPage(self, url):
        """
        :param url:
        :return:页面 bs 对象
        """
        try:
            req = requests.get(url)
        except requests.exceptions.RequestException as e:
            print('{}\n{}'.format(url, e))
            return None
        return BeautifulSoup(req.content, 'html.parser')

    def safeGet(self, pageObj, selector):
        """
        BeautifulSoup select function with a single string CSS selector for each
        piece of information you want to collect and put all of these selectors in a
        dictionary object

        Utilty function used to get a content string from a Beautiful Soup
        object and a selector. Returns an empty string if no object
        is found for the given selector
        """
        selectedElems = pageObj.select(selector)
        if selectedElems is not None and len(selectedElems) > 0:
            # return '\n'.join([elem.get_text() for elem in selectedElems])
            return selectedElems[0].get_text()
        return ''

    def parse(self, url):
        bs = self.getPage(url)
        if bs is not None:
            title = self.safeGet(bs, self.site.titleTag)
            body = self.safeGet(bs, self.site.bodyTag)
            if title != '' and body != '':
                content = Content(url, title, body)
                content.print()

    def crawl(self):
        """
        Get pages from website home page
        """
        bs = self.getPage(self.site.url)
        targetPages = bs.findAll('a', href=re.compile(self.site.targetPattern))
        for targetPage in targetPages:
            targetPage = targetPage.attrs['href']
            if targetPage not in self.visited:
                self.visited.append(targetPage)
                if not self.site.absoluteUrl:
                    targetPage = '{}{}'.format(self.site.url, targetPage)
                self.parse(targetPage)


reuters = Website('Reuters', 'https://www.reuters.com', '^(/article/)',
                  False, 'h1', 'div.StandardArticleBody_body_1gnLA')
crawler = Crawler(reuters)
crawler.crawl()
###############################
"""
	Crawling Multiple Page Types
"""
class Website:
    """Common base class for all articles/pages"""

    def __init__(self, name, url, titleTag, bodyTag):
        self.name = name
        self.url = url
        self.titleTag = titleTag
        self.bodyTag = bodyTag


class Product(Website):
    """Contains information for scraping a product page"""

    def __init__(self, name, url, titleTag, productNumber, price):
        Website.__init__(self, name, url, TitleTag)
        self.productNumberTag = productNumberTag
        self.priceTag = priceTag


class Article(Website):
    """Contains information for scraping an article page"""

    def __init__(self, name, url, titleTag, bodyTag, dateTag):
        Website.__init__(self, name, url, titleTag)
        self.bodyTag = bodyTag
        self.dateTag = dateTag

def parsePage(url):
    
    if '/ideas/' in url:
        

oreilly = Website('O\'Reilly', 'https://oreilly.com', 'h1' '')

###############################
from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://www.pythonscraping.com')
bs = BeautifulSoup(html, 'html.parser')
imageLocation = bs.find('a', {'id': 'logo'}).find('img')['src']
urlretrieve (imageLocation, 'logo.jpg')

###############################
import os
from urllib.request import urlopen
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

download_directory = 'downloaded'
base_url = 'http://pythonscraping.com'

def getAbsoluteURL(base_url, source):
    if source.startswith('http://www.'):
        url = 'http://{}'.format(source[11: ])
    elif source.startswith('http://'):
        url = source
    elif source.startswith('www.'):
        url = 'http://{}'.format(source[4: ])
    else:
        url = '{}/{}'.format(base_url, source)
    if base_url not in url:
        return None
    return url

def getDownloadPath(base_url, absolute_url, download_directory):
    path = absolute_url.replace('www.', '')
    path = path.replace(base_url, '')
    print(path)
    path = download_directory + path
    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return path


html = urlopen('http://www.pythonscraping.com')
bs = BeautifulSoup(html, 'html.parser')
download_list = bs.find_all(src=True)

for download in download_list:
    file_url = getAbsoluteURL(base_url, download.attrs['src'])
    if file_url is not None:
        print(file_url)
        try:
            urlretrieve(file_url, getDownloadPath(base_url, file_url, download_directory))
        except OSError as e:
            print(e)

###############################
import csv

csvFile = open('test.csv', 'w+')
try:
    writer = csv.writer(csvFile)
    writer.writerow(('number', 'number plus 2', 'number times 2'))
    for i in range(10):
        writer.writerow( (i, i+2, i*2))
finally:
    csvFile.close()

###############################
import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup

html = urlopen('http://en.wikipedia.org/wiki/Comparison_of_text_editors')
bs = BeautifulSoup(html, 'html.parser')
table = bs.find_all('table', {'class': 'wikitable'})[0]
rows = table.find_all('tr')

csv_file = open('editors.csv', 'w+')
writer = csv.writer(csv_file)
try:
    for row in rows:
        csv_row = []
        for cell in row.find_all(['td', 'th']):
            csv_row.append(cell.get_text())
        writer.writerow(csv_row)
except Exception as e:
    print()
finally:
    csv_file.close()

###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import datetime
import random
import pymysql
import re

conn = pymysql.connect(host='127.0.0.1', unix_socket='/tmp/mysql.sock',
                       user='root', passwd=None, db='mysql', charset='utf8')
cur = conn.cursor()
cur.execute("USE scraping")

random.seed(datetime.datetime.now())

def store(title, content):
    cur.execute('INSERT INTO pages (title, content) VALUES ("%s", "%s")', (title, content))
    cur.connection.commit()

def getLinks(articleUrl):
    html = urlopen('http://en.wikipedia.org'+articleUrl)
    bs = BeautifulSoup(html, 'html.parser')
    title = bs.find('h1').get_text()
    content = bs.find('div', {'id':'mw-content-text'}).find('p').get_text()
    store(title, content)
    return bs.find('div', {'id':'bodyContent'}).findAll('a', href=re.compile('^(/wiki/)((?!:).)*$'))

links = getLinks('/wiki/Kevin_Bacon')
try:
    while len(links) > 0:
         newArticle = links[random.randint(0, len(links)-1)].attrs['href']
         print(newArticle)
         links = getLinks(newArticle)
finally:
    cur.close()
    conn.close()

###############################
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import pymysql
from random import shuffle

conn = pymysql.connect(host='127.0.0.1', unix_socket='/tmp/mysql.sock',
                       user='root', passwd='root', db='mysql', charset='utf8')
cur = conn.cursor()
cur.execute('USE wikipedia')

def insertPageIfNotExists(url):
    cur.execute('SELECT * FROM pages WHERE url = %s', (url))
    if cur.rowcount == 0:
        cur.execute('INSERT INTO pages (url) VALUES (%s)', (url))
        conn.commit()
        return cur.lastrowid
    else:
        return cur.fetchone()[0]

def loadPages():
    cur.execute('SELECT * FROM pages')
    pages = [row[1] for row in cur.fetchall()]
    return pages

def insertLink(fromPageId, toPageId):
    cur.execute('SELECT * FROM links WHERE fromPageId = %s AND toPageId = %s', 
                  (int(fromPageId), int(toPageId)))
    if cur.rowcount == 0:
        cur.execute('INSERT INTO links (fromPageId, toPageId) VALUES (%s, %s)', 
                    (int(fromPageId), int(toPageId)))
        conn.commit()
def pageHasLinks(pageId):
    cur.execute('SELECT * FROM links WHERE fromPageId = %s', (int(pageId)))
    rowcount = cur.rowcount
    if rowcount == 0:
        return False
    return True

def getLinks(pageUrl, recursionLevel, pages):
    if recursionLevel > 4:
        return

    pageId = insertPageIfNotExists(pageUrl)
    html = urlopen('http://en.wikipedia.org{}'.format(pageUrl))
    bs = BeautifulSoup(html, 'html.parser')
    links = bs.findAll('a', href=re.compile('^(/wiki/)((?!:).)*$'))
    links = [link.attrs['href'] for link in links]

    for link in links:
        linkId = insertPageIfNotExists(link)
        insertLink(pageId, linkId)
        if not pageHasLinks(linkId):
            print("PAGE HAS NO LINKS: {}".format(link))
            pages.append(link)
            getLinks(link, recursionLevel+1, pages)
        
        
getLinks('/wiki/Kevin_Bacon', 0, loadPages()) 
cur.close()
conn.close()

###############################
import smtplib
from email.mime.text import MIMEText

msg = MIMEText('The body of the email is here')

msg['Subject'] = 'An Email Alert'
msg['From'] = 'ryan@pythonscraping.com'
msg['To'] = 'webmaster@pythonscraping.com'

s = smtplib.SMTP('localhost')
s.send_message(msg)
s.quit()

###############################
import smtplib
from email.mime.text import MIMEText
from bs4 import BeautifulSoup
from urllib.request import urlopen
import time

def sendMail(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] ='christmas_alerts@pythonscraping.com'
    msg['To'] = 'ryan@pythonscraping.com'

    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()

bs = BeautifulSoup(urlopen('https://isitchristmas.com/'), 'html.parser')
while(bs.find('a', {'id':'answer'}).attrs['title'] == 'NO'):
    print('It is not Christmas yet.')
    time.sleep(3600)
    bs = BeautifulSoup(urlopen('https://isitchristmas.com/'), 'html.parser')
sendMail('It\'s Christmas!', 
         'According to http://itischristmas.com, it is Christmas!')

###############################
import csv
from io import StringIO
from urllib.request import urlopen

data = urlopen('http://pythonscraping.com/files/MontyPythonAlbums.csv').read().decode('ascii', 'ignore')
data_file = StringIO(data)  # 将字符串数据包装为 StringIO 对象
csv_reader = csv.reader(data_file)

for row in csv_reader:
    print(row)
    
###############################

###############################

###############################

###############################

###############################

###############################

###############################

###############################

###############################

###############################

