from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import mypath
from time import sleep
import link_category_util
from scrapy_util import _link_is_within_playstore
import logging
from selenium.webdriver.remote.remote_connection import LOGGER
LOGGER.setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("scrapy.core.engine").setLevel(logging.WARNING)


def create_selenium_browser():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--log-level=30')
    options.add_argument("--window-size=1920x1080")
    chrome_prefs = {}
    options.experimental_options["prefs"] = chrome_prefs
    chrome_prefs["profile.default_content_settings"] = {"images": 2}
    chrome_prefs["profile.managed_default_content_settings"] = {"images": 2}
    driver = webdriver.Chrome(chrome_options=options, executable_path= mypath.chrome_driver)
    return driver

def extract_all_links(url, driver):
    print('crawling cluster :',url)
    oldheight = 0
    driver.get(url)
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        sleep(1.5)
        height = driver.execute_script("return document.body.scrollHeight")
        if oldheight == height:
            break
        else:
            oldheight = height
            print('scrolled')

    elems = driver.find_elements_by_xpath("//a[@href]")
    output = []
    for elem in elems:
        link = elem.get_attribute('href')
        if link_category_util.link_is_queueable(link) and _link_is_within_playstore(link):
            output.append(link)
    print(len(output),'links extracted')
    return output