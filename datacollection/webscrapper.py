from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import PySimpleGUI as sg
from multiprocessing import Process
import requests, sys, os, platform

# Get the path of the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

# Define a function to scrape scientific articles
def scrape_page(url_data, download_folder, base_site, article_selector, pdf_container_selector):
    i = 0
    html_text = url_data

    # Iterate through the articles on the current page
    for art_link in get_articles(html_text, base_site, article_selector):
        i += 1
        art_name = get_article_name(art_link)
        print("[+]", i, art_name)
        download_pdf(art_link, art_name, base_site, download_folder, pdf_container_selector)

# Function to retrieve HTML data from a URL using requests and BeautifulSoup
def get_data(url):
    data = requests.get(url)
    html_text = BeautifulSoup(data.text, 'html.parser')
    return html_text

# Function to get the URL of the next page of results
def get_next_page(html_text, next_page_selector):
    pagination = html_text.find(*next_page_selector[0])
    if not pagination:
        return None
    next_page_link = pagination.find(*next_page_selector[1])
    if not next_page_link:
        return None

    return str(next_page_link['href'])

# Function to extract article links from the current page
def get_articles(html_text, base_site, article_selector):
    articles_list = html_text.find(*article_selector[0])
    articles = articles_list.find_all(*article_selector[1])

    article_links = [article.find('a', {"class": "c-card__link u-link-inherit"})['href'] for article in articles]
    article_links = [link for link in article_links if link is not None]
    final_article_links = [base_site + link for link in article_links]

    return final_article_links

# Function to get the name of an article from its link
def get_article_name(art_link):
    page_data = get_data(art_link)
    art_name = str(page_data.h1.contents[0]).replace("<i>", "").replace("</i>", "").replace("/", " ").replace("-", "")
    return art_name

# Function to download a PDF article
def download_pdf(art_link, art_name, base_site, download_folder, pdf_container_selector):
    page_data = get_data(art_link)
    download_button_div = page_data.find(*pdf_container_selector[0])
    if download_button_div and art_name is not None:
        download_button_link = download_button_div.find(*pdf_container_selector[1])['href']
        res = requests.get(base_site + download_button_link)

        if res.status_code == 200 and art_name is not None:
            pdf_path = os.path.join(download_folder, f"{art_name[:25]}.pdf")
            # Make a legal filename
            illegal_characters = ["#","%","&","{","}","\\","<",">","*","$","!","'",'"',":","@","+","`","|","="]
            for char in illegal_characters:
                if char in pdf_path:
                    pdf_path.replace(char, " ")

            with open(pdf_path, 'wb') as pdf:
                pdf.write(res.content)
        else:
            print("FAILED TO DOWNLOAD")

# Function to perform a search and return the search results URL using Selenium
def perform_search(search_term):
    search_term = search_term.strip().replace(" ", "+")
    search_results_url = f"https://www.nature.com/search?q={search_term}&journal="
    
    return search_results_url

# Main Logic and Parallelization
def scrape():
    # Define journal-specific selectors (modify as needed)
    nature_article_selector = [['ul', {"class": "app-article-list-row"}], ['li', {"class": "app-article-list-row__item"}]]
    nature_next_page_selector = [['li', {"class":"c-pagination__item", "data-page":"next"}], ['a', {"class": "c-pagination__link"}]]
    nature_pdf_container_selector = [['div', {"class": "c-pdf-container"}] , ['a', {"class": "u-button u-button--full-width u-button--primary u-justify-content-space-between c-pdf-download__link"}]]

    download_folder = dir_path + '/downloaded'
    base_site = 'https://www.nature.com'
    # Create the download folder if it doesn't exist
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)

    # Check if a search term is provided as a command-line argument, otherwise prompt the user
    search_term = sys.argv[1] if len(sys.argv) == 2 else None
    if not search_term:
        search_term = sg.popup_get_text("Enter Search Term:", title="Webscraper Search Term")
    if not search_term:
        raise SystemExit()
    # Good search term for end-to-end testing
    #search_term = "priming self assembly blocks copolymers nanomaterials imaging defect"
    
    # Perform the initial search and get the results URL
    url = perform_search(search_term)
    urls = [base_site + url]
    while url:
        print(f"[+] Added to scrape list: {url}")
        if len(urls) == 1:
            # Need to store data from first iter otherwise nature cause nature wont allow another request
            firstpagedata = get_data(url)
            url = get_next_page(firstpagedata, nature_next_page_selector)
        else:
            url = get_next_page(get_data(url), nature_next_page_selector)
        if url:
            url = base_site + url
            urls.append(url)
            if len(urls) == 20: #Max number of pages that Nature will let you observe
                break

    processes = []
    args_list = [(get_data(url), download_folder, base_site, nature_article_selector, nature_pdf_container_selector) for url in urls[1:]]
    args_list.insert(0, (firstpagedata, download_folder, base_site, nature_article_selector, nature_pdf_container_selector))

    # Sometimes gives error: Max retries exceeded with url, in this case it will just not scrape page=1 of pages
    if platform.system() == 'Linux':
        for i in range(len(urls)):
            scrape_page(*args_list[i])
            
    else:
        for i in range(len(urls)):
            p = Process(target=scrape_page, args= args_list[i])
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

    print("FINISHED SCRAPPING")
