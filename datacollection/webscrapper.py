from bs4 import BeautifulSoup
import requests, sys, os
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from itertools import repeat

# Get the path of the current directory
dir_path = os.path.dirname(os.path.realpath(__file__))

def is_mostly_white(img, threshold=0.7):
    # Convert image to RGB
    img = img.convert("RGB")
    
    # Get image dimensions
    width, height = img.size
    total_pixels = width * height
    
    # Initialize white pixel counter
    white_pixels = 0
    
    # Define what counts as "white"
    white_threshold = (250, 250, 250)  # RGB values

    # Count the number of white pixels
    for pixel in img.getdata():
        if pixel >= white_threshold:
            white_pixels += 1

    # Calculate the ratio of white pixels
    white_ratio = white_pixels / total_pixels

    # Check if the ratio exceeds the threshold
    return white_ratio > threshold

def scrape_page(html_text, download_folder, base_site, article_selector, image_container_selector):
    for art_link in get_articles(html_text, base_site, article_selector):
        art_name = art_name = str(get_data(art_link).h1.contents[0]).replace("<i>", "").replace("</i>", "").replace("/", " ").replace("-", "")
        download_image(art_link, art_name, download_folder, image_container_selector)

def get_data(url):
    data = requests.get(url)
    html_text = BeautifulSoup(data.text, 'html.parser')
    return html_text

def get_next_page(html_text, next_page_selector):
    pagination = html_text.find(*next_page_selector[0])
    if not pagination:
        return None
    next_page_link = pagination.find(*next_page_selector[1])
    if not next_page_link:
        return None
    return str(next_page_link['href'])

def get_articles(html_text, base_site, article_selector):
    articles_list = html_text.find(*article_selector[0])
    articles = articles_list.find_all(*article_selector[1])
    article_links = [article.find('a', {"class": "c-card__link u-link-inherit"})['href'] for article in articles]
    article_links = [link for link in article_links if link is not None]
    final_article_links = [base_site + link for link in article_links]
    return final_article_links

def get_image(link, download_folder, art_name):
    global i
    res = requests.get(link)
    image_extension = os.path.splitext(link)[1]
    illegal_characters = ["#","%","&","{","}","\\","<",">","*","$","!","'",'"',":","@","+","`","|","="]
    image_name = os.path.basename(art_name[:20]).translate(str.maketrans({char: " " for char in illegal_characters}))
    image_path = os.path.join(download_folder, f"{image_name}{i}{image_extension}")
    
    if len(res.content) < 1000 or image_extension == '' or image_extension == '.svg':
        raise(ValueError())
    
    img_bytes = BytesIO(res.content)
    img = Image.open(img_bytes)
    white = is_mostly_white(img)
    if not white and img.size[0] > 250 and img.size[1] > 250:
        with open(image_path, 'wb') as image_file:
            image_file.write(res.content)
            i += 1

def download_image(art_link, art_name, download_folder, image_container_selector):
    global i
    i = 0
    page_data = get_data(art_link)
    if art_name is not None:
        image_objs = page_data.find_all(image_container_selector)
        image_links = [obj['src'] for obj in image_objs]
        links = []
        for i, link in enumerate(image_links):
            if not link.startswith('http'):
                links.append('https:' + link)
            else:
                links.append(link)
        with ThreadPoolExecutor() as executor2:
            executor2.map(get_image, links, repeat(download_folder), repeat(art_name))

def scrape():
    nature_article_selector = [['ul', {"class": "app-article-list-row"}], ['li', {"class": "app-article-list-row__item"}]]
    nature_next_page_selector = [['li', {"class":"c-pagination__item", "data-page":"next"}], ['a', {"class": "c-pagination__link"}]]
    nature_image_container_selector = ['img']

    download_folder = dir_path + '/images'
    base_site = 'https://www.nature.com'
    
    if not os.path.exists(download_folder):
        os.mkdir(download_folder)

    search_term = sys.argv[1] if len(sys.argv) == 2 else None
    if not search_term:
        search_term = input("Enter Search Term: ")
    if search_term == '':
        raise SystemExit()

    url = f"https://www.nature.com/search?q={search_term}&journal="
    urls = [base_site + url]
    while url:
        print(f"[+] Added to scrape list: {url}")
        if len(urls) == 1:
            firstpagedata = get_data(url)
            url = get_next_page(firstpagedata, nature_next_page_selector)
        else:
            url = get_next_page(get_data(url), nature_next_page_selector)
        if url:
            url = base_site + url
            urls.append(url)
            if len(urls) == 20:
                break

    args_list = [(get_data(url), download_folder, base_site, nature_article_selector, nature_image_container_selector) for url in urls[1:]]
    args_list.insert(0, (firstpagedata, download_folder, base_site, nature_article_selector, nature_image_container_selector))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(scrape_page, *args) for args in args_list]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

    print("FINISHED SCRAPING")

if __name__ == "__main__":
    start = time.time()
    scrape()
    print(f"Total Time: {time.time() - start}")