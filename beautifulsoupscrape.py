from bs4 import BeautifulSoup
import requests
import os

url = 'https://www.nature.com/search?q=self+assembly+nanomaterial'
basesite = 'https://www.nature.com'

downloadfolder = 'downloads'
if not os.path.exists(downloadfolder):
    os.mkdir(downloadfolder)

def getdata(url):
	data = requests.get(url)
	htmltext = BeautifulSoup(data.text, 'html.parser')
	return htmltext

def getnextpage(htmltext):
	pagination = htmltext.find('ul', {"class":"c-pagination"})
	if not pagination.find('li', {"data-page":"next"}).find('span',{"class":"c-paginationlink c-paginationlink--disabled"}):
		print(str(pagination.find('li', {"data-page":"next"}).find('a',{"class":"c-pagination__link"})['href']))
		url = str(pagination.find('li', attrs={"data-page":"next"}).find('a',{"class":"c-pagination__link"})['href'])
		return url
	else:
		return None
	
def getarticles(htmltext):
	articleslist = htmltext.find('ul', {"class":"app-article-list-row"})
	articles = articleslist.find_all('li',{"class":"app-article-list-row__item"})
	articlelinks = [article.find('a', {"class": "c-card__link u-link-inherit"})['href'] for article in articles]
	articlelinks = [link for link in articlelinks if link is not None]
	final_articlelinks = [basesite + link for link in articlelinks]
	return final_articlelinks

i=0
while True:
	htmltext = getdata(url)
	for artlink in getarticles(htmltext):
		i += 1
		pagedata = getdata(artlink)
		artname = str(pagedata.h1.contents[0]).replace("<i>", "").replace("</i>", "").replace("/", " ").replace("-", "")
		downloadbuttondiv = pagedata.find('div', {"class": "c-pdf-container"})
		if downloadbuttondiv and artname is not None:
			print(i, artname)
			downloadbuttonlink = downloadbuttondiv.find('a', {"class": "u-button u-button--full-width u-button--primary u-justify-content-space-between c-pdf-download__link"})['href']
			res = requests.get(basesite + downloadbuttonlink)
			if res.status_code == 200 and artname != None:
				pdf_path = os.path.join(downloadfolder, f"{artname[:25]}.pdf")
				with open(pdf_path, 'wb') as pdf:
					pdf.write(res.content)
			else:
				print("FAILED TO DOWNLOAD")

	url = getnextpage(htmltext)
	if not url:
		break
	else:
		url = basesite + url
