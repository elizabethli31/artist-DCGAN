import requests
from bs4 import BeautifulSoup

# Feed any artist (available on wikiart.org)
# You can have multiple artists
# Try to have at least 1000 images
artists = ['claude-monet']

ARTIST_ALL = 'https://www.wikiart.org/en/{artist}/all-works/text-list'
IMG_URL = 'https://uploads4.wikiart.org/images/{artist}/{img_name}.jpg'

def get_all_paintings(soup):

    painting_names = []

    paintings = soup.find_all('li', class_='painting-list-text-row')

    for painting in paintings:
        link = painting.find('a')['href']
        img_name = link.split('/')[-1]
        painting_names.append(img_name)

    return painting_names;

def download_paintings(painting_names, artist):
    for name in painting_names:
        link = IMG_URL.format(artist=artist, img_name=name)
        url = requests.get(link, stream=True)
        with open('../test/data/' + name + '.jpg', 'wb') as fout:
            fout.write(url.content)

def main():
    for artist in artists:
        url = ARTIST_ALL.format(artist=artist)
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')

        painting_names = get_all_paintings(soup)
        download_paintings(painting_names, artist)

if __name__=='__main__':
    main()
