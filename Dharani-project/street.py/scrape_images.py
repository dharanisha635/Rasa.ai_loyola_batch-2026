from icrawler.builtin import BingImageCrawler

# CLEAN STREET KEYWORDS
clean_keywords = [
    ("clean street", 30),
    ("clean road", 30),
    ("clean city street", 25),
    ("well maintained road", 25),
    ("smart city road", 20)
]

# DIRTY STREET KEYWORDS
dirty_keywords = [
    ("dirty street", 30),
    ("garbage road", 30),
    ("trash on street", 25),
    ("polluted road", 25),
    ("street litter", 20)
]

# DOWNLOAD CLEAN IMAGES
for word, count in clean_keywords:

    clean_crawler = BingImageCrawler(
        storage={'root_dir': 'dataset/train/clean'}
    )

    clean_crawler.crawl(
        word,
        max_num=count
    )

# DOWNLOAD DIRTY IMAGES
for word, count in dirty_keywords:

    dirty_crawler = BingImageCrawler(
        storage={'root_dir': 'dataset/train/dirty'}
    )

    dirty_crawler.crawl(
        word,
        max_num=count
    )

print("Image scraping completed!")