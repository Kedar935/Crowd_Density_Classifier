from bing_image_downloader import downloader

# # Keywords
# keywords_sparse = [
#     "sparse crowd", "empty street", "few people in park", "few people in mall",
#     "few people in train station", "less crowded street", "small gathering",
#     "social distancing crowd", "low crowd stadium", "less crowded beach",
#     "quiet shopping mall", "few people walking", "empty market",
#     "empty railway platform", "empty bus station"
# ]

keywords_dense = [
    "dense crowd", "crowded street", "crowded market", "crowded mall",
    "crowded train station", "crowded stadium", "crowded beach", "crowded festival",
    "crowded protest", "crowded concert", "crowded rally", "crowded bus station",
    "crowded metro", "crowded religious event", "crowded airport"
]

# Download function
def download_images(keywords, folder):
    for keyword in keywords:
        print(f"Downloading: {keyword}")
        downloader.download(keyword, limit=30, output_dir=f"dataset/{folder}", adult_filter_off=True)

# Download datasets
#download_images(keywords_sparse, "sparse")
download_images(keywords_dense, "dense")

print("✅ Dataset Download Complete!")
