# Flickr Image Crawler and Vector Database Project

This project consists of three main parts:

1. **Flickr Image Crawler**: Scrapes images from Flickr based on search terms and categories, then downloads and preprocesses these images.
2. **Dataset Preprocessing**: Organizes the downloaded images into training and testing datasets and prepares them for machine learning tasks.
3. **Vector Database**: Builds a vector database using image embeddings and performs image similarity searches using different similarity metrics.

## Part 1: Flickr Image Crawler

### 1. Install Required Libraries

Run the following commands to install the necessary libraries:

```python
%pip install tqdm
!apt-get update
!apt-get install -y wget
%pip install selenium
!apt-get install -y chromium-browser
!apt-get install -y chromium-chromedriver
```

### 2. Collect Data - Crawl URLs from Website

The `UrlScraper` class is used to scrape image URLs from Flickr based on search terms. You can specify the number of images to collect and the number of concurrent workers to speed up the data collection process.

### Code Example

**1. Define the `UrlScraper` class**

```python
class UrlScraper:
    def __init__(self, url_template, max_images=50, max_workers=4):
        """
        Initialize the scraper with the provided URL template, number of images to collect, and number of workers.
        """
        self.url_template = url_template
        self.max_images = max_images
        self.max_workers = max_workers
        self.setup_environment()

    def setup_environment(self):
        """
        Set up the environment for Selenium WebDriver.
        """
        os.environ['PATH'] += ':/usr/lib/chromium-browser/'
        os.environ['PATH'] += ':/usr/lib/chromium-browser/chromedriver/'

    def get_url_images(self, term):
        """
        Scrape image URLs based on the search term.
        """
        # Initialize Chrome browser
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)

        url = self.url_template.format(search_term=term)
        driver.get(url)

        # Begin scraping image URLs
        urls = []
        more_content_available = True

        pbar = tqdm(total=self.max_images, desc=f"Scraping images for {term}")

        while len(urls) < self.max_images and more_content_available:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            img_tags = soup.find_all("img")

            for img in img_tags:
                if len(urls) >= self.max_images:
                    break
                if 'src' in img.attrs:
                    href = img.attrs['src']
                    img_path = urljoin(driver.current_url, href)
                    img_path = img_path.replace("_m.jpg", "_b.jpg").replace("_n.jpg", "_b.jpg").replace("_w.jpg", "_b.jpg")
                    if img_path == "https://combo.staticflickr.com/ap/build/images/getty/IStock_corporate_logo.svg":
                        continue
                    urls.append(img_path)
                    pbar.update(1)

            try:
                load_more_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[@id="yui_3_16_0_1_1721642285931_28620"]'))
                )
                load_more_button.click()
                time.sleep(2)
            except:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)

                new_soup = BeautifulSoup(driver.page_source, "html.parser")
                new_img_tags = new_soup.find_all("img", loading_="lazy")
                if len(new_img_tags) == len(img_tags):
                    more_content_available = False
                img_tags = new_img_tags

        pbar.close()
        driver.quit()
        return urls

    def scrape_urls(self, categories):
        """
        Scrape all image URLs for the given categories and search terms.
        """
        all_urls = {category: {} for category in categories}

        # Multi-threading for efficiency
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_term = {executor.submit(self.get_url_images, term): (category, term)
                              for category, terms in categories.items() for term in terms}

            for future in tqdm(concurrent.futures.as_completed(future_to_term), total=len(future_to_term), desc="Overall progress"):
                category, term = future_to_term[future]
                try:
                    urls = future.result()
                    all_urls[category][term] = urls
                    print(f"\nNumber of images collected for {term}: {len(urls)}")
                except Exception as exc:
                    print(f"\nError with {term}: {exc}")
        return all_urls

    def save_to_file(self, data, filename):
        """
        Save the collected image URLs to a JSON file.
        """
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"Data saved to {filename}")
```
**2. Create an instance of UrlScraper, scrape URLs, and save them to a file:**

```python
  categories = {
    "animal": ["Monkey", "Elephant", "cows", "Cat", "Dog", "bear", "fox", ...],
    "plant": ["Bamboo", "Apple", "Apricot", "Banana", "Bean", ...],
    "furniture": ["bed", "cabinet", "chair", "chests", ...],
    "scenery": ["Cliff", "Bay", "Coast", "Mountains", ...]
}

urltopic = {"flickr": "https://www.flickr.com/search/?text={search_term}"}
scraper = UrlScraper(url_template=urltopic["flickr"], max_images=20, max_workers=5)
image_urls = scraper.scrape_urls(categories)

scraper.save_to_file(image_urls, 'image_urls.json')
  
```

### 3. Collect Data - Download Images from URLs

The `ImageDownloader` class is used to download images from the URLs collected in the previous step and organize them into folders based on their categories.

### Code Example

**1. Define the `ImageDownloader` class**

```python
class ImageDownloader:
    def __init__(self, json_file, download_dir, max_workers=4, delay=1):
        """
        Initialize the downloader with the path to the JSON file containing URLs,
        the directory to save the downloaded images, number of workers, and delay.
        """
        self.json_file = json_file
        self.download_dir = download_dir
        self.max_workers = max_workers
        self.delay = delay
        self.setup_environment()

    def setup_environment(self):
        """
        Create the download directory if it does not exist.
        """
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def download_image(self, url, path):
        """
        Download a single image and save it to the specified path.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            with open(path, 'wb') as file:
                file.write(response.content)
        except Exception as e:
            print(f"Failed to download {url}: {e}")

    def download_images(self):
        """
        Download images for all URLs listed in the JSON file.
        """
        with open(self.json_file, 'r') as file:
            data = json.load(file)
        
        urls = [(url, category, term) for category, terms in data.items() for term, url_list in terms.items() for url in url_list]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for url, category, term in urls:
                folder_path = os.path.join(self.download_dir, category, term)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                filename = os.path.join(folder_path, url.split('/')[-1])
                futures.append(executor.submit(self.download_image, url, filename))
                time.sleep(self.delay)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading images"):
                future.result()

    def export_filename(self):
        """
        Export the list of filenames to a CSV file.
        """
        filenames = []
        for root, _, files in os.walk(self.download_dir):
            for file in files:
                filenames.append(os.path.relpath(os.path.join(root, file), self.download_dir))

        df = pd.DataFrame(filenames, columns=['filename'])
        df.to_csv('filenames.csv', index=False)
        print("Filenames exported to filenames.csv")
```

**2. Create an instance of ImageDownloader, download images, and export filenames:**
```python
  downloader = ImageDownloader(json_file='image_urls.json', download_dir='Dataset', max_workers=4, delay=1)
  downloader.download_images()
  downloader.export_filename()
```

### 4. Data Processing - Clean the Dataset

The `check_and_preprocess_images` function cleans the dataset by removing small or corrupted images and converting non-RGB images to RGB.

### Code Example

**1. Define the `check_and_preprocess_images` function**

```python
from PIL import Image
import os

def check_and_preprocess_images(image_dir):
    """
    Clean the dataset by removing small or corrupted images and converting non-RGB images to RGB.
    """
    for root, _, files in os.walk(image_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Remove images that are too small
                    if img.size[0] < 64 or img.size[1] < 64:
                        os.remove(file_path)
                        continue
                    
                    # Convert non-RGB images to RGB
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                os.remove(file_path)
```

**2. Process and Zip the Cleaned Dataset**
```python
  check_and_preprocess_images('Dataset')

  # Zip the cleaned dataset
  !zip -r /content/drive/MyDrive/Clean_Dataset.zip Dataset
  
  # Copy the filename text file
  !cp filename.txt /content/drive/MyDrive/filename.txt
```

## Part 2: Dataset Preprocessing
### 1. Download Data and Libraries
Download and unzip the cleaned dataset, and install required libraries:
```python
  # Download dataset files
  !gdown --id 1UqWdGQcecZdqjJSmKnFYBPWNBJkgEwWl
  !gdown --id 1tKB4IinxXXXe_teTBW2Ehbrs8GqS5l5B
  
  # Unzip dataset
  !unzip Clean_Dataset
  
  # Install required libraries
  %pip install chromadb
  %pip install open-clip-torch
```
### 2. Process Data - Organize Folder Structure
Organize the dataset into training and testing folders:
```python
  import os
  import shutil
  
  # Define source and target directories
  source_dir = "Dataset"
  train_dir = "data/train"
  test_dir = "data/test"
  
  # Create directories
  os.makedirs(train_dir, exist_ok=True)
  os.makedirs(test_dir, exist_ok=True)
  
  # Read filenames and organize into train and test directories
  with open('filename.txt', 'r') as file:
      filenames = file.readlines()
  
  for filename in filenames:
      filename = filename.strip()
      src_path = os.path.join(source_dir, filename)
      if os.path.exists(src_path):
          # Example logic to split files (e.g., 80% train, 20% test)
          dest_dir = train_dir if hash(filename) % 10 < 8 else test_dir
          shutil.copy(src_path, os.path.join(dest_dir, filename))
```

### 3. Zip the Organized Dataset
```python
  # Zip the organized dataset
  !zip -r /content/drive/MyDrive/data.zip data
```
## Part 3: Project Vector Database and Flickr Image Crawler
### 1. Load Data and Required Libraries
Load the dataset and install necessary libraries:

```python
  # Download dataset files
  !gdown --id 1FMpKFYqyNG0eB-3Ef1Z81s-9ZhuAmmg2
  
  # Unzip dataset
  !unzip data
  
  # Install required libraries
  %pip install chromadb
  %pip install open-clip-torch
```
### 2. Convert Images to Embeddings
Create image embeddings using OpenCLIP:
```python
  import os
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
  
  def get_files_path(path):
      """
      Get a list of file paths from a directory.
      """
      file_paths = []
      for root, _, files in os.walk(path):
          for file in files:
              file_paths.append(os.path.join(root, file))
      return file_paths
  
  def get_single_image_embedding(image):
      """
      Generate an embedding for a single image using OpenCLIP.
      """
      model = OpenCLIPEmbeddingFunction(model_name="ViT-B/32")
      image = Image.open(image).convert("RGB")
      embedding = model.encode(image)
      return embedding
```
## 3.1 Query Images with L2 Collection
Add image embeddings to a collection and perform a search using L2 similarity:

```python
  import chromadb
  from tqdm import tqdm
  
  # Create and populate L2 collection
  chroma_client = chromadb.Client()
  l2_collection = chroma_client.get_or_create_collection(name="l2_collection", metadata={"HNSW_SPACE": "l2"})
  add_embedding(collection=l2_collection, files_path=files_path)
  
  # Search and plot results
  test_path = 'data/test/sample_image.jpg'
  l2_results = search(image_path=test_path, collection=l2_collection, n_results=5)
  plot_results(image_path=test_path, files_path=files_path, results=l2_results)
```
  ### 3.2 Query Images with Cosine Similarity Collection
  Add image embeddings to a collection and perform a search using Cosine similarity:
```python
# Create and populate Cosine collection
cosine_collection = chroma_client.get_or_create_collection(name="Cosine_collection", metadata={"HNSW_SPACE": "cosine"})
add_embedding(collection=cosine_collection, files_path=files_path)

# Search and plot results
cosine_results = search(image_path=test_path, collection=cosine_collection, n_results=5)
plot_results(image_path=test_path, files_path=files_path, results=cosine_results)
```
## Conclusion
This project demonstrates the process of collecting, preprocessing, and organizing image data, as well as utilizing vector databases for image similarity searches. Each part of the project can be executed independently or together to achieve the overall goal.

# Additional Information

For any questions or issues, please open an issue in the repository or contact me at [truonghongkietcute@gmail.com](mailto:truonghongkietcute@gmail.com).

Feel free to customize the project names, descriptions, and any other details specific to your projects. If you encounter any problems or have suggestions for improvements, don't hesitate to reach out. Your feedback and contributions are welcome!

Let me know if there’s anything else you need or if you have any other questions. I’m here to help!
