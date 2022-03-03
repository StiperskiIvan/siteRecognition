from selenium import webdriver
from time import sleep

number_of_images = 50
no_of_item = 1
list_of_sites = ["https://www.theguardian.com/",
                 "https://www.spiegel.de/",
                 "https://cnn.com/",
                 "https://www.bbc.com/",
                 "https://www.amazon.com/",
                 "https://www.ebay.com/",
                 "https://www.njuskalo.hr/",
                 "https://www.google.com/",
                 "https://github.com/",
                 "https://www.youtube.com/"]

list_of_folders = ["theGuardian", "spiegel", "cnn", "bbc", "amazon", "ebay", "njuskalo", "google", "github", "youtube"]

# path = 'C:/Users/Stipe/Downloads/chromedriver_win32/chromedriver.exe'
path = 'C:/Users/Stipe/Downloads/operadriver_win64/operadriver_win64/operadriver.exe'
save_path_base = 'data/training/'
browser = webdriver.Opera(executable_path=path)

browser.get(list_of_sites[no_of_item])
for i in range(number_of_images):
    if i == 0:
        sleep(30)
    sleep(1.5)
    name = save_path_base + list_of_folders[no_of_item] + '/' + list_of_folders[no_of_item] + str(i) + '.jpg'
    browser.save_screenshot(name)
browser.quit()
