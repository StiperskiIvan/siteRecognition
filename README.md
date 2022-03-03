# siteRecognition
ML model to detect learned websites

The purpose of the program is to take an screenshot of full screen browser and return a link to a detected website.

To solve the site detection problem a CNN architecture ML model is used.

Model architecture is shown in the image bellow

<img
  src="/model2.png"
  alt="Alt text"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
  
  Classes used for training are:
  
 ● https://www.theguardian.com/
 
 ● https://www.spiegel.de/
 
 ● https://cnn.com
 
 ● https://www.bbc.com/
 
 ● https://www.amazon.com/
 
 ● https://www.ebay.com/
 
 ● https://www.njuskalo.hr/
 
 ● https://www.google.com/
 
 ● https://github.com/
 
 ● https://www.youtube.com/

 Model is trained using 200 images per class, 90/10 training and testing set and 20% of training set is used for validation set
 Images are 250x250x3 RGB images.
 
 To obtain your own images you can use the **web_scrape_screenshots.py** script, setup your prefered time of saved screenshot, put a path to your browser_driver.exe and you are  ready to go, just run the script.
 
_ For the sake of simplicity only light theme browser images were used for training. _
 
 Training images can be requested on demand from a link https://drive.google.com/file/d/1Ag0CvqG12ObszsJQ6PGycczwZyV8URQh/view?usp=sharing
 
 # Results
 
 Results of classification network are not that good, training accuracy being 77%, that may be due to poorly chosen datasets and images, a too few images for training.
 One trick could be data augumentation to increase our dataset or simply taking more images. 
 
 Another approach would be to try a grayscale insted of an RGB imge for training.
 
 To pinpoint the error a confussion matrix is calculated (included in the **tools.py**) for the test set.
 Test set consisted of 20 images per class.
 On the image bellow you can see that our results are not bad on 8 classes even excellent on some (can't tell due to small testing set)but they really fail on classes youtube and the Guardian. Those images should be more thoroughly picked for another training and that could improve the classification significantlly.
 
 <img
  src="/confusionMatrix.png"
  alt="Alt text"
  style="display: inline-block; margin: 0 auto; max-width: 300px">
 
 # Setup
 
 First part is to clone the project to your local computer
 
 Second you need to install the requirements for the program via ```pip install -r requirements.txt``` command in your terminal
 
 Third you can run the script via your terminal by typing ```python3 main.py --input-path /path/to/your/image.jpg``` and thats it. 
 Folder test images also contains a few images of each class
 
 If you want to retrain the model you can just uncoment in the **main.py** code ```model = train_model()``` and use your own images in folder _data/training_ and _data/testing_
 
 # Future steps

To further improve the code you can do a simple web-api with the use of Flask API by using the decorator functions inside your code, and setting it up on your localhost port, you can use a simple html template for that purpose.

Of course you can pack it all in a Docker file using a .yaml configuration and using docker commands you can the just run an docker container localy and run the program without the need to manually setup requirements

To make this app working on top 1000 Alexa-ranked websites you can upgrade the web scrapper script a bit to collect dailly images from all of the sites from the list so that you can train your model further

The same approach could be used for images of screens.

# Some additional approcahes

First approach: Use teseract OCR to extract text from a webpage for labeling purposes, 
                use keyBert model to extract keywords with weights for that extracted text from teseract
                use OCR and KeyBert model again to extract key words from your image
                input strings to predefined keywords and calculate the scores for all website classes and output the highest score
                
Second approach: Use teseract OCR to extract text from a webpage (use it as a web scrapping option to collect data), 
                 use a few snapshots and extract series of words for each website and label them
                 train a simple model to predict the output
                 you can either extracted keywords from whole detected text or you can use all extracted words to predict the site 
                        
Downside of this approach: OCR pretrained detectors can be a bit large to install as a dependency
