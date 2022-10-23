from matplotlib.pyplot import imshow
from paddleocr import PaddleOCR,draw_ocr
from paddleocr import draw_ocr
from pdf2image import convert_from_path
import numpy as np
import cv2
from PIL import Image
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter lang as `ch`, `en`, `french`, `german`, `korean`, japan
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = 'invoice-sample.jpg'
images = convert_from_path('BA SAT VA Mandiri - Enhancement RPA (final)-1-2-1.pdf')

for i in range(len(images)):
    
 # Convert pages as JPG in the pdf 
    image = images[i].save('page'+ str(i) +'.jpg', 'JPEG')
    namajpg= 'page'+ str(i) +'.jpg'
    img = cv2.imread(namajpg)
    # Convert pages as PNG in the pdf
    convert = Image.open(namajpg)
    namapng = 'page'+ str(i) +'.png'
    test = convert.save(namapng, 'PNG')
    img = cv2.imread(namapng)

#Resise Image 
ima = cv2.imread(namajpg)
down_width = 920
down_height = 1200
down_points = (down_width, down_height)
img = cv2.resize(ima, down_points, interpolation= cv2.INTER_LINEAR)

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 
#image = cv2.imread(img)
gray = get_grayscale(img)
thresh = thresholding(gray)
open = opening(gray)
can = canny(gray)

fromCenter = False
ROIs = cv2.selectROIs('Select ROIs', img, fromCenter)
#ROI_1 = img[ROIs[0][1]:ROIs[0][1]+ROIs[0][3], ROIs[0][0]:ROIs[0][0]+ROIs[0][2]]
#print(ROI_1)
crop_number=0 
boxes_hasil = []
txts_hasil = []
scores_hasil = []
#loop over every bounding box save in array "ROIs"
for rect in ROIs:
 x1=rect[0]
 y1=rect[1]
 x2=rect[2]
 y2=rect[3]
 x=rect[0]
 y=rect[1]
 p=rect[2]
 q=rect[3]
 koordinat=[x1, y1, x2, y2, x, y, p, q]
 koor_text = [x1,y1]
 
 print(koordinat)
        #crop roi from original image
 img_crop=img[y1:y1+y2,x1:x1+x2]
 
        #show cropped image
 #cv2.imshow("crop"+str(crop_number),img_crop)

 #save cropped image
 #cv2.imwrite("crop"+str(crop_number)+".jpeg",img_crop)
        
 crop_number+=1
#baris1 = ROIs[0]
#baris2 = ROIs[1]
#baris3 = ROIs[2]
 
 result = ocr.ocr(img_crop, cls=True)
 for line in result:
    print(line)


# draw result
  
 image = Image.open(namajpg).convert('RGB')
 for line in result:
  boxes = line[0]
 #boxes_hasil = []
  boxes_hasil.append(boxes)
  txts = line[1][0] 
 #txts_hasil = []
  txts_hasil.append(txts)
  print(txts)
  scores = line[1][1]
 #scores_hasil = []
  scores_hasil.append(scores)
 
#im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
print(txts_hasil)
im_show = draw_ocr(image, boxes_hasil, txts_hasil, scores_hasil, font_path='arial.ttf')

#print(im_show) 

im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
read_file = cv2.imread('result.jpg')
cv2.imshow('Hasil', read_file)
cv2.waitKey(0)