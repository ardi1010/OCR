from easyocr import Reader
import argparse
import cv2
from pdf2image import convert_from_path
from PIL import Image
from alignment.align_images import align_images

def cleanup_text(text):
 # Menghapus teks non-ASCII sehingga dapat menuliskan teks pada gambar menggunakan OpenCV
 return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# Membuat Parser Argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
 help="path to input image to be OCR'd")
ap.add_argument("-l", "--langs", type=str, default= 'id',
 help="comma separated list of languages to OCR")
ap.add_argument("-g", "--gpu", type=int, default=-1,
 help="whether or not GPU should be used")
args = vars(ap.parse_args())
print("[INFO] OCR'ing input image...")
# Menyimpan Pdf dengan convert_from_path function
images = convert_from_path('PT Waskita Karya (Persero) Tbk..pdf')

for i in range(len(images)):
    
    # Mengubah halaman pada pdf menjadi JPG 
    image = images[i].save('page'+ str(i) +'.jpg', 'JPEG')
    namajpg= 'page'+ str(i) +'.jpg'
    img = cv2.imread(namajpg)
    # Mengubah halaman pada pdf menjadi PNG
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


templates = cv2.imread(namapng)
down_width = 920
down_height = 1200
down_points = (down_width, down_height)
template = cv2.resize(templates, down_points, interpolation= cv2.INTER_LINEAR)
# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)
#Using ROIS in OCR
fromCenter = False
ROIs = cv2.selectROIs('Select ROIs', img, fromCenter)
#ROI_1 = img[ROIs[0][1]:ROIs[0][1]+ROIs[0][3], ROIs[0][0]:ROIs[0][0]+ROIs[0][2]]
#print(ROI_1)
crop_number=0 

#loop over every bounding box save in array "ROIs"
for rect in ROIs:
 x1=rect[0]
 y1=rect[1]
 x2=rect[2]
 y2=rect[3]
 koordinat=[x1, y1, x2, y2]
 koor_text = [x1,y1]
 aligned = align_images(img, template)
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
 reader = Reader(['id','en'], gpu=args["gpu"] > 0)
 results = reader.readtext(img_crop, detail=1)
 for (bbox, text, prob) in results:
 # display the OCR'd text and associated probability
    print("[INFO] {:.4f}: {}".format(prob, text))
    text = cleanup_text(text)
    #for line in text.split("\n"):
  # if the line is empty, ignore it
     #    if len(line) == 0:
      #    continue
  # line contains any of the filter keywords (these keywords
  # are part of the form itself and should be ignored)
         #lower = line.lower()
         #count = sum([lower.count(x) for x in loc.filter_keywords])
  # if the count is zero then we know we are not examining a
  # text field that is part of the document itself (ex., info,
  # on the field, an example, help text, etc.)
         #if count == 0:
   # update our parsing results dictionary with the OCR'd
   # text if the line is not empty
          #parsingResults.append((loc, line))
 # unpack the bounding box
 
 # cleanup the text and draw the box surrounding the text along
 # with the OCR'd text itself
 
 #cv2.rectangle(img_crop, , (0, 255, 0), 2)
   
    
    cv2.rectangle(img, koordinat, (0, 255, 0), 2)
    cv2.putText(img, text, koor_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_8, False)
# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)