import cv2
import numpy as np
from fastapi import FastAPI,Request,HTTPException
import base64

app = FastAPI()

def readb64(uri):
    # input base64 Image "uri"
    encoded_data = uri
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)

    # convert Base64 to image
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow(img)
    return img

@app.get("/")
async def root():
    return {"message": "This my api"}

@app.get("/api/gethog")
async def get_hog(request:Request):  
    item =  await request.json()
    item_str = item["img"]
    img = readb64(item_str)
    
    img_new = cv2.resize(img, (128,128), cv2.INTER_AREA)
    win_size = img_new.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    
    # Set the parameters of the HOG descriptor using the variablesdefined above
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    # Compute the HOG Descriptor for the gray scale image
    hog_descriptor = hog.compute(img_new)
    # print(hog_descriptor.tolist())
    return {"Hog": hog_descriptor.tolist()}