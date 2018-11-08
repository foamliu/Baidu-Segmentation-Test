import base64

import cv2 as cv
import numpy as np
from aip import AipBodyAnalysis

""" 你的 APPID AK SK """
APP_ID = '14724666'
API_KEY = '9RgncDsMemqT2vFNlPGWsVRM'
SECRET_KEY = 'RqQh1PEKh3REH0cNgBwKCBZGRzeaitsP'

client = AipBodyAnalysis(APP_ID, API_KEY, SECRET_KEY)

""" 读取图片 """


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


image = get_file_content('example.png')

""" 调用人像分割 """
dict = client.bodySeg(image)
labelmap = dict['labelmap']
print(labelmap)
ret = base64.b64decode(labelmap)
print(ret)

with open('labelmap.png', 'wb') as fp:
    fp.write(ret)

mask = cv.imread('labelmap.png')
mask = cv.resize(mask, (640, 360), cv.INTER_NEAREST)
print(mask.shape)
print(np.max(mask))
print(np.min(mask))
mask = mask * 255


img = cv.imread('example.png')
img = img * 0.5 + mask * 0.5
img = img.astype(np.uint8)

cv.imwrite('img_merged.png', img)
cv.imwrite('label_bw.png', mask)
