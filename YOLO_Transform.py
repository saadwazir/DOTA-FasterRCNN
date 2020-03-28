import dota_utils as util
import os
import numpy as np
from PIL import Image


def convert(img_w, img_h,x,y,w,h):
    box = np.zeros(4)
    dw = 1./img_w
    dh = 1./img_h
    x = x/dw
    w = w/dw
    y = y/dh
    h = h/dh
    box[0] = x-(w/2.0)
    box[1] = x+(w/2.0)
    box[2] = y-(h/2.0)
    box[3] = y+(h/2.0)

    #box[0] = box[0] / img_w
    #box[1] = box[1] / img_w
    #box[2] = box[2] / img_h
    #box[3] = box[3] / img_h

    return (box)

## trans dota format to format YOLO(darknet) required
def dota2darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
    """
    filelist = util.GetFileFromThisRootDir(txtpath)
    for fullname in filelist:
        objects = util.parse_dota_poly(fullname)
        name = os.path.splitext(os.path.basename(fullname))[0]
        print (name)
        img_fullname = os.path.join(imgpath, name + '.png')
        img = Image.open(img_fullname)
        img_w, img_h = img.size
        #print (img_w,img_h)
        with open(os.path.join(dstpath, name + '.txt'), 'w') as f_out:
            for obj in objects:
                poly = obj['poly']
                bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
                if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                    continue
                if (obj['name'] in extractclassname):
                    id = obj['name']
                else:
                    continue
                bbox_con = convert(img_w, img_h, bbox[0], bbox[1], bbox[2], bbox[3] )
                outline = str(name) + '.png' + ',' + ','.join(list(map(str, bbox_con))) + ',' +  str(id)
                f_out.write(outline + '\n')
                #print(bbox[0], '--', bbox_con[0])
    print ("-- ALL Done --")
    
    
if __name__ == '__main__':
    ## an example
    dota2darknet('dota-dataset-split/images',
                 'dota-dataset-split/labelTxt',
                 'dota-dataset-split/labels-un',
                 util.wordname_15)

