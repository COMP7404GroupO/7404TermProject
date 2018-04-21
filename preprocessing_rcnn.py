from __future__ import division, print_function, absolute_import
import pickle
import numpy as np 
import selectivesearch
from PIL import Image
import os.path
import skimage
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

# IOU Part 1
#首先判断两个框是否有交集
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    if if_intersect == True:  #如果有交集，整理四个顶点之间的大小关系得出相应的面积，得到交集面积
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1] 
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter

# IOU Part 2
def IOU(ver1, vertice2):
    # vertices in four points存入框的顶点集
    vertice1 = [ver1[0], ver1[1], ver1[2], ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2], vertice2[1], vertice2[3])
    if area_inter:
        area_1 = (ver1[2]-ver1[0]) *(ver1[3]-ver1[1])
        area_2 = vertice2[4] * vertice2[5] 
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False

# Clip Image
def clip_pic(img, rect):
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[x:x_1, y:y_1, :], [x, y, x_1, y_1, w, h]   #返回裁剪以后的图像，各个顶点以及宽高

# Read in data and save data for Alexnet
def load_train_proposals(datafile, num_clss, threshold = 0.5, svm = False, save=False, save_path='dataset/', testrate = 0.1):
    img_path = 'JPEGImages/'
    train_list = open(datafile,'r')
    labels = []
    images = []
    labelstest = []
    imagestest = []
    x = 0
    y = 0
    w = 0
    h = 0
    n = 0
    index = 0
    iou_val = 0
    datasetnum = 1
    datasetcon = 0
    tmp = []
    img = []
    img_lbl = []
    regions = []
    candidates = set()
    ref_rect = []
    ref_rect_int = []


    for line in train_list:          #对于行
        tmp = line.strip().split(' ')
        # tmp0 = image address   #每一行中的组成
        # tmp1 = label
        # tmp2 = rectangle vertices
        img = skimage.io.imread(img_path+tmp[0])  #读地址
        #检测图形中的物体位置是否正确
        '''
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(img)
        showimgt =  tmp[2].split(',')
        showimg = [int(i) for i in showimgt]
        real = mpatches.Rectangle((showimg[0], showimg[1]), showimg[2]-showimg[0],showimg[3]-showimg[1], fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(real)
        plt.show()
        '''
        positiveInTest = 0#record the positive image in testset
        img_lbl, regions = selectivesearch.selective_search(
                               img, scale = 15, sigma=0.5, min_size= 80)  #备选的划分方式
        candidates.clear()
        #candidates = set()  #初始化备选集合
	#print(tmp[0])
        index = int(tmp[1]) #将label转换为数值类型
        ref_rect= tmp[2].split(',')
        ref_rect_int = [int(i) for i in ref_rect]
        proposal_img = []
        proposal_vertice = []
        array_img = []
        resize_img = []
        img_float = []
        negsamples = 0
        for r in regions:
	    # excluding same rectangle (with different segments)
            if r['rect'] in candidates: #python 字典
                continue
#            if r['size'] < 45:
#                continue
	    # resize to 227 * 227 for input
            proposal_img, proposal_vertice = clip_pic(img, r['rect']) #开始按照rect中存储的划分方式裁剪 返回图像与顶点
	    # Delete Empty array
            if len(proposal_img) == 0:
	            continue
            # Ignore things contain 0 or not C contiguous array
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
	            continue
            # Check if any 0-dimension exist
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            array_img = Image.fromarray(proposal_img) 
            #resize 
            resize_img = resize_image(array_img, 227, 227)
            img_float = pil_to_nparray(resize_img)
            candidates.add(r['rect'])          #加入已经存在的划分方式
            n = random.randint(1,10)
            # IOU
            iou_val = IOU(ref_rect_int, proposal_vertice)
            # labels, let 0 represent default class, which is background
            
	    #预处理标签
            if svm == False:     
                label = np.zeros(num_clss+1)  #对于cnn采用one_hot的存储方式
                if iou_val < threshold: #若选择的图片为背景，则label 0 = 1， 若选择的图片为为分类则对应的分类=1
                    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6)) 
                    ax.imshow(img)
                    rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                    plt.show()
                    label[0] = 1
                else:
                    label[index] = 1
                if label[index] == 1:#the positive regions
                    if n <= 10-(testrate*10):
                        images.append(img_float) #转换float32并加入训练集中
                        labels.append(label)
                    else:
                        imagestest.append(img_float)
                        labelstest.append(label)
                        positiveInTest = positiveInTest+1
                    if positiveInTest < 1:#to confirm at least 1 positive in test set
                        imagestest.append(img_float)
                        labelstest.append(label)
                        positiveInTest = positiveInTest+1
                elif negsamples < 50:
                    if n <= 10-(testrate*10):
                        images.append(img_float) #转换float32并加入训练集中
                        labels.append(label)
                    else:
                        imagestest.append(img_float)
                        labelstest.append(label)
                    negsamples = negsamples + 1

            else:
                if iou_val <threshold and negsamples < 50:
                    images.append(img_float)
                    labels.append(0)
                    negsamples = negsamples + 1
                else:
                    images.append(img_float)
                    labels.append(index)
        print(line)       
        #print(len(images))
        
        #output the data
        print(line)
        datasetcon = datasetcon + 1
        if (datasetcon%10)==0 :
            pickle.dump((images, labels, imagestest, labelstest), open(save_path+str(datasetnum)+'dataset.pkl', 'wb'))
            datasetnum = datasetnum + 1
            images.clear()
            labels.clear()
            imagestest.clear()
            labelstest.clear()
        
    
    if len(images) > 0 :
        pickle.dump((images, labels, imagestest, labelstest), open(save_path+str(datasetnum)+'dataset.pkl', 'wb'))
        
    return images, labels , imagestest, labelstest   #返回所有image和lable的集合

def load_from_pkl(dataset_file):
    X, Y, imagestest, labeltest  = pickle.load(open(dataset_file, 'rb'))
    return X,Y,imagestest,labeltest



if __name__ == '__main__':
    
    
    printimage, printlabel, printimagetest, printlabeltest = load_train_proposals('butterfly_list.txt',94,threshold=0.5,save=True)
    print(len(printimage),len(printlabel),len(printimagetest),len(printlabeltest))


    #get the validation set
    
    testsetimage = []
    testsetlabel = []
    listings = os.listdir('dataset')
    for train_file in listings:
        print(train_file)
        X,Y,imagestest,labelstest = load_from_pkl('dataset/'+train_file)
        testsetlabel = testsetlabel + labelstest
        testsetimage = testsetimage + imagestest
    print(len(testsetimage))
    print(len(testsetlabel))
    pickle.dump((testsetimage, testsetlabel), open('dataset/valdataset.pkl', 'wb'))
    #resize the validate set
    i=0
    testimagerestore = []
    testlabelrestore = []

    i=0
    testsetimage,testsetlabel = pickle.load(open('dataset/valdataset.pkl', 'rb'))
    for j in range(len(testsetimage)):
        if testsetlabel[j][0] == 0:
           i=i+1
    print(i)
    print(np.array(testsetimage).shape,np.array(testsetlabel).shape)

    pickle.dump((testimagerestore, testlabelrestore), open('valdataset.pkl', 'wb'))
   
   #testing 
    '''
    var = 0
    for i in printlabel:
        if i[79]==1:
            var = var+1
    print(var)
    var = 0
    for i in printlabeltest:
        if i[79]==1:
            var = var+1
    print(var)
    print(np.array(printimage).shape,np.array(printlabel).shape,np.array(printimagetest).shape,np.array(printlabeltest).shape)
    
    '''
    #load from the data file which have been stored
    '''
    printimage, printlabel, printimagetest, printlabeltest = load_from_pkl('dataset.pkl')
    print(len(printimage),len(printlabel),len(printimagetest),len(printlabeltest))
    var = 0
    for i in printlabel:
        if i[82]==1:
            var = var+1
    print(var)
    var = 0
    for i in printlabeltest:
        if i[82]==1:
            var = var+1
    print(var)
    print(np.array(printimage).shape,np.array(printlabel).shape,np.array(printimagetest).shape,np.array(printlabeltest).shape)
    '''