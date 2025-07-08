#更新时间:20220408
import os
import sys
import cv2
import glob
import time
import math
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
#import pynvml

try:
    import torch
    IF_IMPORT_TORCH=True
except ImportError:
    IF_IMPORT_TORCH=False

try:
    from pycocotools import mask as coco_mask
except ImportError:
    pass

try:
    from scipy.ndimage.morphology import distance_transform_edt
except ImportError:
    pass

#########################通用函数库##################################



#显示所有numpy信息
def PrintInfo(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        print('shape : {} ; dtype : {} ; max : {} ; min : {}'.format(i.shape,i.dtype,i.max(),i.min())  )

def PrintSet(x):
    if not isinstance(x,list):
        x=[x]
    for i in x:
        set_i=set(i.flat)
        print('set : {} ; num : {}' .format(set_i, len(set_i)))
    


#显示多个图像
def ImagesShow(imgs,names=None,figsize=None,if_axis=True,if_full_screen=False,if_split=False,fontsize=20,suptitle=None):
    global IF_IMPORT_TORCH
    if not isinstance(imgs,list):imgs=[imgs]
    for index in range(len(imgs)):
        if isinstance(imgs[index],torch.Tensor) and IF_IMPORT_TORCH:
            imgs[index]=imgs[index].cpu().numpy() 
            if imgs[index].ndim==3:
                imgs[index]=imgs[index].transpose([1,2,0])

    if if_split:
        for index,img in enumerate(imgs):
            if if_full_screen and figsize is None :plt.get_current_fig_manager().full_screen_toggle()
            if not if_axis:plt.axis('off')
            plt.figure(index if names is None else names[index],figsize=figsize)
            if suptitle is not None:plt.suptitle(suptitle,fontsize=fontsize)
            plt.imshow(img)
        plt.show()
    else:
        layout={1:[1,1],2:[1,2],3:[1,3],4:[2,2],5:[2,3],6:[2,3],7:[2,4],8:[2,4],9:[3,3],10:[2,5],11:[3,4],12:[3,4],13:[3,5],14:[3,5],15:[3,5],16:[4,4]}
        plt.figure(figsize=figsize)
        if suptitle is not None:plt.suptitle(suptitle,fontsize=fontsize)
        if if_full_screen and figsize is None:plt.get_current_fig_manager().full_screen_toggle()
        for index,img in enumerate(imgs):
            ax=plt.subplot(layout[len(imgs)][0],layout[len(imgs)][1],index+1)
            if not if_axis:ax.axis('off')
            if names is not None:ax.set_title(names[index],fontsize=fontsize)
            plt.imshow(img)
        plt.show()


#读取多个图像
def ImagesRead(paths,mode='src',size=None):
    if isinstance(paths,str):
        paths=[paths]
    imgs=[]
    for path in paths:
        img=Image.open(path) if mode=='src' else Image.open(path).convert(mode)
        img=np.array(img)
        if size is not None:
            img=cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
        imgs.append(img)
    if len(imgs)!=1:
        return imgs
    else:
        return imgs[0]





#显示 img 和 gt 合成的图像 ，混合标注
def ImgGtMergeShow(img,gt,if_show=True):
    if img.ndim==2: img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    if gt.max()==1:gt=gt*255
    gt=gt[:,:,np.newaxis]
    gt_zero=np.zeros(gt.shape,dtype=np.uint8)
    gt_RGB=np.concatenate( (gt,gt_zero,gt_zero),axis=2 )  
    img_merge=ImageMerge(img,0.5,gt_RGB,0.5)
    if if_show:
        plt.imshow(img_merge)
        plt.show()
    return img_merge



PAL=[0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128, 64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128, 128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128, 192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128, 64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64, 0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64, 128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64, 0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64, 64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192, 192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128, 160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0, 224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64, 0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192, 128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64, 128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32, 128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128, 192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0, 192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64, 160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96, 64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192, 96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32,0, 0, 160, 0, 128, 160, 0, 0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32, 0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192, 160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128, 96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0, 192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32, 64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0, 160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160, 64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128, 96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192, 128,224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96, 192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32, 160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160, 128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32,128, 224, 32, 128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160, 224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0, 224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224, 128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32, 32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32, 64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192, 224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96, 192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64, 96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 224, 224, 192]
def ImageSave(path , img , if_pal=False):
    img=Image.fromarray(img)
    if if_pal:img.putpalette(PAL)
    img.save(path)

#计时器
class MyClock(object):
    def __init__(self):
        self.start_time=time.perf_counter()
    def start(self):
        self.start_time=time.perf_counter()
    def end(self,i=1):
        print('{}th :{} s'.format(i,(time.perf_counter() - self.start_time)))
        self.start_time=time.perf_counter()  
mc = MyClock()


'''
def PrintGpuUsage(gpu_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print('gpu_{} : {} MB'.format(gpu_id,int(meminfo.used/1024/1024)) )
'''

#记录器
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'w')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass


def SetLogFile(file_path='log'):
    sys.stdout = Logger(file_path, sys.stdout)

#生成文件列表
def GeneFileList(path, output_file=None, if_sorted=True, if_suffix=False, if_show=False):
    ids=glob.glob(path)
    ids=[id.split('/')[-1]for id in ids]
    if if_suffix==False:
        ids=[id.split('.')[-2] for id in ids]
    if if_sorted:
        ids=sorted(ids)
    if output_file!=None:
        with open(output_file,'w') as f:
            for id in ids:
                f.write(id+'\n')  
    if if_show: 
        for id in ids:            
            print(id)
    return ids

#合并txt文件
def MergeTxtFiles(files, output_file=None, if_sorted=True):
    lines_all=[]
    for file in files:
        with open(file) as f:
            lines = f.read().splitlines()
            lines_all+=lines
    if if_sorted:
        lines_all=sorted(lines_all)
    if output_file!=None:
        with open(output_file, 'w') as f:
            for i in lines_all:
                f.write(i+'\n')              
    return lines_all

#从txt获取list
def GetListFromTxt(file, if_sorted=False):
    with open(file) as f:
        lines = f.read().splitlines()
    if if_sorted:
        lines=sorted(lines)       
    return lines

#将list写入file
def SetListIntoTxt(file,lines):
    with open(file,'w') as f:
        for line in lines: 
            f.write('{}\n'.format(line))

#将arr写入file
def SetArrIntoTxt(file,arr,interval='  ',dp=3, zfill=None):
    with open(file,'w') as f:
        for v in arr:
            v_str=str(v) if dp is None else ['{:.0f}'.format(v),'{:.1f}'.format(v),'{:.2f}'.format(v),'{:.3f}'.format(v)][dp]
            if zfill is not None:v_str=v_str.zfill(zfill)
            v_str+=interval
            f.write(v_str)

#移除和创建文件夹
def RemoveAndCreateFolder(folders):
    if not isinstance(folders,list):
        folders=[folders]
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)



#将lists写入csv
def SetDataIntoCsv(file,data):
    import csv
    if not file.endswith('.csv'):
        file=file+'.csv'
    with open(file,'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)


#########################图像处理库##################################


#图像混合（同为RGB 或 灰度 ）
def ImageMerge(img1,weight1,img2,weight2):
    result=img1*weight1+img2*weight2
    result[result<0]=0
    result[result>255]=255
    return result.astype(np.uint8)


#图像平移
def ImgTranslate(image, x, y):
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # 返回转换后的图像
    return shifted

#图像旋转
def ImgRotate(image, angle, center=None, scale=1.0,mode=None ,if_get_mat=False):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w // 2 ,h // 2)
        #print(center)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    if if_get_mat:
        return M
    #print(M)
    if mode==None:
        mode=cv2.INTER_LINEAR if len(image.shape)==3 else cv2.INTER_NEAREST #灰度图像时用nearest插值
    rotated = cv2.warpAffine(image, M, (w, h),flags=mode)
    # 返回旋转后的图像
    return rotated

#图像旋转(逆时针)，不裁剪
def ImgRotate_NoCrop(img, degree,mode=None,pad=0):
    height,width=img.shape[:2]
    #旋转后的尺寸
    heightNew=int(width*math.fabs(math.sin(math.radians(degree)))+height*math.fabs(math.cos(math.radians(degree))))
    widthNew=int(height*math.fabs(math.sin(math.radians(degree)))+width*math.fabs(math.cos(math.radians(degree)))) 
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2 +pad
    matRotation[1,2] +=(heightNew-height)/2 +pad
    if mode==None:
        mode=cv2.INTER_LINEAR if len(img.shape)==3 else cv2.INTER_NEAREST #灰度图像时用nearest插值
    #mode=cv2.INTER_LINEAR
    imgRotation=cv2.warpAffine(img,matRotation,(widthNew+1 +pad*2 ,heightNew+1 +pad *2),flags=mode)
    return imgRotation


#rect 为(x,y,w,h)  shape 是 h,w
def CropRectInShape(rect,shape):
    pt_lt=np.array( [rect[0],rect[1]] )
    pt_rb_plus =np.array( [rect[0]+rect[2],rect[1]+rect[3]])

    pt_lt=np.maximum(pt_lt, np.array([0,0]))
    pt_lt=np.minimum(pt_lt, np.array([shape[1],shape[0]]))

    pt_rb_plus=np.maximum(pt_rb_plus, np.array([0,0]))
    pt_rb_plus=np.minimum(pt_rb_plus, np.array([shape[1],shape[0]]))

    rect=[ pt_lt[0] , pt_lt[1] ,  pt_rb_plus[0]-pt_lt[0] , pt_rb_plus[1]-pt_lt[1] ]
    return rect

#rect 为(x,y,w,h)  relax 为扩张边界
def ExpandRect(rect,relax):

    pt_lt=np.array( [rect[0],rect[1]] )
    pt_rb_plus =np.array( [rect[0]+rect[2],rect[1]+rect[3]])

    pt_lt-=np.array(relax)
    pt_rb_plus+=np.array(relax)

    rect=[ pt_lt[0] , pt_lt[1] ,  pt_rb_plus[0]-pt_lt[0] , pt_rb_plus[1]-pt_lt[1] ]
    return rect



def CropImgFromGt(img,gt,relax_pixels=-1):
    if relax_pixels !=-1:
        bbox= cv2.boundingRect(gt)
        bbox=ExpandRect(bbox,relax=relax_pixels)
        bbox=CropRectInShape(bbox,gt.shape)
    else:
        bbox=(0,0,gt.shape[1],gt.shape[0])

    img_new=img[bbox[1]:bbox[1]+bbox[3] , bbox[0]:bbox[0]+bbox[2] , :]
    gt_new=gt[bbox[1]:bbox[1]+bbox[3] , bbox[0]:bbox[0]+bbox[2]]

    return img_new,gt_new,bbox


#获取根据bbox生成的mask 
def MaskFromBbox(bbox,shape):
    mask=np.zeros(shape).astype(np.uint8)
    mask[bbox[1]:bbox[1]+bbox[3] ,bbox[0]:bbox[0]+bbox[2]  ]=1
    return mask



#重新设置图像尺寸
def ImagesResize(imgs,dsize,interpolation=cv2.INTER_LINEAR):
    if not isinstance(imgs,list):
        imgs=[imgs]
    result=[]
    for img in imgs:
        result.append( cv2.resize(img,dsize,interpolation=interpolation) )
    return result

#将0-255的标注图转为伪彩色
def LabelToRgb(label):
    rgb=np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)
    index_set=set(label.flat)
    for index in index_set:
        rgb[label==index]=PAL[index*3:index*3+3]
    return rgb

#拼图 img_arrs [RxCxHxW,D]  gaps [int or float]
def Jigsaw(img_arrs,gaps=0.05,gap_color=255):
    img_arrs=np.array(img_arrs)
    if img_arrs.ndim==4:img_arrs=img_arrs[:,:,:,:,None]
    R,C,H,W,D=img_arrs.shape
    if not isinstance(gaps,(list,tuple)):gaps=[gaps,gaps]
    GR = gaps[0] if isinstance(gaps[0],int) else int(gaps[0]*H)
    GC = gaps[1] if isinstance(gaps[1],int) else int(gaps[1]*W)
    jigsaw_result=np.ones((R*H+(R-1)*GR,C*W+(C-1)*GC,D),dtype=np.uint8)
    jigsaw_result[:,:,:]=gap_color
    for r in range(R):
        for c in range(C):
            jigsaw_result[(H+GR)*r:(H+GR)*r+H,(W+GC)*c:(W+GC)*c+W,:]=img_arrs[r,c]
    return jigsaw_result[:,:,0] if D==1 else jigsaw_result


#编解码图像
def EncodeMask(mask):
    return coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
def DecodeMask(encoded_info):
    return coco_mask.decode(encoded_info)






#########################交互分割##################################

def IoU(pred,gt):
    return ((pred==1)&(gt==1)).sum()/(((pred==1)|(gt==1))&(gt!=255)).sum()

def CheckIisInfos(file,path='/home/frazer/FrazerLin/Dataset/Annotation'):
    model,dataset,list_file,_=Path(file).stem.split('~')
    path=Path(path)/dataset
    infos=np.load(file,allow_pickle=True).item()
    #ids=GetListFromTxt(path/'list'/(list_file+'.txt'))
    ids=list(infos.keys())
    for id in ids:
        gt=ImagesRead(str(path/'gt'/(id+'.png')))
        ious_src=[IoU(DecodeMask(pred),gt) for pred in infos[id]['preds']]
        ious_record=infos[id]['ious']
        delta=np.sum(np.array(ious_src)-np.array(ious_record))
        assert delta<0.01 
    print('Check OK!')


def ParseIisInfos(file,mode='all',path='/home/frazer/FrazerLin/Dataset/Annotation',if_show=True):
    model,dataset,list_file,_=Path(file).stem.split('~')
    meta=[model,dataset,list_file]

    infos=np.load(file,allow_pickle=True).item()
    if mode=='all':
        mode=['noc_miou','@0.85','@0.90']
    elif not isinstance(mode,list):
        mode= [mode]
    
    result=[]

    if 'noc_miou' in mode:
        noc_miou=np.mean(np.array([infos[k]['ious'] for k in infos]),axis=0)
        result.append(noc_miou)
        if if_show: print('noc_miou : ',noc_miou)
        
    for m in mode:
        if '@' in m:
            miou_target=float(m[1:])
            nocs=[]
            for k in infos:
                ious=np.array(infos[k]['ious'])
                nocs.append(np.argmax(ious>=miou_target) if (ious >=miou_target).any() else (len(ious)-1))
            NoC=np.mean(np.array(nocs))
            result.append(NoC)
            if if_show: print(m,':',NoC)

    return (result[0] if len(result)==1 else result)

def SetIisCurveIntoTxt(file,curves,interval='  ',dp=3):
    if not file.endswith('.txt'):file+='.txt'
    with open(file,'w') as f:
        for curve in curves:
            name,data=curve
            f.write('{:<15s}'.format(name))
            for idx,v in enumerate(data):
                v_str=str(v) if dp is None else ['{:.0f}'.format(v),'{:.1f}'.format(v),'{:.2f}'.format(v),'{:.3f}'.format(v)][dp]
                if idx!=(len(data)-1): v_str+=interval
                f.write(v_str)
            f.write('\n')



def DrawIisCurveFig(file,save_file,names=None,colormap=None,xlim=(0,20),ylim=(0,1),xticks=range(0,21,2),yticks=np.arange(0,1.00001,0.1)):
    #read
    with open(file) as f: lines=f.readlines()
    data_ref={line.strip().split()[0]:[float(v) for v in line.strip().split()[1:]] for line in lines}

    if names is None : 
        names=[line.strip().split()[0] for line in lines]
        nicknames=names
    elif isinstance(names,list):
        names=names
        nicknames=names
    elif isinstance(names,dict):
        nicknames=list(names.values())
        names=list(names.keys())
        
    #colormap
    colormap_default=['green','blue', 'red', 'gold','gray', 'peru', 'purple',  'cyan', 'orange', 'lime']
    if colormap is None:
        colormap={ n:c  for n,c in zip(names,colormap_default)} 
    else:
        colormap_remain= [c for c in colormap_default if c not in colormap.values()]
        names_remain=[n for n in names if n not in colormap.keys()]
        colormap.update({ n:c for n,c in zip(names_remain,colormap_remain)})
    colormap[names[-1]]='black'

    #draw
    plt.figure(figsize=(7, 5.4))
    plt.xticks(xticks,fontsize=13)
    plt.yticks(yticks,fontsize=13)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.grid(True, c='lightgrey')
    plt.xlabel('Number of clicks', fontsize=16) 
    plt.ylabel('mIoU score', fontsize=16)  

    plt_lines = []

    for name in names:
        plt_line, = plt.plot([int(i) for i in np.linspace(0, 20, num=21)], data_ref[name], linewidth=3, color=colormap[name])
        plt_lines.append(plt_line)
        
    plt.legend(plt_lines, nicknames, loc = 'lower right',fontsize=20)  

    plt.savefig(save_file, format='pdf',bbox_inches = 'tight')   



def ResizePointMask(img, size):
    (h, w) = img.shape
    if not isinstance(size, tuple): size=( int(w*size), int(h*size) )
    M=np.array([[size[0]/w,0,0],[0,size[1]/h,0]])
    pts_xy=np.argwhere(img==1)[:,::-1]
    pts_xy_new= np.dot( np.insert(pts_xy,2,1,axis=1), M.T).astype(np.int64)
    img_new=np.zeros(size[::-1],dtype=np.uint8)
    img_new[pts_xy_new[:,1],pts_xy_new[:,0]]=1
    return img_new


WHITE,BLACK=(255,255,255),(0,0,0)
RED,GREEN,BLUE=(255,0,0),(0,255,0),(0,0,255)
CYAN,MAGENTA,YELLOW=(0,255,255),(255,0,255),(255,255,0)
PURPLE=(160,32,240)

#special_clicks = [(idx,color),...]
def ImgPredClickMerge(img,pred,clicks=[],click_radius=9,click_border=2,pred_border=2,resize=None,special_clicks=None,pred_border_color=WHITE):
    if img.ndim==2:img=np.expand_dims(img,2).repeat(3,2)
    assert (pred.ndim==2 and img.ndim==3)
    clicks=np.array(clicks)
    if resize is not None:
        if isinstance(resize,int):resize=(resize,resize)
        src_size=pred.shape[::-1]
        img=cv2.resize(img,tuple(resize),interpolation=cv2.INTER_LINEAR)
        pred=cv2.resize(pred,tuple(resize),interpolation=cv2.INTER_NEAREST)
        if len(clicks)>0:
            clicks= np.minimum(np.int64(clicks*(resize[0]/src_size[0], resize[1]/src_size[1],1)+(0.5,0.5,0)),(resize[0]-1,resize[1]-1,1))

    pred_mask=np.uint8(np.expand_dims(pred,2).repeat(3,2)*YELLOW)
    merge=cv2.addWeighted(img,0.7,pred_mask,0.3,0)

    if pred_border>0:
        contours,_=cv2.findContours(pred,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(merge,contours,-1,pred_border_color,pred_border)
    for pt in clicks:
        cv2.circle(merge,tuple(pt[:2]),click_radius,[BLUE,RED][pt[2]],-1)
        cv2.circle(merge,tuple(pt[:2]),click_radius,WHITE,click_border)

    if special_clicks is not None:
        for pt_idx,color in special_clicks:
            cv2.circle(merge,tuple(clicks[pt_idx][:2]),click_radius,color,-1)
            cv2.circle(merge,tuple(clicks[pt_idx][:2]),click_radius,WHITE,click_border)

    return merge


def GenePointMask(points,ref_of_size):
    size=ref_of_size.shape[:2][::-1] if isinstance(ref_of_size,np.ndarray) else tuple(ref_of_size[:2])
    mask=np.zeros(size[::-1]).astype(np.uint8)
    if len(points)!=0: 
        points=np.array(points)
        mask[points[:,1], points[:,0]]=1
    return mask


def GeneClickMap(mask,mode='gauss-10',max_dist=255):
    dist_map =np.float64(np.ones(mask.shape))*max_dist if (mask==0).all() else np.minimum(distance_transform_edt(1-mask), max_dist)
    if mode.startswith('gauss'):
        click_map=np.exp(-2.772588722*(dist_map**2)/(float(mode.split('-')[1])**2))
    elif mode=='dist':
        click_map=1.0-dist_map/max_dist
    elif mode=='src':
        click_map=dist_map
    return click_map


#########################机器学习与数理统计库##################################
#stat =统计表 ，每行是一个数据，每列对应一个属性
#sort_mode 排序列维度,如果是0则不排序  1正序，-1 反序
def StatsShow(stats_list,show_dim=0,sort_mode=0):

    if sort_mode!=0:
        for stat_key in stats_list.keys():
            stat=stats_list[stat_key]
            indices_tmp= np.argsort( stat[:,show_dim] )[::sort_mode]
            stats_list[stat_key]=stat[indices_tmp]

    if show_dim!=-1:
        plt.figure()
        for stat_key in stats_list.keys():
            plt.plot( stats_list[stat_key][:,show_dim])
        plt.show()

    return stat
    # stat[:,:]=0
    # print(stat)



#########################深度学习库##################################
def UnNormPic(img,mean=(0., 0., 0.), std=(1., 1., 1.)):
    img=np.squeeze(img)
    if img.ndim==3:
        return  ((img*std+mean)*255).astype(np.uint8)
    else:
        return (img*255).astype(np.uint8)

def ToNumpy(tensor):
    if tensor.dim()==3:
        if tensor.shape[0]==1:
            return tensor.numpy()[0,:,:]
        else:
            return tensor.numpy().transpose([1,2,0])
    else:
        return tensor.numpy()


def ToUnNormNpPic(tensor,mean=(0., 0., 0.), std=(1., 1., 1.)):
    return  UnNormPic(ToNumpy(tensor),mean,std)

#return (img_tensor.numpy().transpose([1,2,0])*255).astype(np.uint8)


#########################测试矩阵##################################


def continuous_square(side=2,start=1):
    return np.arange(side**2).reshape(side,side)+1

cs2=continuous_square(2)
cs3=continuous_square(3)
cs4=continuous_square(4)


if __name__ == '__main__':
    pass
    import os
    #print(IF_IMPORT_TORCH)
    #print(os.path.dirname('./result_lz'))
    file='./result_lz/DOS~GrabCut~val~infos.npy'
    CheckIisInfos(file)
    ParseIisInfos(file)



    

