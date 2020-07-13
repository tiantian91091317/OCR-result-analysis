'''
1. 遍历文件夹读取文件夹中的json文件和image图片
2. 通过json信息对image图片进行切割，得到新的图片集
3. 将图片集存入到文件夹中去
'''
import os
import json
import cv2
import logging
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from crnn import config

# 遍历指定目录，显示目录下的所有文件名
count = 0
logger = logging.getLogger("split image")
CFG = config.CFG
data_path = CFG.get('crop','data_path')

# threshold when filter an image recognition result by prob offered by ali
prob_thres =0.985
# 整句的prob是每个字的prob连乘得到的
prob_map = {i: pow(prob_thres, i) for i in range(20)}

def init_logger():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()])

class DataConfig(object):
    # todo: 可配置,load_json
    pass


# 处理src_dir目录中的所有的json文件，对他进行切割
def process_folder(label_src_dir, img_src_dir, image_class, dest_dir, label_file_name ='label.txt'):
    """
    读取label文件，对相应的图片进行切割
    :param label_src_dir: label文件的根目录，下面存放着多个label文件，以图片名称命名
    :param img_src_dir: 图片文件的根目录，下面存放着多个图片
    :param image_class: 图片类型
    :param dest_dir: 生成小图的目标目录
    :param label_file_name: 生成label文件的名称
    :return:
    """
    global count
    file_names = os.listdir(label_src_dir)
    label_file = open(label_file_name,"a")
    pbar = tqdm(total=len(file_names))
    i = 0
    # print(len(file_names))
    for one_file_name in file_names:

        prefix_name, subfix_name= os.path.splitext(one_file_name)

        if (subfix_name != '.txt'):
            i += 1
            continue

        # 看看图片存在不
        image_name = prefix_name + ".jpg"
        image_full_path = os.path.join(img_src_dir, image_name)
        if not os.path.exists(image_full_path):
            logger.warning("图片路径不存在：%s",image_full_path)
            i+=1
            continue

        # logger.debug("处理文件：%s",prefix_name)
        process_one_file(label_src_dir, img_src_dir, image_class, prefix_name, dest_dir, label_file)
        i += 1
        pbar.update(i)
    label_file.close()

# 读取文件内容并打印
def process_one_file(src_dir, img_src_dir, image_class, prefix, dest_dir, label_file):

    label_full_path  = os.path.join(src_dir,prefix+".txt")
    image_full_path = os.path.join(img_src_dir,prefix+".jpg")
    image = cv2.imread(image_full_path)

    # label_reader = ProcessedLabelReader()
    label_reader = RawJsonReader()
    polygons, words = label_reader.read(label_full_path)

    if words:
        logger.debug('图片类型:%s,图片名称:%s', image_class, prefix+".jpg")
        polygons = label_reader.revert_polygons(polygons, image)
        times = 1
        try:
            for image in crop_small_images(image,polygons):
                filename = os.path.join(dest_dir, image_class + prefix + "-" + str(times) + ".jpg")
                cv2.imwrite(filename ,image)
                content = filename + " " + words[times-1] + "\n"
                label_file.write(content)
                times += 1
        except Exception:
            import traceback
            traceback.print_exc()
            logger.error('图片处理失败，跳过:%s', image_class + prefix + ".jpg")


class LabelReader(metaclass=ABCMeta):
    """
    读取标签，返回小图框
    """

    def __init__(self):
        pass

    @abstractmethod
    def read(self, label_path):
        """
        处理各种形式的标签文件，可能包括原始json（不同格式），已经处理好的标签文件
        :param label_path: 标签文件的路径
        :return:
            polygons:每个小框的四点坐标，形如[[x1,y1,x2,y2,x3,y3,x4,y4],[x1,y1,x2,y2,x3,y3,x4,y4]]
            words:每个小框内的文字，形如[组成形式, 发照日期]
            polygons 与 words 按顺序对应
        """
        pass

    @staticmethod
    def revert_polygons(polygons, image, size_threshold = 2000):
        """
        为防止调用外部接口时，传输的图片过大，所以对于图片做了缩小处理。
        如果是从raw json中解析的polygons，需要将polygons缩回原图片的尺度。
        :param size_threshold: 之前图片缩小的尺寸阈值
        :param polygons: 文本框坐标
        :param image: 原始图片
        :return: 放大后的文本框坐标
        """
        return polygons




class ProcessedLabelReader(LabelReader):
    """
    处理形如下的标签文件：
    x1,y1,x2,y2,x3,y3,x4,y4,word
    x1,y1,x2,y2,x3,y3,x4,y4,word
    """

    def read(self, label_path):
        f = open(label_path, 'r')
        labels = f.readlines()
        polygons, words = [], []
        for l in labels:
            coords = l.split(',')[:8]
            coords = [[coords[i], coords[i+1]] for i in np.arange(0,8,2)]
            polygons.append(np.array(coords, dtype='int'))
            word = ''.join(l.split(',')[8:])
            words.append(word)
        f.close()

        return polygons, words

class RawJsonReader(LabelReader):
    """
    处理形如下的json报文：
    {
    'success': True,
    'request_id': '20200318220544_fa07d87cdb5aa9ee2fa0feff8be194ab',
    'ret':
        [{'word': 'Page 1 of 1',
        'prob': 0.4809045195579529,
        'rect':
            {'angle': -3.1819396018981934,
            'top': 27.547618865966797,
            'left': 1054.3804931640625,
            'height': 24.06734275817871,
            'width': 134.67535400390625
            }
        }]
    }
    相应的阿里接口：https://market.aliyun.com/products/57124001/cmapi020020.
    html?spm=5176.12127985.1247880.6.434b4f58U47pun&innerSource=search#sku=yuncode1402000000

    """

    def read(self, label_path):
        f = open(label_path, 'r')
        content = f.read()
        try:
            json_array = json.loads(content)
        except json.decoder.JSONDecodeError:
            logger.warning('标签json解析失败，尝试恢复')
            content = self.regulate_json_string(content)
            try:
                json_array = json.loads(content)
            except json.decoder.JSONDecodeError:
                logger.error('恢复失败')
                f.close()
                return None, None

        f.close()
        return parseJson2(json_array)


    @staticmethod
    def revert_polygons(polygons, image, size_threshold=2000):
        """
        raw json 模式下需要做转换
        :param polygons:
        :param image:
        :param size_threshold:
        :return:
        """
        h,w,_ = image.shape
        if h >= size_threshold or w >= size_threshold:
            r1 = h // size_threshold + 1
            r2 = w // size_threshold + 1
            ratio = max(r1, r2)
            polygons = polygons * ratio

        return polygons

    @staticmethod
    def regulate_json_string(json_string):
        """
        用python直接保存json为txt文件，会导致json格式丢失，所以需要格式化保存的文件。
        :param json_string: raw string
        :return: regulated string
        """
        json_string = json_string.replace(': True, ', ': "True", ').replace(': False, ', ': "False", ')
        if '"' not in json_string:
            return json_string.replace('\'','"')

        try:
            ls = json_string.split('{\'word\': "')
            lsn = []
            for i, l in enumerate(ls):
                if i == 0:
                    lsn.append(l)
                    continue
                index = l.find('"')
                lsn.append(l[:index])
                lsn.append(l[index:])
            for i, l in enumerate(lsn):
                if i % 2 == 0:
                    lsn[i] = lsn[i].replace('\'', '\"')
                else:
                    lsn[i] = '{"word": "' + lsn[i]
            ls_new_str = ''.join(lsn)
        except IndexError:
            logger.warning('格式化json文件失败')
            return json_string

        return ls_new_str



def crop_small_images(img,polygons):

    cropped_images = []
    for pts in polygons:
        # crop_img = img[y:y+h, x:x+w]
        # logger.debug("子图坐标：%r",pts)
        pts_np = np.array(pts)
        pts_np = pts_np.reshape(4,2)
        # print(pts_np)
        min_xy = np.min(pts_np,axis=0)
        max_xy = np.max(pts_np,axis=0)

        # if max_xy[0] - min_xy[0] < max_xy[1] - min_xy[1]:

        # print(min_xy[0],min_xy[1],max_xy[0],max_xy[1])
        crop_img = img[min_xy[1]:max_xy[1],min_xy[0]:max_xy[0]]
        cropped_images.append(crop_img)
    return cropped_images


# 解析json结构获得坐标
def parseJson(jsonLine):
    prism_wordsInfo = jsonLine['prism_wordsInfo']
    posList = []
    wordList = []
    result = {}
    for info in prism_wordsInfo:
        dataList = []
        word = info['word']
        wordList.append(word)
        for pos in info['pos']:
            x = pos['x']
            y = pos['y']
            dataList.append(x)
            dataList.append(y)
        posList.append(dataList)
    result['pos'] = posList
    result['word'] = wordList
    return result

def parseJson2(res_data):
    if 'ret' not in res_data:
        print('图片模糊:识别为空')
        return None,None

    all_box = []
    text_arr = []
    for p in res_data['ret']:
        if not sample_filter(p):
            continue
        rect = p['rect']
        word = p['word']
        x1 = rect['left']
        y1 = rect['top']
        w = rect['width']
        h = rect['height']
        a = rect['angle']

        # 计算中心点坐标
        x = x1 + w / 2
        y = y1 + h / 2
        pts = ((x, y), (w, h), a)
        box = cv2.boxPoints(pts)  # 计算矩形的四个顶点
        box = np.int0(box)
        all_box.append(box)
        text_arr.append(word)

    return  np.array(all_box),text_arr


def sample_filter(pred, prob_thres_default=0.8):
    prob = prob_map.get(len(pred['word']), prob_thres_default)
    if pred['prob'] < prob:  # crop差图片可临时修改为 >=
        return False
    if pred['word'] == 'qrcude':  # 过滤掉二维码
        return False
    if (pred['rect']['width'] < pred['rect']['height']) & \
            (pred['rect']['angle'] > -45):  # 过滤掉竖着的文本
        return False
    return True



if __name__ == '__main__':
    init_logger()
    dest_dir = CFG.get('crop', 'dest_dir')
    label_file_name = CFG.get('crop', 'label_file_path')

    try:
        multi_data = open(data_path, 'r').readlines()
        multi_data = [d.strip().split(' ') for d in multi_data]
        multi_data = [{'class': d[0], 'label_src_path': d[1], 'img_src_path': d[2]} for d in multi_data]
        # 感觉可以改成直接读config对象
    except FileNotFoundError:
        print('数据地址有误')
        exit(-1)
    except IndexError:
        print('配置文件格式错误')
        exit(-1)

    # if os.path.exists(label_file_name):  # 控制是否更新标签文件
    #     os.remove(label_file_name)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    for d in multi_data:
        label_s, img_s, img_c = d['label_src_path'], d['img_src_path'], d['class']
        logger.debug("标签文件夹:%s, 图片文件夹:%s, 图片类型：%s", label_s, img_s, img_c)
        process_folder(label_s, img_s, img_c, dest_dir, label_file_name)

    # label_s = 'data/multi_ali/ali/debug/message'
    # img_s = 'data/multi_ali/sample/debug'
    # img_c = 'de'
    # dest_dir = 'data/debug'
    # label_file_name = 'data/debug/label.txt'
    # logger.debug("标签文件夹:%s, 图片文件夹:%s, 图片类型：%s", label_s, img_s, img_c)
    # process_folder(label_s, img_s, img_c, dest_dir, label_file_name)



