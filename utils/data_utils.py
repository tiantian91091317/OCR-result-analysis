"""
处理数据工具类。包括：
1、读取文件，按照行、列维度返回文件内容
2、单独调用crnn接口，返回结果
"""
import os
import logging
import cv2
import numpy as np



logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])
logging = logging.getLogger()

def load_file_2d(file_path, column_sep =' ', row_sep ='\n'):
    """
    读取文件，返回文件内容组成的列表。其中第0个维度为行，第1个维度为列
    :param file_path: string
    :param column_sep: 列分割符，例如' ','\t'
    :return: list
    """
    #todo: 保证维度相同
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if column_sep is None:
                return [l.strip(row_sep) for l in lines]
            else:
                return [l.strip(row_sep).split(column_sep) for l in lines]
    except FileNotFoundError:
        logging.error('文件不存在，请检查文件路径:%s', file_path)
        return None

def save_file_2d(list_2d, file_path, column_sep =' ', row_sep ='\n'):
    """
    将2dlist保存为文件
    :param list_2d:
    :param file_path:
    :param column_sep:
    :param row_sep:
    :return:
    """

    if column_sep is not None:
        lines = [column_sep.join(l) for l in list_2d]
        lines = row_sep.join(lines)
    else:
        lines = row_sep.join(list_2d)
    if os.path.exists(file_path):
        logging.warning('文件【%s】已存在，将覆盖文件', file_path)
    #todo:创建文件夹
    with open(file_path,'w') as f:
        f.write(lines)

    return True




class ImageReader(object):
    """
    在要识别的图片数量很多的情况下，批量读取图片
    """

    def __init__(self, image_list, batch_size = 128):
        """
        :param image_list: list of image paths
        :param batch_size: read by batch
        """
        self.next_index = 0
        self.over = False
        self.batch_size = batch_size
        self.image_list = image_list
        self.total_num = len(image_list)

    def next(self):
        # todo: handle exception
        images = []
        for i in range(self.batch_size):
            try:
                path = self.image_list[self.next_index + i]
            except IndexError:
                logging.info('图片全部处理完毕')
                self.over = True
                self.next_index = 0
                return images, self.next_index
            images.append(self.read_image(path))

        self.next_index += self.batch_size

        return images, self.next_index

    @staticmethod
    def read_image(path):
        try:
            im = cv2.imread(path)
            return im
        except FileNotFoundError:
            logging.warning('图片路径不存在：%s', path)
            return None
        # todo: 异常处理



if __name__ == '__main__':
    test_list = ['data/cropped_ali/baFE852374-9A9C-48ED-9B8C-82FC0AB5060CN10-2-1.jpg',
                 'data/cropped_ali/baFE852374-9A9C-48ED-9B8C-82FC0AB5060CN10-2-2.jpg',
                 'data/cropped_ali/baFE852374-9A9C-48ED-9B8C-82FC0AB5060CN10-2-3.jpg',
                 'data/cropped_ali/baFDF02970-9E50-F4A1-4687-D1A74B4D51B9N10-1-1.jpg',
                 'data/cropped_ali/baFDF02970-9E50-F4A1-4687-D1A74B4D51B9N10-1-2.jpg']
    ir = ImageReader(test_list, batch_size=2)
    while not ir.over:
        print(f'从{ir.next_index}开始读取')
        ims = ir.next()
        img_num =  sum([isinstance(l, np.ndarray) for l in ims])
        print(f'读取成功{img_num}张图片')
    print('全部处理完毕')



