"""
自己的 crnn 模型与标注（或其他公司模型）进行比较。
产出：
1、crnn 模型准确率
2、crnn 识别错误的字符，及相应的GT（待进行）

"""

import time
import logging
import re
from ocr import own_ocr
from utils.data_utils import ImageReader, load_file_2d, save_file_2d
from utils.compare_utils import FileComparator
from nlp import correct_call
import numpy as np
import shutil

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG,
    handlers=[logging.StreamHandler()])
logging = logging.getLogger()


def recognize(label_file, output_file, url, limit=0):
    image_list = load_file_2d(label_file)
    if limit:
        _b, _t = os.path.splitext(label_file)
        label_file = _b + '_0' + _t  # file0:data/test_0.txt
        image_list = [p for p in image_list if 3 < len(p[1]) < 13]
        # np.random.shuffle(image_list)
        # image_list = image_list[:limit]
        save_file_2d(image_list, label_file)
    image_list = [p[0] for p in image_list]
    image_num = len(image_list)

    image_reader = ImageReader(image_list, batch_size=32)
    results = []
    start = time.time()

    while not image_reader.over:
        images, next_index = image_reader.next()
        if not images: break

        words = own_ocr.crnn(images, url)
        assert len(images) == len(words)
        results += words
        logging.info('开始从第【%s】张图片进行识别',next_index)

    end = time.time()
    time_per_image = round((end - start)/image_num, 4)
    logging.info('识别【%d】张图片用时【%f】秒', image_num, end-start)
    logging.info('平均每张图片用时【%f】秒',time_per_image)

    results = [[p, w] for p, w in zip(image_list, results)]
    save_file_2d(results, output_file)

    return time_per_image, label_file, output_file

def correct(label_file, output_file, url):
    """
    测试纠错结果
    :param label_file: 纠错语句
    :param output_file: 纠错后的结果
    :param url: 纠错接口地址
    :return: 每句话的纠错时间
    """
    sent_list = load_file_2d(label_file)
    image_list = [p[0] for p in sent_list]
    sent_generator = (p[1] for p in sent_list if len(p) > 1)
    # 因为有识别出来是空的，所以这里和原文件行数就不一样了，但是没关系，在比较的时候会自动匹配上，跳过空的这行

    results = []
    start = time.time()
    batch = 64
    count = 0

    while True:
        sents = []
        for i in range(batch):
            try:
                sents.append(next(sent_generator))
            except StopIteration:
                break
        if not sents:
            logging.info('已完成所有文本的纠错')
            break

        logging.info('从第%d条文本开始纠错', count)
        sents_corrected = correct_call.correct(sents,url)
        results += sents_corrected
        count += batch


    end = time.time()
    sent_num, char_num, num_per_sent = num_count(sent_list)

    time_per_sent = round((end - start) / sent_num, 4)
    logging.info('纠错【%d】条文本用时【%f】秒', sent_num, end - start)
    logging.info('平均每条文本用时【%f】秒', time_per_sent)
    logging.info('平均每条文本长度【%f】', num_per_sent)

    results = [[p, w] for p, w in zip(image_list, results)]
    save_file_2d(results, output_file)

    return time_per_sent



def num_count(sentences):
    """
    计算没被过滤掉的句子的数量及总字数
    :param sentences:
    :return:
    """
    sent_num = 0
    char_num = 0
    for s in sentences:
        s = s[1]
        if not pass_correct_filter(s):
            continue
        sent_num += 1
        char_num += len(s)

    if sent_num == 0:
        logging.info('共纠错了0个句子')
        return None, None, None
    char_per_sent = round(char_num/sent_num, 4)

    return sent_num, char_num, char_per_sent


def pass_correct_filter(original):
    """
    :param original: 待纠错的一句话
    :return: True for need correction, False for not
    """
    if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', original):
        return False
    chinese_counter = 0
    for c in original:
        if '\u4E00' < c < '\u9FA5':
            chinese_counter += 1
        if chinese_counter >= 3:
            return True

    return False



def compare(file_a, file_b, save_to, *args):

    fc = FileComparator(file_a, file_b)
    st = time.time()
    fc.compare()
    fc.save_report(save_to, *args)
    en = time.time()
    same_ratio = fc.get_same_ratio()
    logging.info('两文件相同行的比例（准确率）：【%f】', same_ratio)
    logging.info('比较总用时【%f】秒', en-st)
    logging.info('比较文件已保存：%s', save_to)

if __name__ == '__main__':

    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    parser.add_argument("--url")
    parser.add_argument("--label")
    parser.add_argument("--report")
    args = parser.parse_args()

    if not args.label or not os.path.exists(args.label):
        print("文件[%s]不存在" % args.label)
        exit(-1)

    url = args.url
    file1 = args.label

    # 后缀 1 代表label，2 代表识别结果，3代表纠错结果
    b, t = os.path.splitext(file1)
    file2 = b + '_2' + t   # file2:data/test_2.txt
    file3 = b + '_3' + t   # file3:data/test_3.txt

    # 后缀 1 代表label和识别结果的比较，2 代表纠错和识别结果的比较，3 代表label和纠错的比较
    b,t = os.path.splitext(args.report)
    report1 = b + '_1' + t
    report2 = b + '_2' + t
    report3 = b + '_3' + t

    # time_info, file1, file2 = recognize(file1, file2, url+'/crnn',limit=1000)
    # compare(file1, file2, report1, f'每张图片用时{time_info}秒')

    compare('data/test.txt', 'data/crnn.txt', 'data/report_test.html')
