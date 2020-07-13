"""
识别大图，识别方式为调用API接口
其中的 own_ocr 有点重复，应该和ocr里面的整合一下
"""
import base64
import cv2
import os
import requests


def nparray2base64(img_data):
    """
        nparray格式的图片转为base64（cv2直接读出来的就是）
    :param img_data:
    :return:
    """
    _, d = cv2.imencode('.jpg', img_data)
    return str(base64.b64encode(d), 'utf-8')

def own_ocr(img_base64, base_url='http://localhost:8080'):
    url = base_url + "/document"
    post_data = {"img": img_base64, "sid": "iamsid", "do_correct": True}
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=post_data, headers=headers)
    data = response.json()
    return data


def parse_doc_result(data):
    rows = data['prism_rowsInfo']
    rows = [r['word'] for r in rows if r['word']]

    return rows



def recognize_contracts(root, save_to):
    """
    识别合同图片
    :param root: 合同图片的根目录
    :param save_to: 识别结果保存的根目录
    :return: None 保存识别结果到指定的目录，一个合同文件（group）保存一个文件
    """
    groups = ['3','4']
    for group in groups:
        print('处理第{}份'.format(group))
        save_name = group + '_corrected.txt'
        save_path = os.path.join(save_to, group, save_name)
        if os.path.exists(save_path):
            os.remove(save_path)
        index = 0

        while True:
            file = os.path.join(root, group + '_' + str(index) + '.jpg')
            if not os.path.exists(file):
                break
            print('识别第{}张图片'.format(index))
            index += 1

            img = cv2.imread(file)
            img = nparray2base64(img)
            result = own_ocr(img, URL)
            rows = parse_doc_result(result)
            with open(save_path, 'a') as f:
                f.write('\n'.join(rows) + '\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    args = parser.parse_args()

    URL = args.url

    file = '/Users/tt/CV/ocr/docscan/data/imgs/0_0.jpg'
    img = cv2.imread(file)
    img = nparray2base64(img)
    result = own_ocr(img, URL)

    root = '/Users/tt/CV/ocr/docscan/data/imgs/'
    path_to = '/Users/tt/CV/ocr/docscan/data/txts/'
    recognize_contracts(root, path_to)
