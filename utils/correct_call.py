"""
调用纠错服务
"""

import requests

def correct(sentences, url):
    """
    多张图片的crnn
    :param images_base64:
    :param url:crnn的地址，可以动态传入，以测试不同版本的crnn
    :return:
    """
    post_data = {'sentences':sentences}

    headers = {'Content-Type': 'application/json'}
    response = requests.post(url,json=post_data, headers=headers)
    result = response.json()
    sents = result['sentences']

    return sents


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--url')
    args = parser.parse_args()

    url = args.url
    sentences = ['我爱北京夭安门','深坝市福田区华富路1018号中航中心26楼']
    print(correct(sentences, url))