# -*- encoding: utf-8 -*-
"""
清洗语料
"""
import glob
import time
import re
import collections
import json
import pickle
from utils.data_utils import load_file_2d

from pdf_reader import read_by_path


# 合同
def generate_corpus(pdf_path, result_path):
    """
    批量读取收集到的合同pdf，作为语料
    :return:
    """
    # 获取文件列表
    # pdf_list = glob.glob(pdf_path+'*.pdf')
    # pdf_list += glob.glob(pdf_path+'*/*.pdf')

    pdf_list = [pdf_path]

    print(f'共有文件{len(pdf_list)}个')

    lines = []
    error_pdf = []

    # 同步读取，存入line列表
    for i,pdf in enumerate(pdf_list):
        print(f'开始读取pdf{i}:{pdf}')
        start = time.time()
        try:
            l = read_by_path(pdf)
            lines += l
            end = time.time()
            print(f'读取完毕，耗时{end-start}秒')
        except Warning as w:
            print(f'读取出错: 文件{i}{pdf}')
            error_pdf.append(pdf)
            print(w)


    # 将line列表保存为txt文本
    with open(result_path,'a+') as f:
        f.write(''.join(lines))

# def segment(l,line_length = 32):  # 把一段话切成固定长度的小段
#     # 递归写法，栈溢出
#     if len(l) <= line_length:
#         return [l+'\n']
#     return [l[:line_length+1]+'\n'] + segment(l[line_length+1:])

def segment(l,line_length = 32):  # 把一段话切成小段
    n = len(l) // line_length
    line_segments = []
    for i in range(n):
        line_segments.append(l[32*i:32*(i + 1)] + '\n')
    line_segments.append(l[32*n:] + '\n')

    return line_segments

# 清洗从word文档粘到txt中的语料
def clean_corpus(corpus_path, result_path):
    # 读入文件
    line_list = []


    with open(corpus_path,'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue  # 删掉空行
            else:
                line_list += segment(line)
    print('语料清洗完毕')
    with open(result_path,'a+') as f:
        for _l in line_list:
            f.write(_l)


def save_file(result_path, line_list):
    with open(result_path,'a+') as f:
        for _l in line_list:
            f.write(_l)


def clean_charset(corpus_path,charset_path,char_list):
    """把corpus中不在charset里面的文字处理一下"""
    # 先跑部分corpus，看一下不在charset里面的都是什么字
    with open(corpus_path,'r') as f:
        sentences = f.readlines()
    with open(charset_path,'r',encoding='utf-8') as c:
        charset = c.readlines()
        charset = [ch.strip("\n") for ch in charset]
        charset = "".join(charset)
        charset = list(charset)
        print(len(charset))
        print("=============================")
        if charset[-1] != " ":
            charset.append(" ")

    for c in char_list:
        if c in charset:
            print(f'在字表中找到了{c}')
        else:
            print(f'！！没有找到{c}')

    chars_not_in_charset = collections.defaultdict(lambda : 0)
    for i,s in enumerate(sentences):
        s = s.strip()
        s = re.sub(r'\t','',s)
        if i % 1000 == 0: print(f'===========完成了{i}个')
        for char in s:
            if char not in charset:
                chars_not_in_charset[char] += 1

    print(len(chars_not_in_charset))
    chars_not_in_charset = sorted(chars_not_in_charset.items(), key=lambda x:x[1], reverse=True)

    save_str = [t[0]+'\n' for t in chars_not_in_charset]
    with open('abnormal_char.txt','w') as f:
        f.write(''.join(save_str))


def update_corpus(corpus_path,corpus_path_new):
    abnormal_chars = {}
    with open('abnormal_char.txt','r') as f:
        for line in f:
            l = line.strip().split('\t')
            if len(l) == 2:
                abnormal_chars[l[0]] = l[1]

    def map_char(chars):
        # print(chars)
        c_list = list(chars)
        for i,c in enumerate(c_list):
            # print(c)
            if c in abnormal_chars.keys():
                c_list[i] = abnormal_chars[c]
                print(f'替换异常汉字{c}')
        return ''.join(c_list)

    with open(corpus_path,'r') as f:
        file_string = f.read()
        file_string = map_char(file_string)

    with open(corpus_path_new,'w') as f:
        f.write(file_string)

def cut_tab(corpus_path,corpus_path_new):
    """把\t前后拆成两行。主要因为实际上在切割行的时候也会拆成两段，同时解决\t字符问题"""
    sentences = []
    with open(corpus_path,'r') as f:
        for l in f:
            sens = l.strip().split('\t')
            # sens = [s+'\n' if s[-1] != '\n' else s for s in sens]  # 给\t前面的字符串加上\n
            sentences += sens

    with open(corpus_path_new,'w') as f:
        f.write('\n'.join(sentences))

def sub_dots_in_index(corpus_path,corpus_path_new):
    """把目录中排版用的一大堆点去掉，因为实际切图时不会切出"""
    sentences = []
    with open(corpus_path,'r') as f:
        for l in f:
            print(l)
            l = re.sub('\.\.','',l)
            print('----->',l)
            sentences.append(l)

    with open(corpus_path_new,'w') as f:
        f.write(''.join(sentences))

def read_charset(charset_file):
    with open(charset_file, 'r', encoding='utf-8') as f:
        charset = f.readlines()
    charset = [ch.strip("\n") for ch in charset]   # 3行的列表
    charset = "".join(charset)   # 大字符串
    charset = list(charset)

    return charset


def char_num_count(source_file, row_filter = '', charset = None, char_count = None, excluded_char_count = None):
    """
    统计给定文本文件中字符的数量。可指定字符集，分别返回字符集内的、与字符集外的字符列表。
    :param source_file: 源文件路径
    :param charset: 字符集文件路径
    :param char_count:
    :param excluded_char_count: 字符集外字符列表保存到的路径
    :return:
    """

    if charset is not None:
        chars = {c:0 for c in read_charset(charset)}
        print(f'字符集共{len(chars)}个字符')
    else:
        chars = collections.defaultdict(lambda x:0)
    excluded_chars = {}

    line_count = 0
    with open(source_file, 'r') as f:
        for i, l in enumerate(f):
            if i % 100000 == 0: print(f'进行到第{i}行')

            filename, _, label = l[:-1].partition(' ')
            if filename.find(row_filter) == -1: continue  # 如果不包含关键词，不做统计
            label = label.replace(' ', '')
            for c in label:
                try:
                    chars[c] += 1
                except KeyError:
                    excluded_chars.setdefault(c,0)
                    excluded_chars[c] += 1
            line_count += 1

    print('全部处理完, 共有{}行'.format(line_count))

    excluded_chars_ordered = sorted(excluded_chars.items(), key=lambda x:x[1], reverse=True)
    chars_ordered = sorted(chars.items(), key=lambda x:x[1], reverse=True)

    if excluded_char_count is not None:
        print('保存集外字符统计:{}'.format(excluded_char_count))
        with open(excluded_char_count, 'w') as f:
            for c in excluded_chars_ordered:
                line = c[0] + '\t' + str(c[1]) + '\t' + str(c[0].encode()) + '\n'
                f.write(line)

    if char_count is not None:
        print('保存字符统计:{}'.format(char_count))
        with open(char_count, 'w') as f:
            for c in chars_ordered:
                line = c[0] + '\t' + str(c[1]) + '\t' + str(c[0].encode()) + '\n'
                f.write(line)

    return chars,excluded_chars

# 处理json格式的维基百科语料
def text2row(text_list):
    """input: text list
    output:rows in a list
    """
    line_list = []
    for t in text_list:
        try:
            t = json.loads(t)['text']
            t = t.split('\n\n')
            for l in t:
                line = re.sub(r'\n', '', l)
                if len(line) == 0:
                    continue  # 删掉空行
                else:
                    line_list += segment(line)
        except TypeError:
            continue

    return line_list

def filter_chinese(char):
    """
    :param char: tuple, (char, anything).
    :return: if char is a Chinese char, return True; else return False
    """
    return u'\u4E00' < char[0] < u'\u9FA5'

def generate_wiki_corpus():
    # rows = []
    # files = glob.glob('/Users/tt/Downloads/wiki_zh/*/wiki_*')
    #
    # for i, file in enumerate(files):
    #     if i % 10 == 0: print(f'处理到第{i}个文件')
    #     with open(file, 'r') as f:
    #         text =f.readlines()
    #         rows += text2row(text)
    #
    # with open('/Users/tt/Downloads/wiki_zh/nlp.txt', 'w') as f:
    #     f.write(''.join(rows))

    c,ec = char_num_count('/Users/tt/Downloads/wiki_zh/nlp.txt', '/Users/tt/CV/ocr/crnn/crnn/config/charset.6883.txt')

    def save_char_frequency(chars,path):
        chars = filter(filter_chinese, chars.items())
        chars = sorted(chars, key=lambda x: x[1], reverse=True)
        chars = [_c[0] + '\t' + str(_c[1]) + '\n' for _c in chars]
        with open(path, 'w') as f:
            f.write(''.join(chars))
        print('done')

    save_char_frequency(c,'/Users/tt/Downloads/wiki_zh/corpus_char_readable_6880.txt')
    save_char_frequency(ec,'/Users/tt/Downloads/wiki_zh/corpus_excluded_char_readable_6880.txt')

    # with open('/Users/tt/Downloads/wiki_zh/corpus_char.txt', 'wb') as f:
    #     pickle.dump(c,f)
    # with open('/Users/tt/Downloads/wiki_zh/corpus_excluded_char.txt', 'wb') as f:
    #     pickle.dump(ec,f)
    #
    # with open('/Users/tt/Downloads/wiki_zh/corpus_excluded_char.txt', 'rb') as f:
    #     c = pickle.load(f)

def merge_chars_frequency(file_list, save_to):
    """
    有来源为不同样本的多个字符数统计文件，进行整合
    :param file_list:
    :return:
    """
    chars_dict = {}
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split('\t') for l in lines]
            for l in lines:
                chars_dict.setdefault(l[0],0)
                chars_dict[l[0]] += int(l[1])

    lines = sorted(chars_dict.items(),key=lambda x: x[1], reverse=True)
    lines = [l[0] + '\t' + str(l[1]) for l in lines]
    lines = '\n'.join(lines)
    with open(save_to, 'w') as f:
        f.write(lines)

    return chars_dict

def test():
    with open('/users/tt/CV/ocr/crnn/data/data_generator/corpus_new.txt') as f:
        new_corpus = f.read()
    for c in ['..','\t','⾦','⽂','⼀']:
        if c in new_corpus:
            print('未通过测试')
        else:
            print('通过测试')

def temp():
    ll = load_file_2d('/Users/tt/Documents/chars_oov_total.txt','\t')
    result = ''.join([l[0] for l in ll])
    print(result)

    f = open('../ocr/crnn/crnn/config/charset.4100.txt', 'r').read()
    print(len(f))
    chars = []
    for i,s in enumerate(f):
        if s in chars:
            print('重复了:',i,s)
        chars.append(s)

    s1 = '衢亳濮漯圳莞儋泸泗颍泸佤岚泾潼祜赉桦洮睢濮沅陉栾涞涿绛溧沭瓯浔嵊婺岱弋谯璧旌柘汶莒荥嵩淇驿澧圩榕岑梓仡麟勐湟坻藁妃蠡骅猗稷芮岢隰磴岫鲅蛟珲讷箐闵邺邳盱眙邗鄞暨衢缙畲鸠庵濉枞歙黟琊埇砀芗诏濂鄱崂峄滕罘朐兖郯茌莘棣郓鄄杞瀍偃郏陟鄢郾漯渑淅浉潢硚陂猇秭浠蕲芙浏淞渌攸醴晖耒汨溆芷禺浈濠禅莞邕覃仫儋碚綦郫邛崃蔺邡犍沐阆珙筠蓥孚湄阡谟麒蒗濞迦灞鄠岐崆峒岷宕晏坂鄯耆伽'
    s2 = '婷晗鑫祺瑾琪倩媛楠馨缤罡闫昊珂睿瑛裱炜怡妍芸宸缪苡烨畈嘟炫鞫邸摽窦雯薇玮钊淼琦珞佥曦钰煜渎璐姣娅晟恪'
    for s in s2:
        if s in s1:
            print('重复了',s)

if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--filter")
    parser.add_argument("--charset")
    parser.add_argument("--count")

    args = parser.parse_args()

    source_file = args.file
    row_filter = args.filter
    charset = args.charset
    char_count = args.count
    prefix = os.path.splitext(char_count)
    excluded_char_count = prefix[0] + '_1' + '.txt'
    print('统计文件{}的字符数量，字符集：{}'.format(source_file, charset))

    char_num_count(source_file, row_filter, charset, char_count, excluded_char_count)