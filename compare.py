"""
文件比较工具
接收2个文件，逐行比较是否相同
返回一系列统计指标
可用于打标时的交叉验证，或者模型准确率测试（和GT比较）
"""
import difflib
import glob
import time

import os
from collections import Counter
import re
from utils.data_utils import load_file_2d

chars = "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ" \
        "！＠＃＄％＾＆＊（）－＿＋＝｛｝［］｜＼＜＞，．。；：､？／×·■﹑"
standard_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_+={}[]|\<>,.。;:、?/x..、"
changeable_pairs = [(c,sc) for c, sc in zip(chars, standard_chars)]
changeable_pairs += [(sc,c) for c, sc in zip(chars, standard_chars)]
ignorable_chars = ' '
force_change_chars = {',':'，', ':':'：', ';':'；', '(':'（', ')':'）'}
epsilon = 0.00000001


class Char(object):

    def __init__(self, char):
        self.char = char
        self.appear_time = 0
        self.miss_to = []
        # 统计量：
        self.miss_time = None
        self.miss_describe = None
        self.miss_percent = None

    def __repr__(self):
        return 'Char({})'.format(self.char)


class CharsCounter(object):

    def __init__(self):
        self.chars = {}
        self.summary = None

    def __getitem__(self, item):
        return self.chars[item]

    def __iter__(self):
        pass

    def __next__(self):
        pass

    def update_appear_time(self, sentence):
        if not sentence:
            return None
        for c in sentence:
            self.chars.setdefault(c, Char(c)).appear_time += 1

    def update_miss(self, chgs, subs, adds):
        if chgs:
            for truth, miss in chgs:   # ('日','目')
                self.chars.setdefault(truth, Char(truth)).miss_to.append(miss)
        if adds:
            for truth in adds:
                self.chars.setdefault(truth, Char(truth)).miss_to.append('add')
        if subs:
            for truth in subs:
                self.chars.setdefault(truth, Char(truth)).miss_to.append('sub')

    def summarize(self):
        for string_char, char in self.chars.items():
            char.miss_time = len(char.miss_to)
            if len(string_char) > 1:  # 如果是多字符的错，暂时不统计错误比例
                char.miss_percent = -1
            else:
                char.miss_percent = char.miss_time / (char.appear_time + epsilon )
            describe = Counter(char.miss_to) # Counter Object
            char.miss_describe = sorted(describe.items(), key=lambda x: x[1], reverse=True)

        by_miss_time = sorted(self.chars.items(), key=lambda x: x[1].miss_time, reverse=True)
        by_miss_time = [x for x in by_miss_time if x[1].miss_time != 0]
        by_miss_percent = sorted(self.chars.items(), key=lambda x: x[1].miss_percent, reverse=True)
        by_miss_percent = [x for x in by_miss_percent if (0 < x[1].miss_percent <= 1)]
        total_appear_time = sum([c.appear_time for _,c in self.chars.items()])
        total_miss_time = sum([c.miss_time * len(s) for s,c in self.chars.items()])
        total_miss_percent = total_miss_time / ( total_appear_time + epsilon )

        self.summary = {'by_miss_time': by_miss_time,
                        'by_miss_percent': by_miss_percent,
                        'total_appear_time': total_appear_time,
                        'total_miss_time': total_miss_time,
                        'total_miss_percent': total_miss_percent}


def is_char_junk(ch):
    return difflib.IS_CHARACTER_JUNK(ch, ws=' \t\n')


class FileComparator(object):

    def __init__(self, file, compare_to_file):
        """
        compare two files, with the second file a reference, could be GT or
        ali recognition results.
        :param file:
        :param compare_to_file:
        """
        self.to_file = load_file_2d(compare_to_file, column_sep=None)
        self.file = load_file_2d(file, column_sep=None)
        self.diff = None
        self.diff_lines = None
        self.differences = None
        self.count_dict = None
        self.chars_counter = CharsCounter()

    def preprocess(self):
        new_file = []
        for line in self.file:
            new_line = ''
            for i,s in enumerate(line):
                if s not in force_change_chars.keys():
                    new_line += s
                    continue
                else:
                    # 原本想要根据语境使用半角或全角字符，但是发现不需要
                    # context = line[i-1:i] + line[i+1:i+2]   # 上下文的字符串
                    # print('context:',context)
                    # if context.isnumeric():   # 如果是数字则不替换
                    #     new_line += s
                    #     continue
                    new_line += force_change_chars[s]
            new_file.append(new_line)
        self.file = new_file


    def compare(self):
        """
        example: 深圳市福田区华富路1018号  vs   深川市福田区华富路1018号
        :return:
        """

        # 全部转换为全角字符
        self.file = [l.replace(' ', '').replace(',', '，').replace(';', '；').replace(':', '：').replace('(', '（').replace(')', '）') for l in self.file]
        self.to_file = [l.replace(' ', '').replace(',', '，').replace(';', '；').replace(':', '：')
                            .replace('(', '（').replace(')', '）') for l in self.to_file if l]

        self.diff = difflib._mdiff(self.file, self.to_file, charjunk=is_char_junk)
        differences = {'changes':[],
                      'adds':[],
                      'subs':[]}
        same_lines = []
        diff_lines = []

        for (_,line_a), (_,line_b), is_different in self.diff:   # 拆包直接拆出需要的文字段落

            if not is_different:
                same_lines.append(line_b)
                self.chars_counter.update_appear_time(line_b)

            else:
                chgs, adds, subs, line_a, line_b = self.char_diffs(line_a, line_b)
                if (not chgs) and (not adds) and (not subs):
                    same_lines.append(line_b)
                    self.chars_counter.update_appear_time(line_b)
                    continue

                differences['changes'] += chgs
                differences['adds'] += adds
                differences['subs'] += subs
                diff_lines.append((line_a, line_b))
                self.chars_counter.update_appear_time(line_b)
                self.chars_counter.update_miss(chgs, adds, subs)


        differences = {k:Counter(v) for k,v in differences.items()}
        self.chars_counter.summarize()

        s_no = len(same_lines)
        t_no = s_no + len(diff_lines)
        count_dict = {'number of same lines:': s_no,
                      'number of total lines:': t_no,
                      'ratio of same:': round(s_no/t_no, 4)}
        self.diff_lines = diff_lines
        self.count_dict = count_dict
        self.differences = differences


    def char_diffs(self,line_a, line_b):
        c, line_a = self.find_patterns(line_a,'change')
        chgs = []
        if c:
            cb, line_b = self.find_patterns(line_b)
            for _c, _cb in zip(c, cb):
                if (_c, _cb) in changeable_pairs:
                    line_b = line_b.replace(_cb, _c)   # 将可以忽略的在line_b替换
                    continue
                chgs.append((_cb, _c))

        adds, line_b = self.find_patterns(line_b,'add')   # 增加只会在line_b标出
        subs, line_a = self.find_patterns(line_a,'sub')   # 减少只会在line_a标出

        line_a = line_a.replace('\1', '')
        line_b = line_b.replace('\1', '')

        return chgs, adds, subs, line_a, line_b

    @staticmethod
    def find_patterns(line, mode='change'):
        """
        在字符串中找到所有被标注的文字，返回被标注的文字列表，以及去掉标注后的原文。标注类型：change,add,sub
        :param line:
        :param mode: 'change','add' or 'sub'
        :return: List of words that have been changed/added/subtracted

        Examples:
        test_line = '\0^修改1\1随便什么\0^修改2\1随便什么\0+增加内容\1结尾'
        find_patterns(test_line, 'change')   --->
        ['修改1','修改2'], 修改1随便什么修改2随便什么\0+增加内容\1结尾
        find_patterns(test_line, 'add')    --->
        ['增加内容']
        find_patterns(test_line, 'sub')   --->
        []
        """

        start, end = 0, 0
        s = []
        if mode == 'change':
            pattern = '\0^'
        elif mode == 'add':
            pattern = '\0+'
        else:
            pattern = '\0-'
        while True:
            start = line.find(pattern,end)
            if start == -1:
                break
            end = line.find('\1', start)
            s.append(line[start + 2:end])

        if end:    # 有修改时，end != 0
            line = line.replace(pattern, '')
        return s, line


    def save_report(self, save_to, mode='standard', *args):
        """
        save comparison report as html file.
        :param save_to: report path.
        :param mode:
        if mode is easy, save a report that could be read easier with images showing correspondingly;
        if mode is whole, save a report generated by python diff of whole page.
        :param args: anything that wanted to be shown in the report.
        :return: None.
        """

        _diff = difflib.HtmlDiff(wrapcolumn=100)
        if mode == 'full':  # show both same lines and different lines in original order
            _h = _diff.make_file(self.file, self.to_file)
        else:  # show only different lines
            _h = _diff.make_file([d[0] for d in self.diff_lines],
                       [d[1] for d in self.diff_lines])
        if mode == 'easy':  # show images for better understanding
            _h = self.easier_report(_h)

        head = [k + '\t' +str(v) +'\n' for k, v in self.count_dict.items()]
        additional_info = '\t'.join(args)
        _h = ''.join(head) + additional_info + _h
        with open(save_to, 'w') as f:
            f.write(_h)

    def get_same_ratio(self):
        if self.count_dict is None:
            raise Warning('未进行文件比较')
        return self.count_dict['ratio of same:']

    @ staticmethod
    def easier_report(report):
        root = 'file:///Users/tt/CV/data_clean/'
        jpgs = set(re.findall(r'\<td nowrap=\"nowrap\"\>(.*?)\.jpg', report))
        pngs = set(re.findall(r'\<td nowrap=\"nowrap\"\>(.*?)\.png', report))

        def add_img(imgs, img_type, report):
            if imgs is None:
                return report
            for i, img in enumerate(jpgs):
                # 展示图片
                before = '<td nowrap="nowrap">' + img + '.' + img_type
                after = '<td><img src="' + root + img + '.' + img_type + '"></td><td nowrap="nowrap">'
                report = re.sub(before, after, report, 1)
                report = re.sub(before, '<td nowrap="nowrap">', report)

            return report

        report = add_img(jpgs, 'jpg', report)
        report = add_img(pngs, 'png', report)

        return report

    def get_diff_dict(self):
        _dd = self.differences
        _cd, _ad, _sd = _dd.values()
        cd = sorted(_cd.items(), key=lambda x: x[1], reverse=True)
        ad = sorted(_ad.items(), key=lambda x: x[1], reverse=True)
        sd = sorted(_sd.items(), key=lambda x: x[1], reverse=True)

        return cd, ad, sd

    def save_changes(self, file):
        changes, adds, subs = self.get_diff_dict()
        lines_c = ['\t'.join(['\t'.join(c), str(n)]) for c,n in changes]
        lines_a = ['\t'.join([c, str(n)]) for c,n in adds]
        lines_s = ['\t'.join([c, str(n)]) for c,n in subs]
        total_c = sum([n for c,n in changes])
        total_a = sum([n for c,n in adds])
        total_s = sum([n for c,n in subs])
        with open(file, 'w') as f:
            f.write('错误总次数:{}'.format(total_c+total_a+total_s))
            f.write('\n===============================\n')
            f.write('替换错误总次数:{}\n'.format(total_c))
            f.write('-------------------------------\n')
            f.write('真实\t识别\t出现次数\n')
            f.write('\n'.join(lines_c))
            f.write('\n===============================\n')
            f.write('减少错误总次数:{}\n'.format(total_a))
            f.write('-------------------------------\n')
            f.write('未识别出的\t出现次数\n')
            f.write('\n'.join(lines_a))
            f.write('\n===============================\n')
            f.write('增加错误总次数:{}\n'.format(total_s))
            f.write('-------------------------------\n')
            f.write('多识别的\t出现次数\n')
            f.write('\n'.join(lines_s))

    def save_chars_report(self, file):
        with open(file, 'w') as f:
            sm = self.chars_counter.summary

            f.write('总字符数【{}】\t错误字符数【{}】\n错误率【{:.2%}】\n'.format(sm['total_appear_time'],
                                                         sm['total_miss_time'],
                                                         sm['total_miss_percent']))

            misses = sm['by_miss_time']
            f.write('按照错误【次数】排序:\n==============================\n')
            f.write('字符\t错误次数\t出现次数\t识别为\n')
            for char_string, char in misses:
                f.write('{}\t{}\t{}\t'.format(char_string, char.miss_time, char.appear_time))
                for miss, times in char.miss_describe:
                    f.write('【{} {}】\t'.format(miss, times))
                f.write('\n')

            misses = sm['by_miss_percent']
            f.write('按照错误【比例】排序:\n==============================\n')
            f.write('字符\t错误比例\t出现次数\t识别为\n')
            for char_string, char in misses:
                f.write('{}\t{:.2%}\t{}\t'.format(char_string, char.miss_percent, char.appear_time))
                for miss, times in char.miss_describe:
                    f.write('【{} {}】\t'.format(miss, times))
                f.write('\n')





def convert_format(file1, file2):
    """将预测格式转换为比较格式
    True data/test/201905290221140526wx6AMWX1559110845570-72.jpg 【25.00】 【25.00】
    转换成------>
    data/test/201905290221140526wx6AMWX1559110845570-72.jpg 25.00
    """
    f = open(file1)
    lines = f.readlines()
    new_lines = []
    for l in lines:
        l = l.strip().split(' ')
        l = l[1] + ' ' + l[-1][1:-1]
        new_lines.append(l)

    with open(file2, 'w') as file:
        file.write('\n'.join(new_lines))

def read_ali_results(root, save_to, *postfix):
    """
    读取阿里识别结果（格式：1150,176,874,172,874,144,1150,148,"识别结果"）
    开头相同的文件为一组（组名为开头），分行后，将识别结果（不带坐标）保存到同以文件
    :param root:
    :param save_to:
    :param postfix:
    :return: 保存结果到 save_to 下
    """
    # 识别结果是按照组（一份pdf文档）命名
    groups = ['0','1','2','3','4','5']
    for group in groups:
        print('处理第{}份文档'.format(group))
        save_name = '_'.join([group, *postfix]) +  '.txt'
        save_path = os.path.join(save_to, group, save_name)
        if os.path.exists(save_path):
            os.remove(save_path)
        index = 0

        while True:
            file = os.path.join(root, group + '_' + str(index) + '.txt')
            if not os.path.exists(file):
                break
            print('读取第{}张图片的识别结果'.format(index))
            index += 1

            with open(file, 'r') as f:
                lines = f.readlines()
            content = []
            for l in lines:
                l = l.strip().split(',', 8)   # [2,2,1,2,1,1,2,2,'识别结果']  最多分割8次（分割8个逗号）
                # 某些识别结果是带双引号的，去掉  "2.1在办理债权资产服务有关事务过程中"
                if l[-1][0] == '"':
                    l[-1] = l[-1][1:-1]
                pos = [int(_l) for _l in l[:-1]]   # 坐标转换为数字
                content.append((tuple(pos),l[-1]))   #  ((2,2,1,2,1,1,2,2,), '识别结果')
            texts = get_rows(content)
            with open(save_path, 'a') as f:
                f.write('\n'.join(texts) + '\n')


def get_rows(content, line_threshold = 10):
    """分行
    :param content: [((2,2,1,2,1,1,2,2,), '识别结果1'), ((2,2,1,2,1,1,2,2,), '识别结果2'), ……]
    :param line_threshold: 纵坐标差距为多大时，认为是两行
    return:
    texts_in_line:['识别结果第1行','识别结果第2行', ……]
    """
    texts_in_line = []
    temp_y = -10
    temp_line = ''

    for line in content:
        if line[0][1] - temp_y > line_threshold:
            if temp_line:
                texts_in_line.append(temp_line)
            temp_y = line[0][1]
            temp_line = line[1]
        else:
            temp_line += line[1]

    texts_in_line.append(temp_line)

    return texts_in_line




if __name__ == '__main__':
    # root = 'data/multi_ali/ali/contract_100/labels/'
    # path_to = '/Users/tt/CV/ocr/docscan/data/txts/'
    # read_ali_results(root, path_to, 'ali','high')

    # # convert_format('data/test_0608.txt', 'data/test_06081.txt')

    root = '/Users/tt/CV/ocr/docscan/data/txts/3'
    # root = 'data/'
    names = ['3','0705']
    fullname = '_'.join(names)
    raw_name = names[0] + '_raw.txt'
    change_name = 'changes_' + fullname + '.txt'
    chars_name = 'chars_' + fullname + '.txt'
    report_name = fullname + '.html'
    st = time.time()

    fc = FileComparator(os.path.join(root,fullname + '.txt'), os.path.join(root,raw_name))
    fc.compare()
    fc.save_changes(os.path.join(root, change_name))
    fc.save_chars_report(os.path.join(root, chars_name))
    fc.save_report(os.path.join(root, report_name), mode = 'full')


    en = time.time()
    print(f'比较完成，总用时【{en-st}】秒')
