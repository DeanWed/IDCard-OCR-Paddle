import re
import json
import string
from paddleocr import PaddleOCR

# 指定检测模型，识别模型和方向分类模型的路径
det_model_dir = './model/ch_det_infer'
rec_model_dir = './model/ch_rec_infer'
cls_model_dir = './model/ch_cls_infer'
rec_char_dict_path = './model/ppocr_keys_v1.txt'

# 初始化ocr模型和后处理模型
ocr = PaddleOCR(det_model_dir=det_model_dir,
                rec_model_dir=rec_model_dir,
                cls_model_dir=cls_model_dir,
                rec_char_dict_path=rec_char_dict_path,
                use_angle_cls=True, lang="ch", show_log=True, use_gpu=False)

# 识别身份证
def findIdcardResult(img_path):

    global ocr


    # 获取模型检测结果
    result = ocr.ocr(img_path, cls=True)
    if not result or not result[0]:
        return None
    #print(result)
    # 将检测到的文字放到一个列表中
    # txtArr = [line[1][0] for line in result[0]]

    try:
        txtArr = []
        for line in result[0]:
           txt = line[1][0]
           # 发现朝鲜文、彝文的身份证
           if (("姓" in txt and "性" in txt and "住" in txt) or ("名" in txt and "别" in txt and "生" in txt)) and line[1][1] < 0.75:
                continue
           else:
                txtArr.append(txt)

        #print(txtArr)
        postprocessing = IdCardStraight(txtArr)
        # # 将结果送入到后处理模型中
        id_result = postprocessing.run()
        #print(id_result)
        return id_result
    except Exception as e:
        pass
    return None
def verifyByIDCard(idcard):
    """
    验证身份证号码是否有效
    """
    sz = len(idcard)
    if sz != 18:
        return False

    weight = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    validate = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']
    sum = 0
    for i in range(len(weight)):
        sum += weight[i] * int(idcard[i])
    m = sum % 11
    return validate[m] == idcard[sz - 1]


class IdCardStraight:
    """
    身份证OCR返回结果，正常的身份证大概率识别没有什么问题，少数名族身份证，壮文、藏文、蒙文基本识别也没问题
    """
    nation_list = ["汉", "蒙古", "回", "藏", "维吾尔", "苗", "彝", "壮", "布依", "朝鲜", "满", "侗", "瑶", "白",
                   "土家", "哈尼", "哈萨克", "傣", "黎", "傈僳", "佤", "畲", "高山", "拉祜", "水", "东乡", "纳西",
                   "景颇", "柯尔克孜", "土", "达斡尔", "仫佬", "羌", "布朗", "撒拉", "毛难", "仡佬", "锡伯", "阿昌",
                   "普米", "塔吉克", "怒", "乌孜别克", "俄罗斯", "鄂温克", "崩龙", "保安", "裕固", "京", "塔塔尔",
                   "独龙", "鄂伦春", "赫哲", "门巴", "珞巴", "基诺"]

    def __init__(self, result):
        self.result = [
            i.replace(" ", "").translate(str.maketrans("", "", string.punctuation))
            for i in result
        ]
        #print(self.result)
        self.out = {"result": {}}
        self.res = self.out["result"]
        self.res["name"] = ""
        self.res["idNumber"] = ""
        self.res["address"] = ""
        self.res["gender"] = ""
        self.res["nationality"] = ""

    def birth_no(self):
        """
        身份证号码
        """
        for i in range(len(self.result)):
            txt = self.result[i]

            # 身份证号码
            if "X" in txt or "x" in txt:
                res = re.findall("\\d*[X|x]", txt)
            else:
                res = re.findall("\\d{18}", txt)

            if len(res) > 0:
                # 验证身份证号码是否有效  因为像藏文会出现刷出的身份证有超过18位数字的情况
                if verifyByIDCard(res[0]):
                    self.res["idNumber"] = res[0]
                    self.res["gender"] = "男" if int(res[0][16]) % 2 else "女"
                    break

    def full_name(self):
        """
        身份证姓名
        """
        # 如果姓名后面有跟文字，则取名后面的字段，如果"名"不存在，那肯定也就没有"姓名",所以在没有"名"的情况下只要判断是否有"姓"就可以了
        # 名字限制是2位以上，所以至少这个集合得3位数，才进行"名"或"姓"的判断
        for i in range(len(self.result)):
            txt = self.result[i]
            if ("姓名" in txt or "名" in txt or "姓" in txt) and len(txt) > 3:
                resM = re.findall("名[\u4e00-\u9fa5]+", txt)
                resX = re.findall("姓[\u4e00-\u9fa5]+", txt)
                if len(resM) > 0:
                    name = resM[0].split("名")[-1]
                elif len(resX) > 0:
                    name = resX[0].split("姓")[-1]
                if len(name) > 1:
                    self.res["name"] = name
                    self.result[i] = "temp"  # 避免身份证姓名对地址造成干扰
                    return

        # 如果姓名或名后面没有跟文字，但是有名 或姓名这个字段出现过的，去后面的集合为名字
        # 如果取的一个几个只有一个字，则接着取后面的集合，一般最多取2个集合就够了
        # 由于像新疆文、彝文这种类型的身份证，识别处理的集合值可能是英文，要进行去除
        indexName = -1
        for i in range(len(self.result)):
            txt = self.result[i]
            if "姓名" in txt or "名" in txt:
                indexName = i
                break
        if indexName == -1:
            for i in range(len(self.result)):
                txt = self.result[i]
                if "姓" in txt:
                    indexName = i
                    break
        if indexName == -1:
            return
        resName = self.result[indexName + 1]
        if len(resName) < 2:
            resName = resName + self.result[indexName + 2]
            self.res["name"] = resName
            self.result[indexName + 2] = "temp"  # 避免身份证姓名对地址造成干扰
        else:
            self.res["name"] = resName
            self.result[indexName + 1] = "temp"  # 避免身份证姓名对地址造成干扰

    def sex(self):
        """
        性别女民族汉
        """
        for i in range(len(self.result)):
            txt = self.result[i]
            if "男" in txt:
                self.res["gender"] = "男"

            elif "女" in txt:
                self.res["gender"] = "女"

    def national(self):
        # 性别女民族汉
        # 先判断是否有"民族xx"或"族xx"或"民xx"这种类型的数据，有的话获取xx的数据，然后在56个名族的字典里判断是否包含某个民族，包含则取对应的民族
        for i in range(len(self.result)):
            txt = self.result[i]
            if ("民族" in txt or "族" in txt or "民" in txt) and len(txt) > 2:
                resZ = re.findall("族[\u4e00-\u9fa5]+", txt)
                resM = re.findall("民[\u4e00-\u9fa5]+", txt)
                if len(resZ) > 0:
                    nationOcr = resZ[0].split("族")[-1]
                elif len(resM) > 0:
                    nationOcr = resM[0].split("民")[-1]

                for nation in self.nation_list:
                    if nation in nationOcr:
                        self.res["nationality"] = nation
                        self.result[i] = "nation"  # 避免民族对特殊情况下名字造成干扰
                        return
        # 如果 "民族" 或 "族" 和对应的民族是分开的，则记录对应对应的位置，取后一位的字符，同样去字典里判断
        indexNational = -1
        for i in range(len(self.result)):
            txt = self.result[i]
            if "族" in txt:
                indexNational = i
                break
        # 如果没有"民族"或 "族" ，则去判断是否含有"民",有则记录对应的位置，取后一位的字符，同样去字典里判断
        if indexNational == -1:
            for i in range(len(self.result)):
                txt = self.result[i]
                if "民" in txt:
                    indexNational = i
                    break
        if indexNational == -1:
            return
        national = self.result[indexNational + 1]
        for nation in self.nation_list:
            if nation in national:
                self.res["nationality"] = nation
                self.result[indexNational + 1] = "nation"  # 避免民族对特殊情况下名字造成干扰
                break

    def address(self):
        """
        地址
        """
        addString = []
        for i in range(len(self.result)):
            txt = self.result[i]
            # 这步的操作是去除下”公民身份号码“里的号对地址的干扰
            txt = txt.replace("号码", "")
            if "公民" in txt:
                txt = "temp"
            # 身份证地址    盟,旗,苏木,嘎查  蒙语行政区划  ‘大学’有些大学集体户的地址会写某某大学

            if (
                    "住址" in txt
                    or "址" in txt
                    or "省" in txt
                    or "市" in txt
                    or "县" in txt
                    or "街" in txt
                    or "乡" in txt
                    or "村" in txt
                    or "镇" in txt
                    or "区" in txt
                    or "城" in txt
                    or "室" in txt
                    or "组" in txt
                    or "号" in txt
                    or "栋" in txt
                    or "巷" in txt
                    or "盟" in txt
                    or "旗" in txt
                    or "苏木" in txt
                    or "嘎查" in txt
                    or "大学" in txt
            ):
                # 默认地址至少是在集合的第2位以后才会出现，避免经过上面的名字识别判断未能识别出名字，
                # 且名字含有以上的这些关键字照成被误以为是地址，默认地址的第一行的文字长度要大于7，只有取到了第一行的地址，才会继续往下取地址
                if i < 2 or len(addString) < 1 and len(txt) < 7:
                    continue
                    # 如果字段中含有"住址"、"省"、"址"则认为是地址的第一行，同时通过"址"
                # 这个字分割字符串
                if "住址" in txt or "省" in txt or "址" in txt:
                    # 通过"址"这个字分割字符串，取集合中的倒数第一个元素
                    addString.insert(0, txt.split("址")[-1])
                else:
                    addString.append(txt)
                self.result[i] = "temp"

        if len(addString) > 0:
            self.res["address"] = "".join(addString)
        else:
            self.res["address"] = ""

    def predict_name(self):
        """
        如果PaddleOCR返回的不是姓名xx连着的，则需要去猜测这个姓名，此处需要改进
        """
        for i in range(len(self.result)):
            txt = self.result[i]
            if self.res["name"] == "":
                if 1 < len(txt) < 5:
                    if (
                            "性别" not in txt
                            and "姓名" not in txt
                            and "民族" not in txt
                            and "住址" not in txt
                            and "出生" not in txt
                            and "号码" not in txt
                            and "身份" not in txt
                            and "nation" not in txt
                    ):
                        result = re.findall("[\u4e00-\u9fa5]{2,4}", txt)
                        if len(result) > 0:
                            self.res["name"] = result[0]
                            break
        for i in range(len(self.result)):
            txt = self.result[i]

    def run(self):
        self.full_name()
        self.sex()
        self.national()
        self.birth_no()
        self.address()
        self.predict_name()
        return json.dumps(self.out, ensure_ascii=False)