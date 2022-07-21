import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from selenium import webdriver
import time
import requests 
from recordclass import recordclass
# 详细的测试思路见 "测试思路.md"
# 这个文档非常重要

# 将四张图片上传到百度网盘，然后获取永久的公有共享链接，避免使用selenium文件框传图
# 能少用前端方式就少用
IMAGE_ZERO_URL = "https://thumbnail0.baidupcs.com/thumbnail/7a20402b4s019cf23c2e0faeb0f531c8?fid=1006717288-250528-1016375879979427&time=1658365200&rt=sh&sign=FDTAER-DCb740ccc5511e5e8fedcff06b081203-kutu4wSViaoHiP8XFtsILJZZeis%3D&expires=8h&chkv=0&chkbd=0&chkpc=&dp-logid=59646311535878824&dp-callid=0&file_type=0&size=c710_u400&quality=100&vuk=-&ft=video"

TestEle = recordclass("TestEle", ["src", "dst", "src_url", "dst_url", "model"])

# 说明：一共应该是4组test element，这里只做一组，作为demo，
# 因为没有那么多时间去训练四组模型。如果有需求的话，
# 可以把4组图片对应的数据集拿下来，自己训练模型。
# 提取图片的可用特征比较耗时，其他还好。

cases = [ 
    TestEle("./data/0.png", "./data/0_res.png", IMAGE_ZERO_URL, None, "./zero_model")
]

# 这里最好用现成的pytest框架
# print函数最好用logger替代。
# 这里只是demo
class TestsRunner(object):
    def __init__(self):
        self.driver = webdriver.Firefox()
        self.driver.set_window_size(800,800)
    
    def init_web_driver(self):
        self.driver.get(r"https://www.google.com/imghp?tbm=isch")
        time.sleep(2)

    def load_tf_model_and_do_predict(self, test_ele):
        dst_img_path = test_ele.dst
        img = Image.open(dst_img_path)
        img=img.resize((28,28),Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        img_arr = 255 - img_arr
        img_arr=img_arr/255.0
        x_predict = img_arr[tf.newaxis,...]
        model = load_model(test_ele.model)
        result = model.predict(x_predict)
        pred= tf.argmax(result, axis=1)
        return int(pred)

    def run_case(self, test_ele):
        image_span = self.driver.find_element_by_xpath("//span[@class='tdPRye']")
        image_span.click()
        # 此处为测试代码，实际工程中应使用waitForCondition函数代替。
        # 等待的条件包括，页面js加载完成
        # 或者特定元素visible
        time.sleep(1)

        image_input = self.driver.find_element_by_id("Ycyxxc")
        image_input.send_keys(test_ele.src_url)

        time.sleep(1)

        search_btn = self.driver.find_element_by_id("RZJ9Ub")
        search_btn.click()

        time.sleep(6)

        # 索引从0开始，所以第三张图片是2
        third_img = self.driver.find_elements_by_xpath("//div[@class='uhHOwf BYbUcd']")[2]
        third_img.click()

        time.sleep(2)

        src = self.driver.find_element_by_xpath("//img[@class='n3VNCb KAlRDb']").get_attribute("src")

        print(src)
        test_ele.dst_url = src

        resp = requests.get(src)
        src_img = resp.content

        with open(test_ele.dst, "wb") as f:
            f.write(src_img)
        
        pred = self.load_tf_model_and_do_predict(test_ele)
        err_msg = "Image is not related with the corresponding src, case: {}".format(test_ele.src)
        print("Current pred is: ", pred)
        assert pred == 0, err_msg
        
    def quit_driver(self):
        self.driver.quit()



if __name__ == '__main__':
    runner = TestsRunner()
    for case in cases:
        runner.init_web_driver()
        runner.run_case(case)
        runner.quit_driver()

