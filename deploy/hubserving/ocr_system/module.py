# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, ".")
import copy

import time

from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, runnable, serving
import cv2
import numpy as np
import paddlehub as hub

from tools.infer.utility import base64_to_cv2
from tools.infer.predict_system import TextSystem
from tools.infer.utility import parse_args
from deploy.hubserving.ocr_system.params import read_params
# from tools.infer.utility import fuzzyMatching, sort_text_score, gunReadXlsx,contrast_img, txt_crop


@moduleinfo(
    name="ocr_system",
    version="1.0.0",
    summary="ocr system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/text_recognition")
class OCRSystem(hub.Module):
    def _initialize(self, use_gpu=False, enable_mkldnn=False):
        """
        initialize with the necessary elements
        """
        cfg = self.merge_configs()

        cfg.use_gpu = use_gpu
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("use gpu: ", use_gpu)
                print("CUDA_VISIBLE_DEVICES: ", _places)
                cfg.gpu_mem = 3000
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        cfg.ir_optim = True
        cfg.enable_mkldnn = enable_mkldnn

        self.text_sys = TextSystem(cfg)
        self.drop_score = cfg.drop_score
        '''load gun dataSet'''
        # dir = 'Gun/gunType.xlsx'
        # self.types, self.types_crop = gunReadXlsx(dir)
        # self.types = []
        # self.types_crop = []

    def merge_configs(self, ):
        # deafult cfg
        backup_argv = copy.deepcopy(sys.argv)
        sys.argv = sys.argv[:1]
        cfg = parse_args()

        update_cfg_map = vars(read_params())

        for key in update_cfg_map:
            cfg.__setattr__(key, update_cfg_map[key])

        sys.argv = copy.deepcopy(backup_argv)
        return cfg

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            images.append(img)
        return images

    def predict(self, images=[], paths=[]):
        """
        Get the chinese texts in the predicted images.
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The inference_result of chinese texts and save path of images.
        """
        # if (len(self.types) == 0 or len(self.types_crop) == 0):
        #     self.types, self.types_crop = gunReadXlsx(dir)
        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError("The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        all_results = []
        txts_rel = []
        scores_rel = []
        for img in predicted_data:
            if img is None:
                logger.info("error in loading image")
                all_results.append([])
                continue
            starttime = time.time()
            '''图像处理部分：
                每中处理方式后都用模型进行了识别，并保存了识别结果和置信度；
                如果多种识别方式一起使用，将按照识别结果的置信度或者字符串长度排序，选择最优的处理方式（该方式将耗费更多时间）；
                如果只选用一种方式，注释其他方式即可。
            '''
            dt_boxes_collect = []
            rec_res_collect = []

            # 不做处理
            dt_boxes, rec_res = self.text_sys(img)
            dt_boxes_collect.append(dt_boxes)
            rec_res_collect.append(rec_res)

            # 增强对比度
            # img_enhance = contrast_img(img, 1.3, 0)
            # dt_boxes, rec_res = self.text_sys(img_enhance)
            # dt_boxes_collect.append(dt_boxes)
            # rec_res_collect.append(rec_res)

            # # 中值滤波
            # img_media = cv2.medianBlur(img, 5)
            # dt_boxes, rec_res = self.text_sys(img_media)
            # dt_boxes_collect.append(dt_boxes)
            # rec_res_collect.append(rec_res)

            # # 直方图均衡化
            # image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))  # 对图像进行分割，10*10
            # image_gray_limit_hist = clahe.apply(image_gray)
            # image_gray_limit_hist = cv2.cvtColor(image_gray_limit_hist, cv2.COLOR_GRAY2BGR)
            # dt_boxes, rec_res = self.text_sys(image_gray_limit_hist)
            # dt_boxes_collect.append(dt_boxes)
            # rec_res_collect.append(rec_res)

            # # 直方图均衡化 + 中值滤波
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))  # 对图像进行分割，20*20
            # image_gray_limit_hist = clahe.apply(img_gray)  # 进行直方图均衡化
            # image_media_gray_limit_hist = cv2.medianBlur(image_gray_limit_hist, 5)
            # image_media_gray_limit_hist = cv2.cvtColor(image_media_gray_limit_hist, cv2.COLOR_GRAY2BGR)
            # dt_boxes, rec_res = self.text_sys(image_media_gray_limit_hist)
            # dt_boxes_collect.append(dt_boxes)
            # rec_res_collect.append(rec_res)

            # 根据识别到的字符串的长度，确定最终结果
            # idx = -1
            # fit_len = -1
            # for i in range(0, len(rec_res_collect)):
            #     rec = rec_res_collect[i]
            #     rectxt = [rec[i][0] for i in range(len(rec))]
            #     txt_len = len(txt_crop(''.join(rectxt)))
            #     if txt_len > fit_len:
            #         fit_len = txt_len
            #         idx = i

            # 根据识别的置信度，确定最终结果
            idx = -1
            fit_score = 0
            for i in range(0, len(dt_boxes_collect)):
                sco_mean = np.array(dt_boxes_collect[i]).mean()
                if (sco_mean > fit_score):
                    fit_score = sco_mean
                    idx = i

            dt_boxes, rec_res = dt_boxes_collect[idx], rec_res_collect[idx]
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]
            if (len(txts) > 0):
                '''按照scores排序'''
                # txts, scores, dt_boxes = sort_text_score(txts, scores, dt_boxes)
                '''筛选score大于drop_score的结果,筛选的结果放到txts_rel中'''
                for idx, txt in enumerate(txts):
                    if scores is not None and scores[idx] < self.drop_score:
                        continue
                    if len(txt) < 2:
                        continue
                    txts_rel.append(txt)
                    scores_rel.append(scores[idx])
        str = ",".join(txts_rel)
        return str
        # 返回{字符串：置信度}
        # result_dic = dict(zip(txts_rel, scores_rel))
        # return result_dic

    @serving
    def serving_method(self, images, crop_info, **kwargs):
        """
        Run as a service.
        images : 图片
        crop_info ： 坐标信息，格式[左上角竖直方向坐标， 左上角水平方向坐标，竖直方向尺寸， 水平方向尺寸]
        """
        images_decode = [base64_to_cv2(image) for image in images]
        print('images_decode ori shape', np.array(images_decode[0]).shape)
        left_top_h = crop_info[0] #左上角竖直方向坐标
        left_top_w = crop_info[1] #左上角水平方向坐标
        height = crop_info[2] #竖直方向尺寸
        width = crop_info[3] #水平方向尺寸
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
        for idx, img in enumerate(images_decode):
            # 均衡化,此处的均衡化与144-151处相同
            # img = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # 裁剪
            images_decode[idx] = img[left_top_h:left_top_h+height, left_top_w:left_top_w+width]
            print("images_decode size = ", np.array(images_decode[0]).shape)
            cv2.imwrite('D:\\workspace\\PaddleOCR-release-2.3\\doc\\test\\res.jpg', images_decode[0])
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    ocr = OCRSystem()
    image_path = [
        '../../../data/photo/0.jpg',
        '../../../data/photo/1.jpg',
    ]
    res = ocr.predict(paths=image_path)
    print(res)