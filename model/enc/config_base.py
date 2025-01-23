import json
import os
from copy import deepcopy
from dataclasses import dataclass

@dataclass
class BaseConfig:
    """基礎配置類，提供配置的複製、繼承、存儲和加載功能"""

    def clone(self):
        """深拷貝當前配置，返回一個新副本"""
        return deepcopy(self)

    def inherit(self, another):
        """
        繼承給定配置中與當前配置共有的鍵值
        - 只會覆蓋共同鍵（key）的值
        :param another: 另一個 BaseConfig 對象
        """
        common_keys = set(self.__dict__.keys()) & set(another.__dict__.keys())  # 獲取共有鍵
        for k in common_keys:
            setattr(self, k, getattr(another, k))  # 設置繼承的值

    def propagate(self):
        """
        將當前配置的屬性值向下傳遞到其成員（如果成員是 BaseConfig 的子類）
        - 遞歸地繼承父配置中的值
        """
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):  # 如果成員屬性是 BaseConfig 類型
                v.inherit(self)  # 繼承父層的公共屬性
                v.propagate()  # 繼續向下傳遞

    def save(self, save_path):
        """
        將當前配置保存為 JSON 文件
        :param save_path: 文件保存路徑
        """
        dirname = os.path.dirname(save_path)
        if not os.path.exists(dirname):  # 如果目錄不存在，創建它
            os.makedirs(dirname)
        conf = self.as_dict_jsonable()  # 將配置轉為可序列化的字典
        with open(save_path, 'w') as f:  # 打開文件並保存
            json.dump(conf, f)

    def load(self, load_path):
        """
        從 JSON 文件加載配置並更新當前對象
        :param load_path: JSON 文件的路徑
        """
        with open(load_path) as f:  # 打開文件並加載配置
            conf = json.load(f)
        self.from_dict(conf)  # 使用加載的字典更新當前配置

    def from_dict(self, dict, strict=False):
        """
        使用字典更新當前配置對象
        :param dict: 要應用的字典
        :param strict: 嚴格模式，如果遇到多餘的鍵，則拋出錯誤
        """
        for k, v in dict.items():
            if not hasattr(self, k):  # 當前配置中不存在該鍵
                if strict:
                    raise ValueError(f"loading extra '{k}'")  # 嚴格模式下報錯
                else:
                    print(f"loading extra '{k}'")  # 提示加載了多餘的鍵
                    continue
            if isinstance(self.__dict__[k], BaseConfig):  # 如果屬性是 BaseConfig 類型
                self.__dict__[k].from_dict(v)  # 遞歸更新子配置
            else:
                self.__dict__[k] = v  # 更新值

    def as_dict_jsonable(self):
        """
        將當前配置轉換為 JSON 可序列化的字典
        - 只保留可序列化的屬性
        :return: 可序列化的字典
        """
        conf = {}
        for k, v in self.__dict__.items():
            if isinstance(v, BaseConfig):  # 如果是 BaseConfig 類型
                conf[k] = v.as_dict_jsonable()  # 遞歸轉換子配置
            else:
                if jsonable(v):  # 檢查是否可序列化
                    conf[k] = v
                else:
                    # 忽略不可序列化的屬性
                    pass
        return conf



def jsonable(x):
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False
