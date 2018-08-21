# -*- coding:utf-8 -*-
import numpy as np

class A:
    def __init__(self, name , age , dic):
        self.name = name
        self.age = age
        self.rng = np.random.RandomState(1234)
        self.__dict__.update(dic)

    def P(self):
        print("name = ",self.name)
        print("age = ",self.age)
        for key in self.__dict__:
            print("key = ",key , "  value = ",self.__dict__[key])




if __name__ == "__main__":
    dic = {}
    dic["gender"] = "male"
    dic["friends"] = ["A","B","C","D","E"]
    a = A("John",25,dic)
    a.P()

