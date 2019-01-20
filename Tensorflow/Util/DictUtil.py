# -*- coding: utf-8 -*-

classify_dict = {
  'cat': '猫', 'dog': '狗', 'car': '车', 'ship': '船',
  'person': '人'
}

def get_classify_dict(classify):
  if(classify_dict.__contains__(classify)):
    return classify_dict[classify]
  return classify