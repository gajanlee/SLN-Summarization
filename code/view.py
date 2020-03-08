#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   view.py
@Time    :   2020/02/05 13:14:33
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from pathlib import Path

def generate_ppt():
    Path("pages/template.html").read_text().format({
        "title": title,
        "author": author,
        "keywords": ", ".join(keywords),
        "abstract": abstract_items,
        "text_items": text_items,
        date: "",
    })