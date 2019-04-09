# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:40:57 2019

@author: boivi
"""

from flask import Flask

app = Flask(__name__)

from app_pkg import routes
