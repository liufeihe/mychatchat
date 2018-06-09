# -*- coding:utf-8 -*-

import os
import urllib
import tornado.web
from common.config import TEMPLATE_PATH


class homeHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        # self.write('Hello, world!')
        self.render(TEMPLATE_PATH+'home.html')
