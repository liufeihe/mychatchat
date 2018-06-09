# -*- coding:utf-8 -*-

import os
import urllib
import tornado.web
from common.config import TEMPLATE_PATH
from common import util, httpserv
from datamgr import chat_mgr
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')

a = 1


class chatHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        # self.write('Hello, world!')
        self.render(TEMPLATE_PATH+'chat.html')

    @httpserv.request
    def post(self, *args, **kwargs):
        global a
        obj = util.json_load_to_str(self.request.body)
        msg = obj.get('msg', '')
        resp = chat_mgr.ask_answer(msg)
        print resp
        # resp = '哈哈'
        a = a + 1
        # resp = 'Hello, world!'
        return httpserv.response('ok', {'msg': resp, 'count': a})
