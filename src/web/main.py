# -*- coding: utf-8 -*-

import os
import tornado.ioloop
import tornado.web
from handler import *
import chat_mgr
import sys
reload(sys)
sys.setdefaultencoding('UTF-8')


settings = {
    # "cookie_secret": "asdfqwerzxcv",
    # "xsrf_cookies": True,
    # "login_url": "/user/login",
    "static_path": os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) + '/templates/static',
}


def make_app():
    return tornado.web.Application([
        (r"/", homeHandler),
        (r"/chat", chatHandler)
    ], **settings)


if __name__ == '__main__':
    # chat_mgr.init_model()

    app = make_app()
    app.listen(8001)
    print 'server is ok.'
    tornado.ioloop.IOLoop.current().start()