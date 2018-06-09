# -*- coding: utf-8 -*-

from elasticsearch import Elasticsearch
from elasticsearch import NotFoundError
from common.config import ES_SEARCH_SERVER, ES_INDEX_NAME


def es_delete():
    pass


def es_update():
    pass


def es_index(doc_name, key, value):
    try:
        if ES_SEARCH_SERVER:
            es = Elasticsearch([ES_SEARCH_SERVER])
            es.index(index=ES_INDEX_NAME, doc_type=doc_name, id=key, body=value, request_timeout=1)
    except Exception, e:
        raise e


def es_search(doc_name, keyword, row_limit=0):
    try:
        if ES_SEARCH_SERVER:
            es_body_dict = {
                'doc.test': {"bool": {
                    "should": [{"query_string": {"default_field": "name", "query": keyword}},
                             {"query_string": {"default_field": "desc", "query": keyword}}],
                    "boost": 1.0}},
                'doc.test.ik': {"bool": {
                    "should": [{"query_string": {"default_field": "name", "query": keyword}},
                               {"query_string": {"default_field": "desc", "query": keyword}}],
                    "boost": 1.0}}
            }
            es = Elasticsearch([ES_SEARCH_SERVER])
            es_body = {'query': es_body_dict[doc_name]}
            if row_limit > 0:
                es_body['size'] = row_limit
            es_body['from'] = 0

            result = es.search(index=ES_INDEX_NAME, doc_type=doc_name, body=es_body, request_timeout=2)
            return result['hits']['hits']
        else:
            return []
    except Exception, e:
        raise e


def es_example():
    doc_test = 'doc.test'  # no ik
    # es_index(doc_test, 'test.1', {'name': 'Brad Peter', 'desc': 'good man'})
    # es_index(doc_test, 'test.2', {'name': 'John Snow', 'desc': 'best man'})
    # es_index(doc_test, 'test.3', {'name': 'Tit Bag', 'desc': 'bad man'})
    # es_index(doc_test, 'test.4', {'name': 'Lilly Brady', 'desc': 'good woman'})
    # es_index(doc_test, 'test.5', {'name': '冯小刚', 'desc': '大导演'})
    # es_index(doc_test, 'test.6', {'name': '徐峥', 'desc': '还不错的导演'})
    # es_index(doc_test, 'test.7', {'name': '姜文', 'desc': '极好的大导演'})
    # es_index(doc_test, 'test.8', {'name': '贾樟柯', 'desc': '文艺大导演'})
    # hits = es_search(doc_test, '文艺大导演')

    doc_test_ik = 'doc.test.ik'
    # es_index(doc_test_ik, 'test.ik.1', {'name': '冯小刚', 'desc': '大导演'})
    # es_index(doc_test_ik, 'test.ik.2', {'name': '徐峥', 'desc': '还不错的导演'})
    # es_index(doc_test_ik, 'test.ik.3', {'name': '姜文', 'desc': '极好的大导演'})
    # es_index(doc_test_ik, 'test.ik.4', {'name': '贾樟柯', 'desc': '文艺大导演'})
    es_index(doc_test_ik, 'test.ik.5', {'name': '贾樟柯儿子', 'desc': '文的艺大导演'})
    es_index(doc_test_ik, 'test.ik.6', {'name': '贾樟柯孙子', 'desc': '不文的艺大导演'})
    hits = es_search(doc_test_ik, '文')

    for hit in hits:
        print 'name: '+hit['_source']['name']+', desc: '+hit['_source']['desc']


if __name__ == '__main__':
    es_example()
