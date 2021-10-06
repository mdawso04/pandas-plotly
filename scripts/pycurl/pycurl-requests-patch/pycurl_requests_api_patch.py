from pycurl_requests import sessions
from configparser import ConfigParser
import pycurl

_config = ConfigParser()
_config.read('config.ini', encoding='utf_8')

def _pycurlSessionWithSettings():
    with sessions.Session() as session:
        if 'pycurl_config' in _config.sections():
            session.curl.setopt(pycurl.PROXY, _config['pycurl_config']['proxy'])
            session.curl.setopt(pycurl.PROXYAUTH, getattr(pycurl, _config['pycurl_config']['proxyauth']))
            session.curl.setopt(pycurl.PROXYUSERPWD, _config['pycurl_config']['proxyuserpwd'])
            session.curl.setopt(pycurl.HTTPPROXYTUNNEL, _config.getboolean('pycurl_config','httpproxytunnel'))
            #DEBUG
            #print(_config['pycurl_config']['proxy'])
            #print(getattr(pycurl, _config['pycurl_config']['proxyauth']))
            #print(_config['pycurl_config']['proxyuserpwd'])
            #print(_config.getboolean('pycurl_config','httpproxytunnel'))
        return session

def request(method, url, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.request(method, url, **kwargs)

def head(url, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.head(url, **kwargs)


def get(url, params=None, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.get(url, params=params, **kwargs)


def post(url, data=None, json=None, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.post(url, data=data, json=json, **kwargs)


def put(url, data=None, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.put(url, data=data, **kwargs)


def patch(url, data=None, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.patch(url, data=data, **kwargs)


def delete(url, params=None, **kwargs):
    with _pycurlSessionWithSettings() as session:
        return session.delete(url, params=params, **kwargs)
