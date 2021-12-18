import datetime
import http.client
import io
from io import BytesIO
import logging

import pycurl

from pycurl_requests import exceptions
from pycurl_requests import models
from pycurl_requests import structures

try:
    from urllib3.util.timeout import Timeout
except ImportError:
    # Timeout not supported
    Timeout = None


# For DEBUGFUNCTION callback
CURLINFO_TEXT = 0
CURLINFO_HEADER_IN = 1
CURLINFO_HEADER_OUT = 2

# Loggers
LOGGER = logging.getLogger('curl')
LOGGER_TEXT = LOGGER.getChild('text')
LOGGER_HEADER_IN = LOGGER.getChild('header_in')
LOGGER_HEADER_OUT = LOGGER.getChild('header_out')
DEBUGFUNCTION_LOGGERS = {LOGGER_TEXT, LOGGER_HEADER_IN, LOGGER_HEADER_OUT}

VERSION_INFO = pycurl.version_info()

#pycurl.py > Request > send

# insert pycurl properties needed for corp networks before final call to pycurl.perform
def send(self):
        try:
            # Avoid urlparse/urlsplit as they only support RFC 3986 compatible URLs
            scheme, _ = self.prepared.url.split(':', 1)
        except ValueError:
            raise exceptions.MissingSchema('Missing scheme for {!r}'.format(self.prepared.url))

        supported_protocols = VERSION_INFO[8]
        if scheme.lower() not in supported_protocols:
            raise exceptions.InvalidSchema('Unsupported scheme for {!r}'.format(self.prepared.url))

        # Request
        self.curl.setopt(pycurl.URL, self.prepared.url)

        if self.prepared.method:
            self.curl.setopt(pycurl.CUSTOMREQUEST, self.prepared.method)

        if self.prepared.method == 'HEAD':
            self.curl.setopt(pycurl.NOBODY, 1)

        # HTTP server authentication
        self._prepare_http_auth()

        self.curl.setopt(pycurl.HTTPHEADER, ['{}: {}'.format(n, v) for n, v in self.prepared.headers.items()])

        if self.prepared.body is not None:
            if isinstance(self.prepared.body, str):
                body = io.BytesIO(self.prepared.body.encode('iso-8859-1'))
            elif isinstance(self.prepared.body, bytes):
                body = io.BytesIO(self.prepared.body)
            else:
                body = self.prepared.body

            self.curl.setopt(pycurl.UPLOAD, 1)
            self.curl.setopt(pycurl.READDATA, body)

        content_length = self.prepared.headers.get('Content-Length')
        if content_length is not None:
            self.curl.setopt(pycurl.INFILESIZE_LARGE, int(content_length))

        # Response
        self.curl.setopt(pycurl.HEADERFUNCTION, self.header_function)
        self.curl.setopt(pycurl.WRITEDATA, self.response_buffer)

        # Options
        if self.connect_timeout is not None:
            timeout = int(self.connect_timeout * 1000)
            self.curl.setopt(pycurl.CONNECTTIMEOUT_MS, timeout)

        if self.read_timeout is not None:
            timeout = int(self.read_timeout * 1000)
            self.curl.setopt(pycurl.TIMEOUT_MS, timeout)

        if self.allow_redirects:
            self.curl.setopt(pycurl.FOLLOWLOCATION, 1)
            self.curl.setopt(pycurl.POSTREDIR, pycurl.REDIR_POST_ALL)
            self.curl.setopt(pycurl.MAXREDIRS, self.max_redirects)

        # Logging
        if any((l.isEnabledFor(logging.DEBUG) for l in DEBUGFUNCTION_LOGGERS)):
            self.curl.setopt(pycurl.VERBOSE, 1)
            self.curl.setopt(pycurl.DEBUGFUNCTION, debug_function)
            
        #### PATCH START ####
        
        self.curl.setopt(pycurl.PROXY, 'http://someproxy.com')
        self.curl.setopt(pycurl.PROXYAUTH, pycurl.HTTPAUTH_ANYSAFE)
        self.curl.setopt(pycurl.PROXYUSERPWD,':')
        self.curl.setopt(pycurl.HTTPPROXYTUNNEL, True)
        print('_pycurl.py send: curl options patched!')
        
        #### PATCH END ####

        return self.perform()