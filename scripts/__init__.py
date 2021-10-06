import sys
import os

# add scripts folder to path
# these are the main scripts
scriptsDir = os.path.dirname(__file__)
kintoneDir = os.path.join(scriptsDir, 'kintone', '')
sys.path.append(scriptsDir)
sys.path.append(kintoneDir)

from configparser import ConfigParser
_config = ConfigParser()
_config.read('config.ini', encoding='utf_8')
if 'pycurl_config' in _config.sections() and _config.getboolean('pycurl_config','use_pycurl'):
    
    # add scripts/pycurl-requests-0.2.1/pycurl_requests to path
    # needed to replace 'requests' with 'pycurl' (required to access external SSPI/SSO services, ie kintone, from within corporate networks)
    pcDir = os.path.join(scriptsDir, 'pycurl', "pycurl-requests-0.2.1", "")
    sys.path.append(pcDir)

    # add scripts/pycurl-requests-0.2.1/pycurl_requests to path
    # needed to set corporate network proxy settings for 'pycurl'
    pcpDir = os.path.join(scriptsDir, 'pycurl', "pycurl-requests-patch", "")
    sys.path.append(pcpDir)

    # Override requests - all requests calls will be replaced with pyCurl from now!
    import pycurl_requests
    import pycurl_requests_api_patch
    pycurl_requests.patch_requests()
    #DEBUG
    #print("Requests patch complete!!")
    import requests

    # Additional - patch requests.api methods to include our corporate network proxy settings 
    requests.request = pycurl_requests_api_patch.request
    requests.head = pycurl_requests_api_patch.head
    requests.get = pycurl_requests_api_patch.get
    requests.post = pycurl_requests_api_patch.post
    requests.put = pycurl_requests_api_patch.put
    requests.patch = pycurl_requests_api_patch.patch
    requests.delete = pycurl_requests_api_patch.delete

# Load basic query API 'SOURCE'
from .PQ import SOURCE
