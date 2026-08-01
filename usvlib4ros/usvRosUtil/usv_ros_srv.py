import json
import time

import roslibpy

from usvlib4ros.usvRosUtil import USVRosbridgeClient,LogUtil


class RosSrvCallProxy:
    """
    调用ros服务
    """
    def __init__(self,serviceName=None,srvType=None):
        self.serviceName = serviceName
        self.srvType = srvType
        self.callback = None
        self.service = roslibpy.Service(USVRosbridgeClient.ros, self.serviceName, self.srvType)

    def callService(self, request, timeout=3):
        """调用服务，阻塞到服务返回"""
        response = self.service.call(request, timeout=timeout)
        return response


class RosSrvAdvertiseProxy:
    """
    注册ros服务
    """
    def __init__(self,serviceName=None,srvType=None,callback=None):
        self.serviceName = serviceName
        self.srvType = srvType
        self.callback = None
        self.isAdvertiseMode = False
        self.service = roslibpy.Service(USVRosbridgeClient.ros, serviceName, srvType)
        self.service.advertise(self.__defaultCallback)

    def __defaultCallback(self, request):
        response = {}
        if self.callback is not None:
            response = self.callback(request)
        return response


