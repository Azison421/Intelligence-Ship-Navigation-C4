import time
import logging
import threading
import roslibpy
import traceback
import inspect

"""
ros环境下安装 ros-melodic-rosbridge-server (melodic是ros版本)
apt-get install ros-melodic-rosbridge-server

启动 roscore 和 rosbridge
roscore
roslaunch rosbridge_server rosbridge_websocket.launch

rosbridge 有 tcp，udp ，websocket 三种启动方式，默认端口 9090
roslibpy 仅支持与 rosbridge的websocket 连接

------------------------------------------------------------

在非ros环境下，安装roslibpy 
pip install roslibpy -i https://pypi.tuna.tsinghua.edu.cn/simple

"""


class LogUtil:

    @classmethod
    def info(cls, msg):
        frame = inspect.stack()[1]
        name = frame[1].rsplit('/', 1)[-1]
        lino = frame[2]
        print("%s(%s):%s" % (name, lino, msg))

    @classmethod
    def error(cls, msg):
        traceback.print_exc()
        print(msg)

    @classmethod
    def debug(cls, msg):
        frame = inspect.stack()[1]
        name = frame[1].rsplit('/', 1)[-1]
        lino = frame[2]
        print("%s(%s):%s" % (name, lino, msg))


class USVRosbridgeClient:
    Host = "192.168.3.35"
    Port = 9090
    ros = None

    @classmethod
    def initUSVRosBridgeConnection(cls, host, port):
        cls.Host = host
        cls.Port = port
        cls.ros = roslibpy.Ros(host=cls.Host, port=cls.Port)
        cls.ros.run()
    pass

    @classmethod
    def initRoslibpyLogger(cls):
        """初始化roslibpy日志模块"""
        logging.getLogger('twisted').setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        logging.getLogger('twisted').addHandler(ch)



    def callService(self,serviceName=None,srvType=None,request=None,callback=None,errorCallback=None,timeout=0):
        """
        callback = None is Blocking mode else No-Blocking mode

        :param serviceName:
        :param srvType:
        :param request dict eg

        :param callback:
        :param errorCallback:
        :param timeout:
        :return:
        """

        self.service = roslibpy.Service(self.ros,serviceName,srvType)
        response = self.service.call(request,callback=callback,errback=errorCallback,timeout=timeout)

        if response is not None:
            #print("%s response is %s"%(serviceName,response))
            pass
        # self.service.unadvertise()
        return response

    def advertiseService(self,serviceName=None,srvType=None,callback=None):
        self.service = roslibpy.Service(self.ros, serviceName, srvType)
        self.service.advertise(callback)

    def get_param(self,param_name, default=None):
        try:
            self.ros.wait_for_service('/rosapi/get_param')
            param_value = self.ros.get_param(param_name)
            if param_value is None:
                param_value = default
        except Exception as e:
            param_value = default
        return  param_value

    pass

