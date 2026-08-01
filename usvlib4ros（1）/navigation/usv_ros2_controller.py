import json
import time
import threading
import roslibpy

"""
ros环境下安装 ros-melodic-rosbridge-server (melodic是ros版本)
apt-get install ros-melodic-rosbridge-server

启动 roscore 和 rosbridge
roscore
roslaunch rosbridge_server rosbridge_websocket.launch

rosbridge 有 tcp，udp ，websocket 三种启动方式，默认端口 9090
roslibpy 仅支持与 rosbridge的websocket 连接

在非ros环境下，安装roslibpy 
pip install roslibpy -i https://pypi.tuna.tsinghua.edu.cn/simple

"""

from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.usvRosUtil import RosSrvCallProxy, RosSrvAdvertiseProxy, RosSubscriberProxy, RosPublisherProxy,USVRosbridgeClient
from usvlib4ros.usvRosUtil.ros_debug_logger import debug_logger, configure_debug_output
from usvlib4ros.msg.global_data import GlobalData,DictToObject,Constants
from usvlib4ros.msg.parameter import Parameter


class Ros2Controller:

    def __init__(self, host, port, deviceId, globalData:GlobalData, 
                 enable_debug: bool = True,
                 laser_debug_freq: float = 0.5,
                 device_debug_freq: float = 1.0,
                 control_debug_freq: float = 2.0):
        self.host = host
        self.port = port

        USVRosbridgeClient.initRoslibpyLogger()
        USVRosbridgeClient.initUSVRosBridgeConnection(host=host,port=port)

        self.deviceId = deviceId
        self.globalData = globalData
        
        # 配置调试输出
        self.enable_debug = enable_debug
        if enable_debug:
            configure_debug_output(
                laser_frequency=laser_debug_freq,
                device_status_frequency=device_debug_freq,
                control_frequency=control_debug_freq,
                enable_all=True
            )
            # 设置具体话题的调试频率
            debug_logger.set_topic_frequency(f"usv/{deviceId}/laser/scan", laser_debug_freq)
            debug_logger.set_topic_frequency(f"usv/device/status/{deviceId}", device_debug_freq)
            debug_logger.set_topic_frequency(f"usv/scada/status/{deviceId}", control_debug_freq)

        """订阅消息"""

        """订阅设备状态和控制"""
        self.deviceStatusSubscriber = RosSubscriberProxy(topicName=f"usv/device/status/{deviceId}",
                                                    msgType="message_pkg/DeviceStatus",
                                                    callback=self.__listenerDeviceStatusCallback)

        self.scadaStatusSubscriber = RosSubscriberProxy(topicName=f"usv/scada/status/{deviceId}",
                                                         msgType="message_pkg/ScadaStatus",
                                                         callback=self.__listenerScadaStatusCallback)

        """订阅激光雷达"""
        self.laserSubscriber = RosSubscriberProxy(topicName=f"usv/{deviceId}/laser/scan",
                                                    msgType="sensor_msgs/LaserScan",
                                                    callback=self.__listenerLaserCallback)

        """独立线程发布消息"""
        self.navigationStatusPublisher = RosPublisherProxy(topicName=f"usv/navigation/status/{deviceId}",
                                                         msgType="message_pkg/NavigationStatus",)
        self.navigationStatusPublisher.startPublicTopicThread(frequency=10,topicMsg=globalData.navigation_output_data)

        """调用ros服务"""
        self.deviceControllerSrvCall = RosSrvCallProxy(serviceName=f"usv/server/{deviceId}",
                                                       srvType="message_pkg/DeviceController")


        """根据autoCommand信息请求route"""
        th = threading.Thread(target=self.topicHandler)
        th.setDaemon(True)
        th.start()
        pass

    def __listenerDeviceStatusCallback(self, msgDict):
        # 调试打印设备状态
        if self.enable_debug:
            debug_logger.log_device_status(f"usv/device/status/{self.deviceId}", msgDict)
            
        self.globalData.device_data = DictToObject(**msgDict)
        parameterAdjusterDict = msgDict['parameter_adjuster']
        version = parameterAdjusterDict['version']
        subVersion = parameterAdjusterDict['subversion']
        data = parameterAdjusterDict['data']
        for parameterDict in data:
            name = parameterDict['name']
            parameter = GlobalData.getInstance().parameterAdjustMap.get(name)
            if parameter is not None:
                parameter.read(parameterDict)

        parameterMonitorDict = msgDict['parameter_monitor']
        version = parameterMonitorDict['version']
        subVersion = parameterMonitorDict['subversion']
        data = parameterMonitorDict['data']
        for parameterDict in data:
            name = parameterDict['name']
            parameter = GlobalData.getInstance().parameterMonitorMap.get(name)
            if parameter is not None:
                parameter.read(parameterDict)

    def __listenerScadaStatusCallback(self, msgDict):
        # 调试打印SCADA状态（控制指令相关）
        if self.enable_debug:
            debug_logger.log(f"usv/scada/status/{self.deviceId}", msgDict, 
                           "message_pkg/ScadaStatus", "SCADA Control Status")
        self.globalData.scada_data = DictToObject(**msgDict)

    def __listenerLaserCallback(self, msgDict):
        # 调试打印激光雷达数据
        if self.enable_debug:
            debug_logger.log_laser_scan(f"usv/{self.deviceId}/laser/scan", msgDict)
        self.globalData.laser_data = DictToObject(**msgDict)

    def initParameterList(self):
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Parameter_Init,
            "data": "some",
        }
        response = self.deviceControllerSrvCall.callService(request)
        print(response)
        return response['code'] == 1

    def typeCheck(self, value, registerType):
        arrayValue = value
        sameType = True
        if type(value).__name__ != 'list' and type(value).__name__ != 'tuple':
            arrayValue = [value]

        for data in arrayValue:
            dataType = type(data).__name__
            if dataType != registerType:
                if registerType == 'str' and dataType == 'unicode':
                    continue
                elif registerType == 'float' and dataType == 'int':
                    continue
                else:
                    sameType = False
                    break
            pass
        pass
        return sameType

    def registAdjustParameter(self, name, valueType, defaultValue):
        """
        :param name: str,eg "/usv/auto/nav/param/maxSpeed"
        :param defaultValue: valyeType类型的默认值 或 valueType类型的数组或元组
        :param valueType: str in ['int','str','bool','float']

        eg:
        registAdjustParameter("/usv/status","int",1)
        registAdjustParameter("/usv/status","int",[1])
        registAdjustParameter("/usv/status","int",[1,2])
        registAdjustParameter("/usv/status","int",(1,2))
        :return:
        """

        if valueType not in ['int', 'bool', 'str', 'float']:
            raise Exception("valueType %s not in ['int','bool','str','float'] " % (valueType))

        if type(name).__name__ != 'str':
            raise Exception("Name error . %s is not str." % (name))

        if type(defaultValue).__name__ != 'list' and type(defaultValue).__name__ != 'tuple':
            defaultValue = [defaultValue]

        typeOk = self.typeCheck(value=defaultValue, registerType=valueType)
        if typeOk is False:
            raise Exception("Defaultvalue %s is not instance of %s ." % (defaultValue, valueType))

        value = json.dumps(defaultValue)
        parameter = {"name": name, "type": valueType, "value": value}
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Parameter_Init,
            "data": f"{parameter}",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            LogUtil.error(response['data'])

        return response['code'] == 1, parameter
        pass

    def registParameter(self, parameter):
        parameterDict = parameter.write()
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Parameter_Register_Adjuster,
            "data": f"{parameterDict}",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            LogUtil.error(response['data'])

        return response['code'] == 1, parameter
        pass

    def registMonitorParameter(self, parameter):
        parameterDict = parameter.write()
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Parameter_Register_Monitor,
            "data": f"{parameterDict}",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            LogUtil.error(response['data'])

        return response['code'] == 1, parameter
        pass

    def adjustParameter(self, parameter):
        parameterDict = parameter
        if type(parameter).__name__ == 'dict':
            parameterDict = parameter
        elif type(parameter).__name__ == 'Parameter':
            parameterDict = parameter.write()

        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Parameter_Adjuster,
            "data": f"{parameterDict}",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            LogUtil.error(response['data'])

        return response['code'] == 1, parameter
        pass

    def getRoute(self):
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Get_Route,
            "data": "",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            LogUtil.error(response['data'])

            return self.globalData.route
        else:
            data = response['data']
            data = json.loads(data)
            return DictToObject(**data)
        pass

    def topicHandler(self):
        while True:
            try:
                routeUpdateTime, oldRoute = self.globalData.device_data.route_version, self.globalData.route
                oldRouteCreateTime = oldRoute.version
                if routeUpdateTime != oldRouteCreateTime:
                    """系统导航路线更新时间与 本地缓存路线更新时间不一致，请求最新路线"""
                    route = self.getRoute()
                    """通知导航系统路线已更新 """
                    self.globalData.updateRouteInfo(routeUpdateTime=route.version, route=route)
            except Exception as e:
                LogUtil.error(e)
            finally:
                time.sleep(0.1)
            pass
        pass


if __name__ == "__main__":

    USVRosbridgeClient.initRoslibpyLogger()
    USVRosbridgeClient.initUSVRosBridgeConnection("192.168.3.35", 9090)
    """订阅消息示例"""

