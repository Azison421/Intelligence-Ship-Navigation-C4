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
from usvlib4ros.msg.global_data import GlobalData,DictToObject,Constants
from usvlib4ros.msg.parameter import Parameter


class Ros2Controller:

    def __init__(self, host, port, deviceId, globalData:GlobalData):
        self.host = host
        self.port = port

        USVRosbridgeClient.initRoslibpyLogger()
        USVRosbridgeClient.initUSVRosBridgeConnection(host=host,port=port)

        self.deviceId = deviceId
        self.globalData = globalData

        """调用ros服务"""
        self.deviceManageSrvCall = RosSrvCallProxy(serviceName=f"usv/service/device/manage",
                                                       srvType="message_pkg/DeviceManage")
        self.create_ship()
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


    def __listenerDeviceStatusCallback(self, msgDict):
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
        self.globalData.scada_data = DictToObject(**msgDict)

    def __listenerLaserCallback(self, msgDict):
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

        typeOk = cls.typeCheck(value=defaultValue, registerType=valueType)
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

    def create_ship(self):
        request = {
            "device_id": f"{self.deviceId}",
            "action": 1,
        }
        response = self.deviceManageSrvCall.callService(request)
        LogUtil.info(response)
        return response['result'] == 1

    def reset_unity(self):
        """data : 1 导航请求复位unity环境，2 ：unity回复环境复位已完成"""
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Reset,
            "data": "1",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            err_msg = response['data']
            return False
        return True

    def set_auto_work(self):
        """data : 2 """
        request = {
            "client_id": "navigation",
            "action": Constants.Request_Action.Set_Work_Model,
            "data": "2",
        }
        response = self.deviceControllerSrvCall.callService(request)
        if response['code'] != 1:
            err_msg = response['data']
            return False
        return True






if __name__ == "__main__":

    USVRosbridgeClient.initRoslibpyLogger()
    USVRosbridgeClient.initUSVRosBridgeConnection("192.168.3.35", 9090)
    """订阅消息示例"""

