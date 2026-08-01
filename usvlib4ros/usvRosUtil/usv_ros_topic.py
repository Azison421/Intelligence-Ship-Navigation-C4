import time
import threading

import roslibpy
import threading

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

from usvlib4ros.usvRosUtil import USVRosbridgeClient, LogUtil
from usvlib4ros.msg.global_data import GlobalData, DictToObject


class RosSubscriberProxy:
    """rosbridge 订阅消息代理"""

    def __init__(self, topicName, msgType, callback=None ,defaultMsg=None):
        self.topicName = topicName
        self.msgType = msgType
        self.callback = callback
        self.msgData = defaultMsg

        self.cond = threading.Condition()
        self.topic = roslibpy.Topic(USVRosbridgeClient.ros, topicName, msgType)
        self.topic.subscribe(callback=self.__defaultSubscriberCallback)

    def __defaultSubscriberCallback(self, msgDict):
        with self.cond:
            try:
                self.msgData = DictToObject(**msgDict)
            except Exception as e:
                print(f"RosSubscriberProxy Error(46):\r {msgDict}")
            if self.callback is not None:
                self.callback(msgDict)
            else:
                print("Recvived message type= %s : %s" % (type(message), message))
            pass
            self.cond.notify()

    def getMsgData(self):
        """
        非阻塞方法，立刻返回最新的缓存消息。如果一直没有接收到订阅的消息时返回 None
        """
        return self.msgData

    def wait_for_message(self,timeout=None):
        """
        timeout=None:一直等待，
        阻塞方法，收到新的消息后立即返回该消息，
        超时后返回最新的历史消息或者未收到历史消息返回None
        """
        with self.rosBridgeClient.cond:
            self.rosBridgeClient.cond.wait(timeout=timeout)
        return self.msgData


class RosPublisherProxy:
    """rosbridge 发布消息代理"""

    def __init__(self,topicName,msgType):
        self.topicName = topicName
        self.msgType = msgType
        self.msgData = None
        self.rosBridgeClient = None
        self.topic = roslibpy.Topic(USVRosbridgeClient.ros, topicName, msgType)
        self.th = None

    def publish(self,msgData):
        self.topic.publish(msgData)

    def startPublicTopicThread(self, frequency= 1, topicMsg=None):
        self.th = threading.Thread(target=self.__run, args=(1.0/frequency, topicMsg))
        self.th.setDaemon(True)
        self.th.start()

    def __run(self, loopTime, topicMsg):
        """定时发送话题线程"""
        while USVRosbridgeClient.ros.is_connected:
            msg = topicMsg.to_dict()
            self.publish(msg)
            time.sleep(loopTime)






