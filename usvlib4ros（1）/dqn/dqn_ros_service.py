import threading
import time

from usvlib4ros import USVRosbridgeClient,LogUtil,DictToObject


class USVURL:

    #系统功能
    shipManageSrv = {"url":"usv/000/service/device/manage","msgType": "message_pkg/DeviceManage"} #根据deviceId动态创建船对象

    @classmethod
    def init(cls,deviceId="001"):

        cls.AdjustParameterTopic = {"url": "usv/%s/ctl/scada/parameter/adjustList"%(deviceId), "frequency": 1, "msgType": ""}
        cls.MonitorParameterTopic = {"url": "usv/%s/ctl/scada/parameter/monitorList"%(deviceId), "frequency": 1}
        # 输入
        cls.LaserScanTopic = {"url": "usv/%s/laser/scan"%(deviceId), "frequency": 1, "msgType": "sensor_msgs/LaserScan"}
        cls.CompressedImageTopic = {"url": "usv/%s/camera/image/compressed"%(deviceId), "frequency": 1,"msgType": "sensor_msgs/CompressedImage"}
        # cls.CameraInfoTopic = {"url": "usv/%s/camera/info"%(deviceId), "frequency": 1,"msgType": "sensor_msgs/CompressedImage"}
        cls.IMUTopic = {"url": "usv/%s/imu" % (deviceId), "frequency": 1,"msgType": "sensor_msgs/Imu"}
        # cls.GPSTopic = {"url": "usv/%s/gps" % (deviceId), "frequency": 1,"msgType": "sensor_msgs/Gps"}
        # cls.DeepImageTopic = {"url": "usv/%s/deep/image/compressed" % (deviceId), "frequency": 1, "msgType": "sensor_msgs/LaserScan"}
        # cls.DeepInfoTopic = {"url": "usv/%s/deep/info" % (deviceId), "frequency": 1, "msgType": "sensor_msgs/LaserScan"}
        # cls.DeepPointsTopic = {"url": "usv/%s/deep/points" % (deviceId), "frequency": 1, "msgType": "sensor_msgs/LaserScan"}

        # 输出
        cls.ResetStatusTopic = {"url": "usv/%s/ctl/scada/train/reset/status"%(deviceId), "frequency": 1,
                                "msgType": "message_pkg/ResetStatus"}
        cls.DqnActionTopic = {"url": "usv/%s/ctl/scada/auto/status"%(deviceId), "frequency": 50, "msgType": "message_pkg/AutoStatus"}

        cls.TrainStatusTopic = {"url": "usv/%s/ctl/scada/train/status"%(deviceId), "frequency": 10,
                                "msgType": "message_pkg/TrainStatus"}

        cls.TraninPosTopic = {"url": "usv/%s/scada/ctl/action/trainPos"%(deviceId), "frequency": 1, "msgType": "message_pkg/TrainPos"}
        cls.RouteDetailTopic = {"url": "usv/%s/ctl/scada/route/detail"%(deviceId), "frequency": 1, "msgType": "message_pkg/Route"}
        cls.PoseTopic = {"url": "usv/%s/scada/ctl/action/pose"%(deviceId), "frequency": 1, "msgType": "message_pkg/Pose"}

        cls.SetWorkModelSrv = {"url": "usv/%s/server/switchWorkModel"%(deviceId), "msgType": "message_pkg/SetWorkModel"}
        cls.ResetSrv = {"url": "usv/%s/server/requestReset"%(deviceId), "msgType": "message_pkg/CommonCmd"}
        cls.TrainActionSrv = {"url": "usv/%s/server/train/action"%(deviceId), "msgType": "message_pkg/CommonCmd"}
    pass


class RosSubscriberProxy:
    """rosbridge 订阅消息代理"""

    def __init__(self,):
        self.topicName = None
        self.msgType = None
        self.msgData = None
        self.rosBridgeClient = None
        self.callback = None

    def subscriber(self,topicName,msgType,callback=None):
        """
        :param topicName:str,话题名称
        :param msgType:str,消息类型，
        :param callback:回调函数，非None时收到消息后调用，
        :return:
        """
        self.topicName = topicName
        self.msgType = msgType
        self.callback = callback

        self.rosBridgeClient = USVRosbridgeClient()
        self.rosBridgeClient.subscriber(topicName=topicName,msgType=msgType,callback=self.__rosBridgeCallback)

    def getMsgData(self):
        """
        非阻塞方法，立刻返回最新的缓存消息。如果一直没有接收到订阅的消息时返回 None
        """
        return self.msgData

    def wait_for_message(self,timeout=None,callback=None):
        """
        timeout=None:一直等待，
        callback 非空时，先执行callback；后返回。
        阻塞方法，收到新的消息后立即返回该消息，
        超时后返回最新的历史消息或者未收到历史消息返回None
        """
        self.callback = callback
        oldData = self.msgData
        with self.rosBridgeClient.cond:
            self.rosBridgeClient.cond.wait(timeout=timeout)
        return self.msgData

    def __rosBridgeCallback(self,msgDict):
        # LogUtil.info(msgDict)
        with self.rosBridgeClient.cond:
            self.msgData = DictToObject(**msgDict)
            if self.callback is not None :
                self.callback(self.msgData)
            self.rosBridgeClient.cond.notify()


lock = threading.Lock()
lock.acquire()

lock.release()


class RosSrvProxy:

    def __init__(self):
        self.rosBridgeClient = None
        pass

    def callService(self,serviceName,srvType,request,timeout=5):
        self.rosBridgeClient = USVRosbridgeClient()
        resp = self.rosBridgeClient.callService(serviceName=serviceName,srvType=srvType,request=request,timeout=timeout)
        return DictToObject(**resp)

    def advertiseService(self,serviceName,srvType,callback):
        self.rosBridgeClient = USVRosbridgeClient()
        resp = self.rosBridgeClient.advertiseService(serviceName=serviceName,srvType=srvType,callback=callback)

class DQN_ROS_Service:
    """"""
    """订阅话题"""
    laserScanSubscriber = RosSubscriberProxy()
    resetStatusSubscriber = RosSubscriberProxy()
    compressedImageSubscriber = RosSubscriberProxy()
    trainPosSubscriber = RosSubscriberProxy()
    routeSubscriber = RosSubscriberProxy()
    poseSubscriber = RosSubscriberProxy()

    actionPublisher = None
    statusPublisher = None

    actionMsg = {'adviseThrottle':0,'adviseRudder':0,'adviseHeading':0.0,'pointIndex':0,"distance":0.0}
    statusMsg = {"e":0,"step":0,"score":0,"loss":0,"maxE":0,"status":1} #status : 1 就绪，2 ： 训练中，3 部署

    trainStatusMsg = {'unityCmd':0,'status':0,'cycle':0}

    resetSrvProxy = RosSrvProxy()
    setAutoSrvProxy = RosSrvProxy()
    trainActionProxy = RosSrvProxy()

    """常量"""
    RESET_NONE = 0
    RESET_START = 1
    RESET_FINISH = 2

    TRAIN_START = 1
    TRAIN_READY = 0


    @classmethod
    def startService(cls,rosHost="",deviceId="001"):
        """"""
        USVRosbridgeClient.Host = rosHost
        USVURL.init(deviceId)

        cls.callShipManageSrv(deviceId=deviceId)

        cls.actionPublisher = USVRosbridgeClient()
        cls.statusPublisher = USVRosbridgeClient()

        """订阅LaserScan数据"""
        cls.laserScanSubscriber.subscriber(topicName=USVURL.LaserScanTopic['url'],
                                    msgType=USVURL.LaserScanTopic['msgType'])

        """训练模式需要功能："""
        cls.resetStatusSubscriber.subscriber(topicName=USVURL.ResetStatusTopic['url'],
                                    msgType=USVURL.ResetStatusTopic['msgType'])

        cls.compressedImageSubscriber.subscriber(topicName=USVURL.CompressedImageTopic['url'],
                                             msgType=USVURL.CompressedImageTopic['msgType'])

        cls.trainPosSubscriber.msgData = DictToObject(**{"x":0,"y":0,"z":0,"targetX":0,"targetY":0,"targetZ":0})
        cls.trainPosSubscriber.subscriber(topicName=USVURL.TraninPosTopic['url'],
                                             msgType=USVURL.TraninPosTopic['msgType'])

        cls.routeSubscriber.subscriber(topicName=USVURL.RouteDetailTopic['url'],
                                          msgType=USVURL.RouteDetailTopic['msgType'])

        cls.poseSubscriber.subscriber(topicName=USVURL.PoseTopic['url'],
                                          msgType=USVURL.PoseTopic['msgType'])

        cls.actionPublisher.startPublicTopicThread(topicName=USVURL.DqnActionTopic['url'],frequency=USVURL.DqnActionTopic['frequency'],
                                                   msgType=USVURL.DqnActionTopic['msgType'],topicMsg=cls.actionMsg)

        cls.statusPublisher.startPublicTopicThread(topicName=USVURL.TrainStatusTopic['url'],frequency=USVURL.TrainStatusTopic['frequency'],
                                                   msgType=USVURL.TrainStatusTopic['msgType'],topicMsg=cls.statusMsg)
        pass

    @classmethod
    def registerTrainActionProxy(cls):
        """注册开始训练服务，等待unity按下开始训练按钮，调用本服务"""
        cls.trainActionProxy.advertiseService(serviceName=USVURL.TrainActionSrv['url'],srvType=USVURL.TrainActionSrv['msgType'],callback=cls.trainActionServiceCallback)
        pass

    @classmethod
    def callShipManageSrv(cls,deviceId):
        """
        rossrv info message_pkg/DeviceManage
        string deviceId
        int8 action
        ---
        int8 result

        action= 1: 使用deviceId注册船对象，如果deviceId已注册则无操作 ; action = 0 使用deviceid注销船对象，注销后deviceid相关话题无输出。
        """
        rosbridgeClient = USVRosbridgeClient()
        request = {'deviceId': deviceId,'action':1}
        response = rosbridgeClient.callService(USVURL.shipManageSrv['url'],USVURL.shipManageSrv['msgType'],request)
        return response     #{'reqult':1}

    @classmethod
    def updateTrainStatus(cls,e,step,score,loss,maxE,status):
        cls.statusMsg['e'] = e
        cls.statusMsg['score'] = score
        cls.statusMsg['step'] = step
        cls.statusMsg['loss'] = loss
        cls.statusMsg['maxE'] = maxE
        cls.statusMsg['status'] = status

    @classmethod
    def trainActionServiceCallback(cls,request,response):
        print(request)
        cls.trainStatusMsg['unityCmd'] = request['value']

        if cls.trainStatusMsg['unityCmd'] == 1:
            """异步调用setWorkModeSrv，设置为自动模式"""
            t = threading.Thread(target=cls.setWorkModel)
            t.setDaemon(True)
            t.start()
            pass

        response['result'] = 1
        return True
        pass

    @classmethod
    def setWorkModel(cls):
        response = cls.setAutoSrvProxy.callService(USVURL.SetWorkModelSrv['url'], USVURL.SetWorkModelSrv['msgType'],
                                                   {"workModel": 2}, timeout=5)

    @classmethod
    def isTrainning(cls):
        return cls.trainStatusMsg['unityCmd'] == 1

    @classmethod
    def updateVehicleAction(cls,twist,heading,distance, index=0):
        cls.actionMsg['adviseThrottle'] = int(twist.linear.x)
        cls.actionMsg['adviseRudder'] = int(twist.angular.z)
        cls.actionMsg['adviseHeading'] = heading
        cls.actionMsg['distance'] = distance
        cls.actionMsg['pointIndex'] = index
        #print(cls.actionMsg)
        pass

    @classmethod
    def get_dest_position(cls):
        """返回目标点坐标 x，y """
        return cls.trainPosSubscriber.msgData.targetX,cls.trainPosSubscriber.msgData.targetZ

    @classmethod
    def wait_dest_position_refresh(cls):
        """返回目标点坐标 x，y """
        cls.wait_for_message_trainPos()
        return cls.get_dest_position()

    @classmethod
    def wait_for_message_laserScan(cls,timeout=5):
        return cls.laserScanSubscriber.wait_for_message(timeout=timeout)

    @classmethod
    def wait_for_message_trainPos(cls,timeout=5):
        return cls.trainPosSubscriber.wait_for_message(timeout=timeout)

    @classmethod
    def wait_for_message_compressedImage(cls, timeout=5):
        return cls.compressedImageSubscriber.wait_for_message(timeout=timeout)

    @classmethod
    def get_route(cls):
        return cls.routeSubscriber.msgData

    @classmethod
    def get_pose(cls):
        return cls.poseSubscriber.msgData

    @classmethod
    def wait_for_service_reset(cls,timeout=5):
        """
        call resetAction.srv and wait resetStatus goto cls.RESET_FINISH
        """
        request = {"value":cls.RESET_START}       #resetAction : 1 开始复位， 2 : 复位结束
        response = cls.resetSrvProxy.callService(USVURL.ResetSrv['url'],USVURL.ResetSrv['msgType'],request,timeout=timeout)

        if response.result == 0:
            raise Exception("%s service call timeout ."%(USVURL.ResetSrv['url']))
        pass
    pass


if __name__=="__main__":

    USVRosbridgeClient.Host = "192.168.3.119"
    USVRosbridgeClient.Port = 9090

    DQN_ROS_Service.routeSubscriber.subscriber(topicName=USVURL.RouteDetailTopic['url'],
                                   msgType=USVURL.RouteDetailTopic['msgType'])

    DQN_ROS_Service.poseSubscriber.subscriber(topicName=USVURL.PoseTopic['url'],
                                   msgType=USVURL.PoseTopic['msgType'])

    gpsSubscriber = RosSubscriberProxy()
    gpsSubscriber.subscriber(topicName="usv/001/sensor",msgType="vehicle_msgs_pkg/SensorInfo")

    # proxy = RosSrvProxy()
    # proxy.callService(USVURL.SetWorkModelSrv['url'],USVURL.SetWorkModelSrv['msgType'],{"workModel":2})
    while True:
        # data = DQN_ROS_Service.wait_for_message_laserScan(1)
        # print(data)
        route = DQN_ROS_Service.get_route()
        print(route.__str__())
        pose = DQN_ROS_Service.get_pose()
        print(pose.__str__())
        # resetStatus = DQN_ROS_Service.wait_for_service_reset(5)
        # print(resetStatus)
        time.sleep(1)
        # data = DQN_ROS_Service.wait_for_message_compressedImage(timeout=1)
        # if data is not None:
        #     format = data.format
        #     img8S = data.data
        #     img8 = np.fromstring(img8S,np.uint8)
        #     img = cv2.imdecode(img8,cv2.IMAGE_CO)
        #     file = "1.%s"%(format)
        #     cv2.imwrite(file,img8)

