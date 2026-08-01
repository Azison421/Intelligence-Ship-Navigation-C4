import threading
import time
from types import SimpleNamespace


class DictToObject:
    """有一定的适用范围，仅用于已测试过的对象：2D雷达、device、nav、scada 话题对象"""
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            if isinstance(value,dict):
                setattr(self,key,DictToObject(**value))
            elif isinstance(value,list):
                if key == "ranges":
                    """特异化封装，提速: laser_scan 对象中 ranges数组"""
                    setattr(self, key, value)
                else:
                    setattr(self, key, [])
                    for v in value:
                        if isinstance(v,dict):
                            getattr(self,key).append(DictToObject(**v))
                        else:
                            #TODO 未测试对象可能存在错误
                            getattr(self, key).append(value)
                            break
                        pass
                pass
            else:
                setattr(self, key , value)
            pass
        pass

    def __str__(self):
        return f"{self.__dict__}"

    def to_dict(self):
        """将对象转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                # 递归转换嵌套的 DictToObject
                result[key] = value.to_dict()
            elif isinstance(value, list):
                # 处理列表
                result[key] = [
                    item.to_dict() if isinstance(item, DictToObject) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    pass


class Constants:

    class Request_Action:

        Set_Work_Model = 1
        Set_Route = 10
        Get_Route = 11
        Set_Route_Start_Index = 12
        Set_Task = 30
        Reset = 31

        Parameter_Init = 20
        Parameter_Register_Adjuster = 21
        Parameter_Register_Monitor = 22
        Parameter_Adjuster = 23
        pass

    class WorkMode:
        AutoReturn = -1
        Ready = 0
        Manual = 1
        Auto = 2

    pass


class Point(DictToObject):

    def __init__(self):
        self.lng = 0.0
        self.lat = 0.0
        self.high = 0.0
        self.cruiseSpeed = 0.0
        pass


class GlobalData:

    Instance = None

    @classmethod
    def getInstance(cls):
        if cls.Instance is None:
            cls.Instance = GlobalData()

        return cls.Instance

    def __init__(self):
        """"""

        """input"""
        self.device_data = DictToObject(**{
            "time": 0.0,  # s,float
            "work_model": 0,    # (0, 1, 2, -1) 0：就绪，1：遥控；2：自动
            "pose_model": 2,    # (2, 1), default 2
            "task_status": 0,   # (0, 1), 0 停止导航或结束训练;1 开启导航或训练
            "route_version": 0,     # s ,int

            "throttle_percent": 0,  # [-100,100]
            "rudder_percent": 0,    # [-100,100]

            "reset_status": 0,      # (0 , 1, 2); 0：无动作，1：开始复位；2：复位结束
            "reset_request_time": 0.0,  # 设置 reset_status = 1 的时间，s float
            "parameter_monitor": {
                "version": "",
                "subversion":"",
                "data":[],
            },
            "parameter_adjuster": {
                "version": "",
                "subversion": "",
                "data": [],
            }
        })

        self.scada_data = DictToObject(**{
            "time": 0.0,  # s,float
            "manual": {
                "handle_ok":0,
                "throttle_percent": 0,
                "rudder_percent": 0,
            },
            "pose": {
                "lng":0.0,
                "lat": 0.0,
                "high": 0.0,
                "lng": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "speed": 0.0,
                "rotate_speed": 0.0,
            },
        })

        self.laser_data = DictToObject(**{})

        """output"""
        self.navigation_output_data = DictToObject(**{
            "time" : 0.0,             # s
            "advise_throttle": 0,    # [-100,100]
            "advise_rudder": 0,      # [-100,100]
            "advise_heading": 0.0,  # °
            "point_index": 0,
            "distance": 0.0,        # m

            "e": 0,
            "step": 0,
            "score": 0,
            "loss": 0.0,
            "max_e": 0,
            "status": 0,
        })

        """temp"""
        self.request = DictToObject(**{
            "client_id":"",
            "action": 0,
            "data": "",
        })

        self.response = DictToObject(**{
            "client_id": "",
            "action": 0,
            "code":0,
            "data": "",
        })

        self.route = DictToObject(**{
            "id": "",
            "name": "",
            "version": 0.0,
            "start_index": 0,
            "points": [],
            "obstacles": [],
        })

        self.__routeLock = threading.Lock()
        self.__routeUpdateTime = 0.0  # 单位 s, 系统缓存路线更新时间
        self.parameterAdjustMap = {}
        self.parameterMonitorMap = {}

    def getRouteInfo(self):
        try:
            self.__routeLock.acquire()
            return self.__routeUpdateTime, self.route
        finally:
            self.__routeLock.release()

    def updateRouteInfo(self, route, routeUpdateTime):
        try:
            self.__routeLock.acquire()
            self.route = route
            self.__routeUpdateTime = routeUpdateTime
        finally:
            self.__routeLock.release()

    def updateThrottleRudderOutput(self,adviseSpeed, adviseRotate, advisedHeading,nextPointIndex,shipToNextWPDistance):
        self.navigation_output_data.advise_throttle = int(adviseSpeed)
        self.navigation_output_data.advise_rudder = int(adviseRotate)
        self.navigation_output_data.advise_heading = advisedHeading
        self.navigation_output_data.point_index = int(nextPointIndex)
        self.navigation_output_data.distance = shipToNextWPDistance
        self.navigation_output_data.time = time.time()
        pass

    def updateAlgorithmOutput(self,e, step, score,loss,max_e,status):
        self.navigation_output_data.e = int(e)
        self.navigation_output_data.step = int(step)
        self.navigation_output_data.score = int(score)
        self.navigation_output_data.loss = loss
        self.navigation_output_data.max_e = int(max_e)
        self.navigation_output_data.status = int(status)
        self.navigation_output_data.time = time.time()
        pass


if __name__ == "__main__":
    data = {'id': 'Single Point', 'name': '单点', 'version': 1779327963679, 'desc': '', 'start_index': 0, 'points': [{'name': '', 'lng': 116.174315746387, 'lat': 39.7388280594258, 'high': 0.0, 'cruiseSpeed': 1.5}], 'obstacles': []}
    d_Data = DictToObject(**data)
    point = d_Data.points[0]
    lng = point.lng
