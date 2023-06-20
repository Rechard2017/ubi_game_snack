import threading
import grpc
import sys
from pathlib import Path
import numpy as np
import argparse
import time
import json
from datetime import datetime
import random
import pandas as pd

_current_root = str(Path(__file__).resolve().parents[1])
sys.path.append(_current_root)
sys.path.append('.')
print(_current_root)
import contest.snake_pb2 as dealer_pb2
import contest.snake_pb2_grpc as rpc
from lib.simple_logger import simple_logger,file_logger
sys.path.append('/data1/qzb/client_new')
# from snack_function_new_version import MyStrategy, check_intime
# from snack_function_version2 import MyStrategy, check_intime
from snack_function_version3 import MyStrategy, check_intime

class Client(object):
    def __init__(self, username: str, key: str, logger, address='139.196.39.76', port=7777):
        self.username = username
        self.key = key
        self.address = address
        self.port = port
        # create a gRPC channel + stub
        channel = grpc.insecure_channel(self.address + ':' + str(self.port), compression=grpc.Compression.Gzip)
        self.conn = rpc.SnakeStub(channel)

        self._lock = threading.Lock()
        self.data_record = []         # data save
        self._decision_so_far = []  # history of the decision info from the server
        self._is_started = True     # 控制背景心跳
        self._new_response = []     # response list from the server
        self._new_request = []      # request list waiting to send to the server

        self.init_score = 0
        self.target = None

        self.logger = logger
        self.step = -1

        if self.logger is None:
            self.logger = simple_logger()

        self.stoped = False
        self.round = 0

        self.cond = threading.Condition()
        self.heart_beat_interval = 0.1

        self.logger.info('self.key is inited to ' + self.key)
        self.login(self.username, self.key)  # 这里是阻塞的，不登录无法进行游戏

        self._updater = threading.Thread(target=self.run)  # 维持heartbeat

        self._updater.setDaemon(True)
        self._updater.start()

    def __del__(self):
        self._is_started = False

    def login(self, user_id, user_pin):
        '''
        登录模块
        '''
        while True:
            # try:
                request = dealer_pb2.LoginRequest()
                request.user_id = user_id
                request.user_pin = user_pin
                self.logger.info('waiting for connect')
                response = self.conn.login(request)

                if response:
                    if response.success:
                        self.init_score = response.init_score
                        self.logger.info('login success, init score:%d' % self.init_score)
                        return
                    else:
                        self.logger.info('login failed.' + response.reason)
                        time.sleep(3)

            # except grpc.RpcError as error:
            #     print(error)
            #     self.logger.info('login failed. will retry one second later')
            #     time.sleep(1)

    def client_reset(self, u: str, logger):
        self.username = u
        # create a gRPC channel + stub
        channel = grpc.insecure_channel(self.address + ':' + str(self.port), compression=grpc.Compression.Gzip)
        self.conn = rpc.SnakeStub(channel)

        self._decision_so_far = []  # history of the decision info from the server
        self._new_response = []     # response list from the server
        self._new_request = []      # request list waiting to send to the server

        self.init_score = 0

        self.logger = logger
        self.step = -1
        if self.logger is None:
            self.logger = simple_logger()

        self.stoped = False
        self.round = 0

    def chat_with_server(self):
        '''
        通信相关
        '''
        while True:
            self.cond.acquire()
            while True:
                while len(self._new_request) != 0:
                    # yield a resquest from the request list to the server
                    msg = self._new_request.pop(0)
                    yield msg
                self.cond.wait()
            self.cond.release()

    def add_request(self, msg):
        self.cond.acquire()
        self._new_request.append(msg)
        self.cond.notify()
        self.cond.release()

    def run(self):
        """
        维持心跳，定期监听需要获得的消息
        """
        while self._is_started:
            # heartbeat
            msg = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                           msg_type=dealer_pb2.ActionRequest.HeartBeat)
            self.add_request(msg)

            time.sleep(self.heart_beat_interval)
            if self.stoped:
                self.client_reset(self.username, self.logger)
        return

    def start(self):
        '''
        处理从server发回的消息
        '''
        responses = self.conn.GameStream(self.chat_with_server())
        for res in responses:
            self._new_response.append(res)

            if res.msg_type == dealer_pb2.ActionResponse.GameDecision:
                # server asking for a decision from the client
                self.logger.info('request decision')

                round_output = json.loads(res.game_info)
                table = round_output['tableinfo']['table_num']

                if not check_intime(round_output):
                    print(' time delay happend!! ')
                    continue

                self.logger.info(f'桌子号：{table}; 玩家位置: {res.user_pos}; 总局数：{res.tp_number}; 总分： {res.total_score}; 参与局数： {res.game_number}; 局内步数：{res.table_round}')
                print('this is response: last act: ', round_output["gameinfo"]["Player"][res.user_pos]['LastAct'], ' act: ', round_output["gameinfo"]["Player"][res.user_pos]['Act'])
                start = time.time()
                ActTemp, self.target = MyStrategy(res.user_pos, round_output, self.target)
                end = time.time()
                self.data_record.append([res.user_pos, round_output, ActTemp])

                request = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                                   msg_type=dealer_pb2.ActionRequest.GameDecision,
                                                   game_info=ActTemp)

                self.add_request(request)

                print(' strategy time: ', end - start)
                print(' this is strategy act: ', ActTemp)

            elif res.msg_type == dealer_pb2.ActionResponse.RoundEnd:
                self.logger.info(f'{res.tp_number} 局游戏结束')
                self.logger.info(f'玩家位置: {res.user_pos}; 总局数：{res.tp_number}; 总分： {res.total_score}; 参与局数： {res.game_number}; 局内步数：{res.table_round}')
                pd.to_pickle(self.data_record,f"/data1/qzb/client_new/round_result/round_{res.tp_number}_result.pkl")

            elif res.msg_type == dealer_pb2.ActionResponse.StateUpdate:
                round_output = json.loads(res.game_info)
                self.step += 1
                self.logger.info('server time: %s' % round_output['info_time'])
                self.logger.info('killCount is: %d' % round_output['gameinfo']['Player'][res.user_pos]['Kill'])
                self.logger.info('starCount is: %d' % round_output['gameinfo']['Player'][res.user_pos]['SaveLength'])

            elif res.msg_type == dealer_pb2.ActionResponse.StateReady:  # 询问是否准备好
                self.logger.info('request ready')
                request = dealer_pb2.ActionRequest(user_id=self.username, user_pin=self.key,
                                                   msg_type=dealer_pb2.ActionRequest.StateReady)

                self.add_request(request)
                self.data_record = []

            elif res.msg_type == dealer_pb2.ActionResponse.GameEnd:
                self.logger.info('game end')
                self._is_started = False
                self.stoped = True
                return


#**********************************NOTICE**************************************
# You should make sure that your username and key is right,
# You should never keep more than one connection to the server at the same time.
#******************************************************************************

if __name__ == '__main__':

    # Ubiq108  zTmYI3kCvqZIXwsI
    # Test050  S4rCJ4TXcGVpQNaH
    # Test043  40FJIqk2cua2nKtm
    # Test021  jrdrUtS4xnTUkMaf
    
    # 139.196.39.76  7777 
    # 139.224.244.36 9001 模拟
    parser = argparse.ArgumentParser(description="Args")
    parser.add_argument('--username', type=str,default='Test050')
    parser.add_argument('--key', type=str,default='S4rCJ4TXcGVpQNaH')
    parser.add_argument('--address', type=str, default='139.196.39.76')  
    parser.add_argument('--port', type=int, default=7777)
    args = parser.parse_args()

    # logger = simple_logger()
    month = str(datetime.now().month).rjust(2,'0')
    day = str(datetime.now().day).rjust(2,'0')
    hour = datetime.now().hour
    minute = datetime.now().day
    logger = file_logger(f'/data1/qzb/client_new/clientlog/test_{month}_{day}_{hour}_{minute}.txt')

    c = Client(args.username, args.key, logger, args.address, args.port)
    c.start()
    exit()

