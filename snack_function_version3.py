import numpy as np
import pandas as pd
import random
import itertools
import datetime

# 地图展示
def show_map_sign(Position, wall):
    map = np.array([0]*55*40).reshape(55,40)
    for i in Position:
        map[i[0],i[1]]=1
        if i[0]==Position[0][0] and i[1]==Position[0][1]:
            map[i[0],i[1]] = 9
    for i in wall:
        map[i[0],i[1]]=2

    for x in range(map.shape[0]):
        print()
        for y in range(map.shape[1]):
            print(map[x,y], end=' ')

# 检测server信息的时间与收到时间是否一致，否则跳过此条信息
def convert_time(time_string):
    time_list = time_string.split(':')
    hour = float(time_list[0])
    minute = float(time_list[1])
    second = float(time_list[2])
    time_sec = hour * 3600 + minute * 60 + second
    return time_sec

def check_intime(GameInfo_):
    server_time_string = GameInfo_["info_time"].split(' ')[1]
    server_time = convert_time(server_time_string)
    my_time_string = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d-%H:%M:%S').split('-')[1]
    my_time = convert_time(my_time_string)
    print('server_time_string:', server_time_string)
    print('my_time_string:', my_time_string)
    if (my_time - server_time) > 3:
        return False
    else:
        if (my_time - server_time) > 2:
            print('warning: hign delay')
        return True

# 头被堵死的情况
# 力量道具剩余2
# 外边存在力量道具大于2的蛇
# 这次击杀会造成我的排名或积分下降 no
def kill_myself_check(Num_, GameInfo_):
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    max_strong = pd.DataFrame(live_sit_dic).T['strong'].max()
    if my_strong <= 2 and max_strong > 2:
        return True
    else:
        return False

# 自杀策略2
def kill_myself2(Num_, GameInfo_):
    ActList = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    WallPosition, _ = wall_position(GameInfo_, scale=-1)
    PositionNow = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    if my_speed > 7:
        my_speed = 7
        
    for i in ActList:
        PositionMove = list(np.sum([PositionNow, ActList[i]], axis=0))
        #检查是否撞
        if PositionMove in WallPosition:
            return i
        else:
            continue
        
    snack_target = snake_target(GameInfo_, PositionNow, my_speed)
    # 所有目标墙
    target_pos = [i for i in snack_target if i in WallPosition]
    if len(target_pos)==0:
        return ''

    path_cost_list = []
    action_string_list = []
    for pos in target_pos:
        action_string, path_cost = leap_of_faith(Num_, GameInfo_, pos)
        path_cost_list.append(path_cost)
        action_string_list.append(action_string)
    
    index = np.argmin(path_cost_list)
    action_string = action_string_list[index]
    return action_string
    
# 5步之内有蛇 - 蛇的距离
def danger_snack_distance(Num_,GameInfo_):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    strong_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] > 1) and (Num_ != i)]
    if len(strong_list) == 0:
        return {}

    d = {}
    cost_dict={0:1, 2:1, 3:1, 4:1, 5:1}
    for snack_id in strong_list:
        w_map = map_generate(GameInfo_)
        head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
        w_map[head[0],head[1]] = 0
        action_string, _, _ = a_star_search(w_map, my_head, head, cost_dict)
        d[snack_id] = len(action_string)
    
    return d

# 最佳小心速度
def my_careful_speedlimit(Num_,GameInfo_):
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    distance = danger_snack_distance(Num_,GameInfo_)
    danger_increase = False
    if len(distance)!=0 and pd.DataFrame([distance]).min().min() <= 5:
        danger_increase = True

    speed_dict = {0:my_length-1, 1:15, 2:12, 3:10, 4:8, 5:5, 6:3, 7:2, 8:1}

    if my_strong <= 2:
        speedlimit = 10000
    else:
        if my_length + my_save >= 120:
            if danger_increase:
                speedlimit = speed_dict[5]
            else:
                speedlimit = speed_dict[3]

        elif my_length + my_save >= 80:
            if danger_increase:
                speedlimit = speed_dict[5]
            else:
                speedlimit = speed_dict[2]

        elif my_length + my_save >= 40:
            if danger_increase:
                speedlimit = speed_dict[5]
            else:
                speedlimit = speed_dict[4]
        
        elif my_length + my_save >= 20:
            if danger_increase:
                speedlimit = speed_dict[6]
            else:
                speedlimit = speed_dict[4]
        elif my_length + my_save > 10:
            if danger_increase:
                speedlimit = speed_dict[7]
            else:
                speedlimit = speed_dict[3]
        else:
            if danger_increase:
                speedlimit = speed_dict[8]
            else:
                speedlimit = speed_dict[0]
    return speedlimit

# 危险块：墙壁 返回值 墙壁position：二维列表， 当前测量出的缩圈次数
# (to do)如果在墙刷新的时候，正对墙冲进去必死
# 缩圈次数最高19
def wall_position(GameInfo_, scale=-1):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    WallPosition  = GameInfo_["gameinfo"]["Map"]["WallPosition"]
    signal = 1
    while signal:
        scale = scale + 1
        if scale >= 20:
            break
        # 计算scale次缩圈后的最内层墙环
        x_boundry = (scale, Length-1-scale)
        y_boundry = (scale, Width-1-scale)
        WallPosition_round = []
        for x in range(x_boundry[0],x_boundry[1] + 1):
            for y in range(y_boundry[0],y_boundry[1] + 1):
                if x==x_boundry[0] or x==x_boundry[1] or y==y_boundry[0] or y==y_boundry[1]:
                    WallPosition_round.append([x,y])
        for i in WallPosition_round:
            signal = 1
            if i in WallPosition:
                break
            signal = 0
    scale = scale - 1
    return  WallPosition, scale

# 所有预期蛇身的位置（下一轮），这些一定不能碰
# (todo) ‘反复横跳的蛇’的尾部不是危险块，其他蛇的尾部有可能前进但没法判断(ok)
# 使用留存长度判断
def snack_body_position(GameInfo_):
    whole_SnakePosition = []
    for i_snake in range(len(GameInfo_["gameinfo"]["Player"])):
        if GameInfo_["gameinfo"]["Player"][i_snake]["IsDead"]:
            continue
        if(len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake]) == 0):
            continue
        SnakePosition  = GameInfo_["gameinfo"]["Map"]["SnakePosition"][i_snake]
        # 没有留存长度，尾部会往前一格
        if GameInfo_["gameinfo"]["Player"][i_snake]['SaveLength'] == 0:
            SnakePosition = SnakePosition[:-1]
        whole_SnakePosition = whole_SnakePosition + SnakePosition
    whole_SnakePosition = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in whole_SnakePosition])]
    return whole_SnakePosition

# 地图生成标注
def map_generate(GameInfo_):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    # suger坐标
    suger_pos = GameInfo_['gameinfo']['Map']['SugarPosition']
    # 速度道具坐标
    speed_pos = GameInfo_['gameinfo']['Map']['PropPosition'][0]
    # 力量道具坐标
    strong_pos = GameInfo_['gameinfo']['Map']['PropPosition'][1]
    # 双倍道具坐标
    double_pos = GameInfo_['gameinfo']['Map']['PropPosition'][2]
    # 危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    level1 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level1])]
    # 地图
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    for x in range(Length):
        for y in range(Width):
            if [x,y] in level1:
                w_map[x][y] = 1
            if [x,y] in suger_pos:
                w_map[x][y] = 2
            if [x,y] in speed_pos:
                w_map[x][y] = 3
            if [x,y] in strong_pos:
                w_map[x][y] = 4
            if [x,y] in double_pos:
                w_map[x][y] = 5
    return w_map 

# 谨慎地图生成标注
# 危险块包括level1 + level2
def careful_map_generate(Num_,GameInfo_):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    # suger坐标
    suger_pos = GameInfo_['gameinfo']['Map']['SugarPosition']
    # 速度道具坐标
    speed_pos = GameInfo_['gameinfo']['Map']['PropPosition'][0]
    # 力量道具坐标
    strong_pos = GameInfo_['gameinfo']['Map']['PropPosition'][1]
    # 双倍道具坐标
    double_pos = GameInfo_['gameinfo']['Map']['PropPosition'][2]
    # 危险块 
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    level1 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level1])]
    level2 = GameInfo_['level2'] #danger_snack_field(Num_, GameInfo_)
    danger = level1 + level2
    danger = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in danger])]
    # 地图
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    for x in range(Length):
        for y in range(Width):
            if [x,y] in danger:
                w_map[x][y] = 1
            if [x,y] in suger_pos:
                w_map[x][y] = 2
            if [x,y] in speed_pos:
                w_map[x][y] = 3
            if [x,y] in strong_pos:
                w_map[x][y] = 4
            if [x,y] in double_pos:
                w_map[x][y] = 5
    return w_map 

# 搜索道具时的最优速度
def optimal_max_speed(Num_, GameInfo_, action_string, useful_list=[2,3,4,5]):
    action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    w_map = map_generate(GameInfo_)
    speed = 0
    for action in action_string:
        speed = speed + 1
        # 头部移动
        my_head = list(np.sum([my_head, action_map[action]], axis=0))
        if w_map[my_head[0],my_head[1]] in useful_list:
            break
    speed_long = speed + my_length + my_save - 1
    return speed, speed_long

def snake_target(GameInfo_, head, speed):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    danger = []
    for row in range(2*speed+1):
        danger.append([head[0]-speed+row,head[1]])
        if row <= speed:
            bound = row
        else:
            bound = speed-(row - speed)
        for num in range(1,bound+1):
            danger.append([head[0]-speed+row,head[1]-num])
            danger.append([head[0]-speed+row,head[1]+num])

    danger = [i for i in danger if i[0]>=0 and i[0]<Length and i[1]>=0 and i[1]<Width]
    return danger

def danger_snack_field(Num_, GameInfo_):
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    strong_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] > 1) and (Num_ != i)]
    if my_strong <= 1:
        longer_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['length'] >= my_length ) and (Num_ != i)]
    else:
        longer_list = []
        
    killer_list = list(set(strong_list + longer_list))
    if len(killer_list) == 0:
        return []
    
    level2 = []
    for snack_id in killer_list:
        head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
        save = GameInfo_["gameinfo"]["Player"][snack_id]['SaveLength']
        length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id])
        speed = GameInfo_["gameinfo"]["Player"][snack_id]['Speed']
        
        if my_strong > 1 and speed > length + save:
            speed = length + save

        snack_target = snake_target(GameInfo_, head, speed)
        snack_head_in_pool = head_in_pool(snack_id, GameInfo_)
        pos_can_get = []
        for i in snack_head_in_pool:
            pos_can_get = pos_can_get + snack_head_in_pool[i]

        snack_target = [tar for tar in snack_target if tar in pos_can_get]
        level2 = level2 + snack_target

    level2 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level2])]
    return level2

# # 二级危险块定义，开启careful模式时, 不会吃这些位置的道具, 这个只是防止道具撞车发生
# def danger_snack_field(Num_, GameInfo_):
#     my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
#     my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
#     dead_list, sit_dic = situation(GameInfo_)
#     live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
#     strong_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] > 1) and (Num_ != i)]
#     if my_strong <= 1:
#         longer_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['length'] >= my_length ) and (Num_ != i)]
#     else:
#         longer_list = []
        
#     killer_list = list(set(strong_list + longer_list))
#     if len(killer_list) == 0:
#         return []
    
#     level2 = []
#     for snack_id in killer_list:
#         head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
#         speed = GameInfo_["gameinfo"]["Player"][snack_id]['Speed']
#         level2 = level2 + snake_target(GameInfo_, head, speed)
        
#     level2 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level2])]
#     return level2

# 危险蛇似乎应该满足最低长度，否则不危险
def killer_snack_field(Num_, GameInfo_):
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    killer_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] > 1) and (live_sit_dic[i]['length'] >= 5 ) and (Num_ != i)]
    if len(killer_list) == 0:
        return []
    
    level2 = []
    for snack_id in killer_list:
        head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
        speed = GameInfo_["gameinfo"]["Player"][snack_id]['Speed']
        level2 = level2 + snake_target(GameInfo_, head, speed)
        
    level2 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level2])]
    return level2

# function to search the path
def a_star_search(grid, begin_point, target_point, cost):
    action_string = ''
    if grid[target_point[0]][target_point[1]] == 1:
        # print('target dead block')
        return "", 10000, 10000
    
    # the cost map which pushes the path closer to the goal
    heuristic = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            heuristic[i][j] = abs(i - target_point[0]) + abs(j - target_point[1])
            if grid[i][j] == 1:
                heuristic[i][j] = 10000  # added extra penalty in the heuristic map
    # the actions we can take
    delta = [[-1, 0],  # a
             [0, -1],  # s
             [1, 0],  # d
             [0, 1]]  # w
    action_map = {(-1, 0):'a',(0, -1):'s',(1, 0):'d',(0, 1):'w'}

    close_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the referrence grid
    close_matrix[begin_point[0]][begin_point[1]] = 1
    action_matrix = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]  # the action grid

    x = begin_point[0]
    y = begin_point[1]
    g = 0
    f = g + heuristic[x][y]
    cell = [[f, g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand
    
    while not found and not resign:
        if len(cell) == 0:
            resign = True
            return "", 10000, 10000
        else:
            cell.sort()  # to choose the least costliest action so as to move closer to the goal
            cell.reverse()
            next = cell.pop()
            x = next[2]
            y = next[3]
            g = next[1]
            f = next[0]

            if x == target_point[0] and y == target_point[1]:
                found = True
            else:
                # delta have four steps
                for i in range(len(delta)):  # to try out different valid actions
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):  # 判断可否通过那个点
                        if close_matrix[x2][y2] !=1 and grid[x2][y2] != 1:
                            g2 = g + cost[grid[x2][y2]]
                            f2 = g2 + heuristic[x2][y2]
                            cell.append([f2, g2, x2, y2])
                            close_matrix[x2][y2] = 1
                            action_matrix[x2][y2] = i
    # invpath = []
    x = target_point[0]
    y = target_point[1]
    # invpath.append([x, y])  # we get the reverse path from here
    action_string = ""

    path_cost = 0
    path_cost_base = 0
    while x != begin_point[0] or y != begin_point[1]:
        path_cost = path_cost + cost[grid[x][y]]
        path_cost_base = path_cost_base + 1
        action_string = action_map[(delta[action_matrix[x][y]][0],delta[action_matrix[x][y]][1])] + action_string
        x2 = x - delta[action_matrix[x][y]][0]
        y2 = y - delta[action_matrix[x][y]][1]
        x = x2
        y = y2
        # invpath.append([x, y])
    return action_string, path_cost, path_cost_base

# 目标策略-随机选定指定的一类目标中的一个（method:1:随机选取一个;2:尽可能选择自定义cost最低的;3:选cost最低的，速度剩余继续选目标）
def target_strategy(Num_, GameInfo_, max_pool, target=2, cost_dict={0:1, 2:-1, 3:-1, 4:-1, 5:-1}, last_target=None, method=1, useful_list=[2,3,4,5], snack_id=None):
    speed =  GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    careful_speed_limit = my_careful_speedlimit(Num_,GameInfo_)

    # 0为自由通行节点，1为障碍,2为糖，3为速度道具，4为力量道具，5为双倍道具
    if cost_dict==None:
        cost_dict = {0:1, 2:1, 3:1, 4:1, 5:1}

    begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    _, scale = wall_position(GameInfo_, scale=-1)
    if scale < 19:
        target_list = target_set_advance(GameInfo_, choice=target, overall=False)
    else:
        target_list = target_set_advance(GameInfo_, choice=target, overall=True)
    
    target_list = [tar for tar in target_list if tar in max_pool]
    
    if len(target_list)==0:
        return '', 0, last_target

    
    # 靠近目标
    if snack_id != None:
        snack_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
        snack_nearby = snake_target(GameInfo_, snack_head, 5)
        new_target_list = [tar for tar in target_list if tar in snack_nearby]
        if len(new_target_list) != 0:
            target_list = new_target_list.copy()

    w_map1 = careful_map_generate(Num_, GameInfo_)
    w_map2 = map_generate(GameInfo_)
    count = 0
    for w_map in [w_map1, w_map2]:
        if count == 1 and careful_speed_limit < speed:
            speed = careful_speed_limit
        count = count + 1
        action_string = ''
        if last_target in target_list:
            action_string, _, _ = a_star_search(w_map, begin, last_target, cost_dict)

        if len(action_string) != 0:
            max_speed, max_speed_long = optimal_max_speed(Num_, GameInfo_, action_string, useful_list)
            if max_speed_long < speed:
                speed = max_speed_long
            break
            # return action_string[:speed], max_speed, last_target

        # method == 1
        if method==1:
            target_list_temp = target_list.copy()
            while(len(action_string)==0):
                if len(target_list_temp) == 0:
                    break
                this_target = target_list_temp.pop(random.randint(0,len(target_list_temp)-1))
                action_string, _, _ = a_star_search(w_map, begin, this_target, cost_dict)

            if len(target_list_temp) == 0:
                continue
            else:
                max_speed, max_speed_long = optimal_max_speed(Num_, GameInfo_, action_string, useful_list)
                if max_speed_long < speed:
                    speed = max_speed_long
                last_target = this_target.copy()
                break
                # return action_string[:speed], max_speed, last_target

        # method == 2 优先选择cost最低，推荐costdict与目标相符：
        # 推荐的cost_dict: 初期速度与力量均衡发展，{0:1, 2:-0.5, 3:-1.5, 4:-1.5, 5:-0.5}
        # 迫切的需要某一种时，{0:1, 2:0, 3:-0.5, 4:-1.5, 5:0}
        if method==2:
            target_list_temp = target_list.copy()
            path_cost_list = []
            action_string_list = []
            # 计算cost
            for tar in target_list_temp:
                action_string, path_cost, _ = a_star_search(w_map, begin, tar, cost_dict)
                path_cost_list.append(path_cost)
                action_string_list.append(action_string)

            # 优先选择cost最低
            index = np.argmin(path_cost_list)
            action_string = action_string_list[index]
            if len(action_string) != 0:
                last_target = target_list_temp[index]
                max_speed, max_speed_long = optimal_max_speed(Num_, GameInfo_, action_string, useful_list)
                if max_speed_long < speed:
                    speed = max_speed_long
                break
                # return action_string[:speed], max_speed, last_target
            else:
                continue
            
        # method == 3 优先选择cost最低，并持续搜索目标，推荐costdict与目标相符
        if method==3:
            target_list_temp = target_list.copy()
            action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
            action_string = ''
            my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
            my_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
            
            while len(action_string) < speed:
                if len(target_list_temp)==0:
                    break
                # 计算cost
                path_cost_list = []
                action_string_list = []
                for tar in target_list_temp:
                    action_string_s, path_cost, _ = a_star_search(w_map, begin, tar, cost_dict)
                    path_cost_list.append(path_cost)
                    action_string_list.append(action_string_s)

                # 优先选择cost最低
                index = np.argmin(path_cost_list)
                action_string_s = action_string_list[index]

                if len(action_string_s) != 0:
                    action_string = action_string + action_string_s
                    last_target = target_list_temp[index]
                    # 更新地图
                    for action in action_string_s:
                        # 头部移动
                        begin = list(np.sum([begin, action_map[action]], axis=0))
                        w_map[begin[0],begin[1]] = 1
                        if begin in target_list_temp:
                            target_list_temp.remove(begin)
                        my_body = [begin] + my_body
                        # 尾部移动
                        if my_save > 0:
                            my_save = my_save - 1
                        else:
                            tail = my_body.pop()
                            w_map[tail[0],tail[1]] = 0              
                else:
                    break

            if len(action_string) != 0:
                max_speed, max_speed_long = optimal_max_speed(Num_, GameInfo_, action_string, useful_list)
                if max_speed_long < speed:
                    speed = max_speed_long
                break
                # return action_string[:speed], max_speed, last_target
            else:
                continue

    if len(action_string) == 0:
        return '', 0, last_target
    else:
        return action_string[:speed], max_speed, last_target

# 带有备选选项的目标策略，移动速度不可超过自身长度(后路)
def target_strategy_wrap(Num_, GameInfo_, max_pool, target_type, backup_target, cost_dict = {0:1, 2:-1, 3:-1, 4:-1, 5:-1}, last_target=None, method=2, useful_list=[2,3,4,5], snack_id=None):
    # 0为自由通行节点,2为糖，3为速度道具，4为力量道具，5为双倍道具
    action, near_tool, last_target = target_strategy(Num_, GameInfo_, max_pool, target=target_type, cost_dict=cost_dict, last_target=last_target, method=method,useful_list=useful_list, snack_id=snack_id)
    if len(action) != 0:
        return action, near_tool, last_target

    for backup in backup_target:
        # 没有最优选项，所以不是很在乎现在的目标cost
        cost_dict = {0:1, 2:-1, 3:-1, 4:-1, 5:-1}
        action, near_tool, last_target = target_strategy(Num_, GameInfo_, max_pool, target=backup, cost_dict=cost_dict, last_target=last_target, method=method,useful_list=useful_list, snack_id=snack_id)
        if len(action) != 0:
            return action, near_tool, last_target
    return '', 0, last_target

# 掉头
def turnover(Num_, GameInfo_):
    snack_position = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
    antidirection = {(0,1):"w", (0,-1):"s", (-1,0):"a", (1,0):"d"}
    action_string = ''
    for i in range(len(snack_position)-1):
        first = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][i]
        second = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][i+1]
        action_string = action_string + antidirection[tuple(np.array(second)-np.array(first))]
    return action_string

# 两种防御策略可考虑以别的目标为终点
# 闪现寻找力量道具
def flash_for_strong(Num_, GameInfo_):
    cost_dict = {0:1, 2:1, 3:1, 4:1, 5:1}
    action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']

    _, scale = wall_position(GameInfo_, scale=-1)
    if scale < 19:
        target_list = target_set_advance(GameInfo_, choice=4, overall=False)
    else:
        target_list = target_set_advance(GameInfo_, choice=4, overall=True)
        
    w_map = map_generate(GameInfo_)

    level2 = killer_snack_field(Num_, GameInfo_)
    target_list = [tar for tar in target_list if tar not in level2]

    if len(target_list) == 0:
        return ''

    path_cost_list = []
    action_string_list = []
    # 计算cost
    for tar in target_list:
        action_string, path_cost, _ = a_star_search(w_map, begin, tar, cost_dict)
        path_cost_list.append(path_cost)
        action_string_list.append(action_string)

    # 优先选择cost最低, 寻找二级危险块外的目标飞跃
    while len(action_string_list) != 0:
        index = np.argmin(path_cost_list)
        action_string = action_string_list[index]

        if len(action_string) == 0:
            return ''

        head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
        my_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
        my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']

        for action in action_string[:my_speed]:
            # 头部移动
            head = list(np.sum([head, action_map[action]], axis=0))
            my_body = [head] + my_body
            # 尾部移动
            if my_save > 0:
                my_save = my_save - 1
            else:
                my_body.pop()

        # 是否逃离二级危险
        safe = True
        for i in my_body:
            if i in level2:
                safe = False
                break

        if safe:
            return action_string[:my_speed]
        else:
            path_cost_list.pop(index)
            action_string_list.pop(index)     
    return ''

# 回身吃力量道具，不能处理长度为1的情况！
def turn_over_for_strong(Num_, GameInfo_, pos_list): 
    cost_dict = {0:1, 2:1, 3:1, 4:1, 5:1}
    tail_begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][-1]
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    action_string1 = turnover(Num_, GameInfo_)

    w_map = map_generate(GameInfo_)
    for pos in pos_list:
        w_map[pos[0],pos[1]] = 1

    _, scale = wall_position(GameInfo_, scale=-1)
    if scale < 19:
        target_list = target_set_advance(GameInfo_, choice=4, overall=False)
    else:
        target_list = target_set_advance(GameInfo_, choice=4, overall=True)

    if len(target_list) == 0:
        if len(action_string1) >= (my_length + my_save - 1):
            return action_string1
        else:
            return ''

    path_cost_list = []
    action_string_list = []
    # 计算cost
    for tar in target_list:
        action_string, path_cost, _ = a_star_search(w_map, tail_begin, tar, cost_dict)
        path_cost_list.append(path_cost)
        action_string_list.append(action_string)
    
    while len(action_string_list) != 0:
        index = np.argmin(path_cost_list)
        action_string = action_string_list[index]

        if (len(action_string1) + len(action_string)) >= (my_length + my_save - 1):

            return (action_string1 + action_string)[:my_speed]

        if len(action_string) == 0:
            return ''
        else:
            path_cost_list.pop(index)
            action_string_list.pop(index) 
    return ''

# 保护机制，当力量道具为0，并完全暴露在二级危险块时启动
# 如果有（体长+留存+1）的速度，可闪现
# 如果有（体长+留存-1）的速度，可调头
# 如果没有可操作的速度，可自尽
def defense_mechanism(Num_, GameInfo_):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
    my_length = len(my_body)
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']

    # 检查安全性，头部三点
    pos1 = list(np.sum([my_head, [0,1]], axis=0))
    pos2 = list(np.sum([my_head, [0,-1]], axis=0))
    pos3 = list(np.sum([my_head, [1,0]], axis=0))
    pos4 = list(np.sum([my_head, [-1,0]], axis=0))
    pos_list = [pos1, pos2, pos3, pos4]
    # 蛇头附近的安全块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]
    pos_list = [i for i in pos_list if i not in my_body]

    level2 = killer_snack_field(Num_, GameInfo_)
    # 三点,存在不在危险块中时,则安全
    safe_pos_list = [i for i in pos_list if i not in level2]

    # 三点全部在危险块中时,两种逃生策略
    action_string = ''
    if len(safe_pos_list) == 0:
        # 闪现逃跑找力量道具
        if my_speed >= (my_length + my_save + 1):
            # print('defense: flash')
            action_string = flash_for_strong(Num_, GameInfo_)

        # 回身跑
        if len(action_string) == 0 and my_length > 1 and my_speed >= (my_length + my_save - 1):
            # print('(flash cancel) defense: turnover')
            action_string = turn_over_for_strong(Num_, GameInfo_, pos_list)
            antidirection = {"s":"w", "w":"s", "d":"a", "a":"d"}
            anti_action_string = ""
            for act in action_string:
                anti_action_string = antidirection[act] + anti_action_string

            # 如果这招上次用过了，就不用了
            if my_length >= 10 and GameInfo_["gameinfo"]["Player"][Num_]['Act'] == anti_action_string:
                # print(" same techniq! ")
                action_string = ""

        # 是否自尽？(to do)
    return action_string

def AI0(Num_, GameInfo_):
    pool = GameInfo_['pool']
    antidirection = {(0,1):"w", (0,-1):"s", (-1,0):"a", (1,0):"d"}
    #自身头部位置
    PositionNow = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    ActList = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}

    # 危险块
    wall_pos, scale = wall_position(GameInfo_, scale=-1)
    level1 = wall_pos + snack_body_position(GameInfo_)
    level1 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level1])]

    for i in ActList:
        PositionMove = list(np.sum([PositionNow, ActList[i]], axis=0))
        #检查是否撞
        if PositionMove in level1:
            continue
        else:
            return i
    
    second = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][1]
    tail = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][-1]

    tail_check = False
    for i in ActList:
        PositionMove = list(np.sum([tail, ActList[i]], axis=0))
        #检查尾部是否安全
        if PositionMove in level1:
            continue
        else:
            tail_check = True

    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length =  GameInfo_["gameinfo"]["Player"][Num_]['Score_len']
    my_save_length = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    left_delete = my_save_length + my_length - (my_speed + 1)
    # 需要抓紧出来
    action_string = turnover(Num_, GameInfo_)
    # 非安全模式下，转向
    if tail_check and scale <= 12 and turnover_or_not(Num_, GameInfo_, pool):
        # print('turn over')
        if left_delete <= 0:
            if my_save_length != 0:
                return action_string[:my_save_length]
            else:
                return action_string
        else:
            if left_delete > my_speed:
                return action_string[:my_speed]
            else:
                return action_string[:left_delete]
    elif tail_check and my_save_length == 0 and my_speed >= (my_length - 1):
        # print('turn over')
        return action_string
    else:
        if kill_myself_check(Num_, GameInfo_):
            action_string = kill_myself2(Num_, GameInfo_)
            if len(action_string) != 0:
                # print(' kill myself now ')
                return action_string
    
        return antidirection[tuple(np.array(second)-np.array(PositionNow))]

# 多目标 A star
def multi_a_star_search(map, begin_point, target_point_list, cost):
    grid = map.copy()
    action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    final_action = ''
    final_path_cost = 0
    final_path_cost_base = 0
    for i in range(len(target_point_list)):
        action_string, path_cost, path_cost_base = a_star_search(grid, begin_point, target_point_list[i], cost)
        if len(action_string) == 0:
            return '',10000,10000
        else:
            grid[begin_point[0],begin_point[1]] = 1
            for j in action_string:
                begin_point = list(np.sum([begin_point, action_map[j]], axis=0))
                grid[begin_point[0],begin_point[1]] = 1
            final_action = final_action + action_string
            final_path_cost = final_path_cost + path_cost
            final_path_cost_base = final_path_cost_base + path_cost_base
    return final_action, final_path_cost, final_path_cost_base

# 返回存货蛇的编号，长度，速度，力量情况
def situation(GameInfo_):
    dead_list = []
    dic = {}
    for num in range(len(GameInfo_["gameinfo"]["Player"])):
        if(GameInfo_["gameinfo"]["Player"][num]["IsDead"]):
            dead_list.append(num)
        dic[num] = {}
        dic[num]['length'] = GameInfo_["gameinfo"]["Player"][num]['Score_len']
        dic[num]['speed'] = GameInfo_["gameinfo"]["Player"][num]['Speed']
        dic[num]['speedtime'] = GameInfo_["gameinfo"]["Player"][num]['Prop']['speed']
        dic[num]['strong'] = GameInfo_["gameinfo"]["Player"][num]['Prop']['strong']
    return dead_list, dic

# 击杀所需的地图
def kill_map_generate(GameInfo_, snack_id):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    # suger坐标
    suger_pos = GameInfo_['gameinfo']['Map']['SugarPosition']
    # 速度道具坐标
    speed_pos = GameInfo_['gameinfo']['Map']['PropPosition'][0]
    # 力量道具坐标
    strong_pos = GameInfo_['gameinfo']['Map']['PropPosition'][1]
    # 双倍道具坐标
    double_pos = GameInfo_['gameinfo']['Map']['PropPosition'][2]
    # 危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    targetSnakePosition = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id]
    level1 = set([(i[0],i[1]) for i in level1])-set([(i[0],i[1]) for i in targetSnakePosition])
    level1 = [[i[0],i[1]] for i in level1]
    # 地图
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    for x in range(Length):
        for y in range(Width):
            if [x,y] in level1:
                w_map[x][y] = 1
            if [x,y] in suger_pos:
                w_map[x][y] = 2
            if [x,y] in speed_pos:
                w_map[x][y] = 3
            if [x,y] in strong_pos:
                w_map[x][y] = 4
            if [x,y] in double_pos:
                w_map[x][y] = 5
    return w_map 

# 是否存在其他击杀者
def other_killer(Num_, GameInfo_, snack_id, try_list):
    # print('check other killer')
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    killer_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] > 1 ) and (live_sit_dic[i]['length'] > 7 ) and (Num_ != i)]
    if len(killer_list) == 0:
        return False
    
    kill_map = kill_map_generate(GameInfo_, snack_id)
    for killer in killer_list:
        killer_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][killer][0]
        killer_speed = GameInfo_["gameinfo"]["Player"][killer]['Speed']
        killer_length = GameInfo_["gameinfo"]["Player"][killer]['Score_len']
        killer_save = GameInfo_["gameinfo"]["Player"][killer]['SaveLength']
        
        all_action = ''
        path_cost_list = []
        all_action_list = []
        for pos_l in try_list:
            all_action, path_cost, _ = multi_a_star_search(kill_map, killer_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
            path_cost_list.append(path_cost)
            all_action_list.append(all_action)
            
        index = np.argmin(path_cost_list)
        all_action = all_action_list[index]

        if len(all_action) != 0 and killer_speed >= len(all_action) and len(all_action) <= (killer_length + killer_save - 1):
            return True

    return False

# 三点击杀存在失败性，因此无法忽略目标蛇的身体进行击杀, 体长至少5
def kill_snack_3pos_version(Num_, GameInfo_, snack_id):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = GameInfo_["gameinfo"]["Player"][Num_]['Score_len']
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的四个位置，需要删除已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 蛇头附近的安全块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    # 蛇在墙里或所有位置已被占领
    if len(pos_list) == 0:
        return False, '', 10000

    try_list = [list(i) for i in itertools.permutations(pos_list)]

    # 3pos 无法使用kill版本map
    kill_map = map_generate(GameInfo_)

    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(kill_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]
    path_cost = path_cost_list[index]

    if len(all_action)==0:
        return False, '', 10000
    else:
        if my_speed >= len(all_action):
            if len(all_action) <= (my_length + my_save - 1) or (not other_killer(Num_, GameInfo_, snack_id, try_list)):
                return True, all_action, path_cost
            else:
                # 击杀有风险
                # print(' kill is risk ')
                return False, '', 10000

        else:
            # 不能一击必杀，靠近并积攒力量+速度
            w_map = map_generate(GameInfo_)
            path_cost_list = []
            all_action_list = []
            for chase_pos in pos_list:
                all_action, _, path_cost1 = a_star_search(w_map, my_head, chase_pos, cost={0:1, 2:0, 3:-1, 4:-1, 5:0})
                path_cost_list.append(path_cost1)
                all_action_list.append(all_action)
                
            index = np.argmin(path_cost_list)
            all_action = all_action_list[index]

            if len(all_action) != 0:
                return False, all_action[:my_speed], path_cost
            else:
                return False, '', 10000  #没有力量道具的蛇下回合肯定死

def kill_snack_4pos_version(Num_, GameInfo_, snack_id):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = GameInfo_["gameinfo"]["Player"][Num_]['Score_len']
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的四个位置，需要删除已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 除去目标蛇的危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    targetSnakePosition = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id]
    level1 = set([(i[0],i[1]) for i in level1])-set([(i[0],i[1]) for i in targetSnakePosition])
    level1 = [[i[0],i[1]] for i in level1]
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    # 蛇在墙里或四个位置已被占领
    if len(pos_list) == 0:
        return False, '', 10000

    try_list = [list(i) for i in itertools.permutations(pos_list)]
    kill_map = kill_map_generate(GameInfo_, snack_id)

    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(kill_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]
    path_cost = path_cost_list[index]

 
    if len(all_action)==0:
        return False, '', 10000
    else:
        if my_speed >= len(all_action):
            if len(all_action) <= (my_length + my_save - 1) or (not other_killer(Num_, GameInfo_, snack_id, try_list)):
                return True, all_action, path_cost
            else:
                # 击杀有风险
                # print(' kill is risk ')
                return False, '', 10000

        else:
            # 不能一击必杀，靠近并积攒力量+速度
            w_map = map_generate(GameInfo_)
            path_cost_list = []
            all_action_list = []
            for chase_pos in pos_list:
                all_action, _, path_cost1 = a_star_search(w_map, my_head, chase_pos, cost={0:1, 2:0, 3:-1, 4:-1, 5:0})
                path_cost_list.append(path_cost1)
                all_action_list.append(all_action)

            index = np.argmin(path_cost_list)
            all_action = all_action_list[index]

            if len(all_action) != 0:
                return False, all_action[:my_speed], path_cost
            else:
                return False, '', 10000  #没有力量道具的蛇下回合肯定死

# 有自损的击杀
def damage_kill(Num_, GameInfo_, snack_id):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    my_length = GameInfo_["gameinfo"]["Player"][Num_]['Score_len']
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的四个位置，需要删除已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 危险块
    level1 = wall_position(GameInfo_, scale=-1)[0]
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    # 蛇在墙里
    if len(pos_list) == 0:
        return False, '', 10000

    try_list = [list(i) for i in itertools.permutations(pos_list)]
    
    # 损伤式攻击，眼里只有墙
    kill_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    for x in range(Length):
        for y in range(Width):
            if [x,y] in level1:
                kill_map[x][y] = 1
    
    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(kill_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]
    path_cost = path_cost_list[index]

    if len(all_action)==0:
        return False, '', 10000
    else:
        if my_speed >= len(all_action) and len(all_action) <= (my_length + my_save - 1):
                return True, all_action, path_cost
        else:
            # 不能一击必杀，靠近并积攒力量+速度
            w_map = map_generate(GameInfo_)
            path_cost_list = []
            all_action_list = []
            for chase_pos in pos_list:
                all_action, _, path_cost = a_star_search(w_map, my_head, chase_pos, cost={0:1, 2:0, 3:-1, 4:-1, 5:0})
                path_cost_list.append(path_cost)
                all_action_list.append(all_action)

            index = np.argmin(path_cost_list)
            all_action = all_action_list[index]
                
            if len(all_action) != 0:
                return False, all_action[:my_speed], path_cost
            else:
                return False, '', 10000 #没有力量道具的蛇下回合肯定死
            
# 当活着的人中存在力量小于10的蛇，一定启动力量掠夺
# 当自己的力量道具时长为第一，或者可以延续到结束时，第一的蛇长达到我的两倍或超100，或所有人力量道具超20，且自己的力量可再坚持50以上回合->增长长度
def strong_or_length(Num_, GameInfo_):
    strongest = False
    time_enough = False
    chance = False

    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i: sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])

    my_strong = live_sit_dic[Num_]['strong']
    my_length = live_sit_dic[Num_]['length']
    second_strong_val = list(pd.DataFrame(live_sit_dic).T.sort_values(by='strong',ascending=False)['strong'])[1]
    min_strong_val = (pd.DataFrame(live_sit_dic).T.sort_values(by='strong',ascending=False)['strong']).min()
    first_snack_id = pd.DataFrame(live_sit_dic).T.sort_values(by='length',ascending=False).index[0]
    first_length = list(pd.DataFrame(live_sit_dic).T.sort_values(by='length',ascending=False)['length'])[0]
    
    if my_strong > second_strong_val:
         strongest = True
    
    if (my_strong + rd_num) > 150:
        time_enough = True
    
    if ( (first_snack_id != Num_ and first_length > 60) or first_length > 2*my_length or min_strong_val > 20) and my_strong >= 40:
        chance = True

    if strongest or time_enough or chance:
        return True
    else:
        return False
        
# ===准备阶段===
# 状态 体长<10 或 速度<10 或 力量时间<10时 进入
# 常规准备状态策略
# 2为糖，3为速度道具，4为力量道具，5为双倍道具
# 人多时是否要远离对应的蛇，以此选择目标道具？（to do）
def prepare_Strategy(Num_, GameInfo_, myState, max_pool, target=None):
    mylength, myspeed, myspeedtime, mystrong =  myState
    # 防御机制首先启动
    if mystrong <= 1:
        # print(' defense? ')
        action = defense_mechanism(Num_, GameInfo_)
        if len(action) != 0:
            # print(' defense mechanism activate ')
            return action, target

    if myspeed < 4 or myspeedtime < 10:
        # print('for speed ')
        if myspeed <= 2:
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=3, backup_target=[4,5,2], cost_dict = {0:1, 2:0, 3:-0.5, 4:-0.5, 5:0}, last_target=target, method=3, useful_list=[2,3,4,5])
            return action, target
        elif mystrong > 1:
            # 速度高于体长体长优先
            if myspeed > mylength:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=2, backup_target=[3,4,5], cost_dict = {0:1, 2:-0.5, 3:-0.5, 4:-0.5, 5:-0.5}, last_target=target, method=3, useful_list=[2,3,4,5])
                return action, target
            else:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=3, backup_target=[4,5,2], cost_dict = {0:1, 2:-0.5, 3:-0.5, 4:-0.5, 5:-0.5}, last_target=target, method=3, useful_list=[2,3,4,5])
                return action, target
        else:
            #挑近的吃
            action1, near1, target1 = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=3, backup_target=[4,5,2], cost_dict = {0:1, 2:0, 3:-0.5, 4:-0.5, 5:0}, last_target=target, method=3, useful_list=[3,4])
            action2, near2, target2 = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[3,5,2], cost_dict = {0:1, 2:0, 3:-0.5, 4:-0.5, 5:0}, last_target=target, method=3, useful_list=[3,4])
            if near1 <= near2 and near1 != 0:
                action = action1
                target = target1
            else:
                action = action2
                target = target2
            
            return action, target
    if mystrong < 7:
        # print('for strong ')
        if mystrong <= 1:
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[3,5,2], cost_dict = {0:1, 2:0, 3:0, 4:-0.5, 5:0}, last_target=target, method=3, useful_list=[4])
            return action, target
        if myspeed >= 6:
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[3,5,2], cost_dict = {0:1, 2:-0.5, 3:-0.5, 4:-0.5, 5:-0.5}, last_target=target, method=3, useful_list=[2,3,4,5])
            return action, target
        else:
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[3,5,2], cost_dict = {0:1, 2:0, 3:-0.5, 4:-0.5, 5:0}, last_target=target, method=3, useful_list=[2,3,4,5])
            return action, target

    # 最近的可攻击蛇
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    chase_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] <= 3 ) and (Num_ != i)]
    if len(chase_list)==0:
        snack_id = None
    elif len(chase_list)==1:
        snack_id = chase_list[0]
    else:
        my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
        snack_id = chase_list[0]
        small_distance = 1000
        for i in chase_list:
            w_map = map_generate(GameInfo_)
            snack_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][i][0]
            w_map[snack_head[0],snack_head[1]] = 0
            _, _, path_cost = a_star_search(w_map, my_head, snack_head, cost={0:1, 2:1, 3:1, 4:1, 5:1})
            # distance = abs(snack_head[0]-my_head[0]) + abs(snack_head[1]-my_head[1])
            if path_cost < small_distance:
                small_distance = path_cost
                snack_id = i

    if mylength < 8:
        # print('for length ')
        if GameInfo_["gameinfo"]["Player"][Num_]['Prop']['double'] >= 1:
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=2, backup_target=[5,3,4], cost_dict = {0:1, 2:-0.5, 3:-0.5, 4:-0.5, 5:-0.5}, last_target=target, method=3, useful_list=[2, 5], snack_id=snack_id)
            return action, target
        else:    
            action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=5, backup_target=[2,3,4], cost_dict = {0:1, 2:-1, 3:-0.5, 4:-0.5, 5:-1.5}, last_target=target, method=3, useful_list=[2, 5], snack_id=snack_id)
            return action, target

    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])

    if myspeedtime >= 5:
        if strong_or_length(Num_, GameInfo_):
            # print(' prepare rob length tools ')
            if GameInfo_["gameinfo"]["Player"][Num_]['Prop']['double'] > 1:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=2, backup_target=[4,3,5], cost_dict = {0:1, 2:-1, 3:-1, 4:-1, 5:-1}, last_target=target, method=3, useful_list=[2, 3, 4, 5], snack_id=snack_id)
            else:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=5, backup_target=[4,3,2], cost_dict = {0:1, 2:-1, 3:-1, 4:-1, 5:-1}, last_target=target, method=3, useful_list=[2, 3, 4, 5], snack_id=snack_id)
        else:
            # print(' prepare rob strong tools ')
            if myspeed > 50 and (myspeedtime + rd_num) > 150:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[5,2,3], cost_dict = {0:1, 2:-1.5, 3:1, 4:-1, 5:-1}, last_target=target, method=3, useful_list=[2,4,5], snack_id=snack_id)
            else:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=4, backup_target=[3,5,2], cost_dict = {0:1, 2:-1.5, 3:-1, 4:-1, 5:-1}, last_target=target, method=3, useful_list=[2,3,4,5], snack_id=snack_id)
        return action, target
    else:
        # print(' prepare need prepare speed ')
        action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool,target_type=3, backup_target=[4,5,2], cost_dict = {0:1, 2:-1, 3:-1, 4:-1, 5:-1}, last_target=target, method=3, useful_list=[2,3,4,5], snack_id=snack_id)
        return action, target
    
# 检查分数是否升高
def score_raise_check(Num_, GameInfo_, kill_snack_id, loss_length):
    # 当前分数,排名
    score_frame = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
        d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']
        d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
        score_frame.append(d)
    score_frame = pd.DataFrame(score_frame).rank(method='average')
    score_frame = (score_frame['kill']*1.5 + score_frame['time'] + score_frame['length'])/3.5
    my_score = score_frame[Num_]
    my_rank = score_frame.rank(ascending=False,method='min')[Num_]

    # 杀蛇后的分数与排名
    score_frame_aft = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        if i==Num_:
            d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len'] - loss_length
            d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill'] + 2
        else:
            d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
            d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']

        if GameInfo_["gameinfo"]["Player"][i]['IsDead'] or i == kill_snack_id:
            d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
        else:
            d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time'] + 1
        score_frame_aft.append(d)
    score_frame_aft = pd.DataFrame(score_frame_aft).rank(method='average')
    score_frame_aft = (score_frame_aft['kill']*1.5 + score_frame_aft['time'] + score_frame_aft['length'])/3.5
    my_score_aft = score_frame_aft[Num_]
    my_rank_aft = score_frame_aft.rank(ascending=False,method='min')[Num_]

    if my_score_aft >= my_score or my_rank_aft <= my_rank:
        return True
    else:
        return False

# 在场内搜索没有力量道具的蛇/或长度小于1的蛇，进行击杀，优先攻击离得近的蛇
def search_target_snack(Num_, GameInfo_):
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    if my_strong > 1:
        not_strong_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] <= 1 ) and (Num_ != i)]
        very_short_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['length'] == 1 ) and (Num_ != i)]
        kill_list = list(set(not_strong_list + very_short_list))
    else:
        kill_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] <= 1 ) and (live_sit_dic[i]['length'] < live_sit_dic[Num_]['length'] ) and (Num_ != i)]
        
    if len(kill_list) == 0:
        return False, ''

    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    action_string1 = ''
    action_string2 = ''
    action_string3 = ''
    # 四维击杀
    if (my_length + my_save) >= 7:
        path_cost_list = []
        action_string_list = []
        kill_or_not_list = []
        action_string1 = ''
        for snack_id in kill_list:
            kill_or_not, action_string1, path_cost = kill_snack_4pos_version(Num_, GameInfo_, snack_id)

            path_cost_list.append(path_cost)
            action_string_list.append(action_string1)
            kill_or_not_list.append(kill_or_not)

        index = np.argmin(path_cost_list)
        action_string1 = action_string_list[index]
        kill_or_not = kill_or_not_list[index]

        if kill_or_not:
            # print(' kill four pos ! ')
            return True, action_string1
    # 三维击杀
    if (my_length + my_save) >= 5:
        path_cost_list = []
        action_string_list = []
        kill_or_not_list = []
        action_string2 = ''
        for snack_id in kill_list:
            kill_or_not, action_string2, path_cost = kill_snack_3pos_version(Num_, GameInfo_, snack_id)

            path_cost_list.append(path_cost)
            action_string_list.append(action_string2)
            kill_or_not_list.append(kill_or_not)

        index = np.argmin(path_cost_list)
        action_string2 = action_string_list[index]
        kill_or_not = kill_or_not_list[index]

        if kill_or_not:
            # print(' kill three pos ! ')
            return True, action_string2
    
    # 自伤击杀
    if my_strong > 1:
        kill_list = [i for i in live_sit_dic.keys() if (live_sit_dic[i]['strong'] <= 1 ) and (live_sit_dic[i]['length'] >= 10 ) and (Num_ != i)]
    else:
        kill_list = []

    if len(kill_list) != 0 and (my_length + my_save) >= 7:
        path_cost_list = []
        action_string_list = []
        kill_or_not_list = []
        action_string3 = ''
        for snack_id in kill_list:
            kill_or_not, action_string3, path_cost = damage_kill(Num_, GameInfo_, snack_id)

            path_cost_list.append(path_cost)
            action_string_list.append(action_string3)
            kill_or_not_list.append(kill_or_not)

        index = np.argmin(path_cost_list)
        action_string3 = action_string_list[index]
        kill_or_not = kill_or_not_list[index]

        if kill_or_not:
            if score_raise_check(Num_, GameInfo_, kill_list[index], len(action_string3)):
                # print(' kill damage self ! ')
                return True, action_string3
            else:
                action_string3 = ''

    # 靠近
    # print(' chase? ')
    if len(action_string1) != 0:
        return False, action_string1

    elif len(action_string2) != 0:
        return False, action_string2

    elif len(action_string3) != 0:
        return False, action_string3

    else:
        return False, ''

# 信仰之跃 无视一切冲向目标地点 用于自尽 / 最终一换一（to do）
def leap_of_faith(Num_, GameInfo_, target_pos):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']

    begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    action_string, path_cost, _ = a_star_search(w_map, begin, target_pos, cost={0:1, 2:1, 3:1, 4:1, 5:1})

    speed =  GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    if speed >= len(action_string):
        return action_string, path_cost
    else:
        return '', path_cost

# 无人之时，撞墙
def kill_myself(Num_, GameInfo_):
    WallPosition, scale = wall_position(GameInfo_, scale=-1)
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    x_boundry = (scale, Length-1-scale)
    y_boundry = (scale, Width-1-scale)
    WallPosition_round = []
    for x in range(x_boundry[0],x_boundry[1] + 1):
        for y in range(y_boundry[0],y_boundry[1] + 1):
            if x==x_boundry[0] or x==x_boundry[1] or y==y_boundry[0] or y==y_boundry[1]:
                WallPosition_round.append([x,y])
    # 所有目标墙
    target_pos = [i for i in WallPosition_round if i in WallPosition]
    path_cost_list = []
    for pos in target_pos:
        action_string, path_cost = leap_of_faith(Num_, GameInfo_, pos)
        if len(action_string)>0:
            return action_string
        else:
            path_cost_list.append(path_cost)
    
    index = np.argmin(path_cost_list)
    begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    # 危险块
    level1 = snack_body_position(GameInfo_)
    # 地图
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    for x in range(Length):
        for y in range(Width):
            if [x,y] in level1:
                w_map[x][y] = 1
    
    action_string, _, _ = a_star_search(w_map, begin, target_pos[index], cost={0:1, 2:1, 3:1, 4:1, 5:1})
    
    return action_string

# 返回 1.是否撞墙死亡 2.是否撞自身cost长度
def fatal_or_not(Num_, GameInfo_, all_action):
    action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    begin_point = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]

    wall_pos = wall_position(GameInfo_, scale=-1)[0]
    move_num = len(all_action) - GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    if move_num < 0:
        move_num = 0
    left_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][:-move_num]

    sig = False
    for j in all_action:
        begin_point = list(np.sum([begin_point, action_map[j]], axis=0))
        if begin_point in wall_pos:
            return True, None 
        if begin_point in left_body:
            sig = True
    return False, sig

# 只剩两条蛇，且另一条蛇更短，在没有力量道具的那一刻执行
def kill_snack_fatal(Num_, GameInfo_, snack_id):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的四个位置，需要删除已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    targetSnakePosition = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id]
    level1 = set([(i[0],i[1]) for i in level1])-set([(i[0],i[1]) for i in targetSnakePosition])
    level1 = [[i[0],i[1]] for i in level1]
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    # 蛇在墙里或四个位置已被占领
    if len(pos_list) == 0:
        return False, ''

    try_list = [list(i) for i in itertools.permutations(pos_list)]
    
    # 自杀式攻击，无视一切
    kill_map = np.array([0]*(Length*Width)).reshape(Length,Width)

    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(kill_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]

    if len(all_action)==0:
        return False, ''
    else:
        if my_speed >= len(all_action):
            dead, _ = fatal_or_not(Num_, GameInfo_, all_action)
            if score_raise_check3(Num_, GameInfo_, snack_id, dead, len(all_action)):
                return True, all_action
            else:
                return False, ''
        else:
            # 不能一击必杀，靠近并积攒力量+速度
            w_map = map_generate(GameInfo_)
            path_cost_list = []
            all_action_list = []
            for chase_pos in pos_list:
                action_string, _, path_cost = a_star_search(w_map, my_head, chase_pos, cost={0:1, 2:0, 3:-1, 4:-1, 5:0})
                path_cost_list.append(path_cost)
                all_action_list.append(action_string)    
            index = np.argmin(path_cost_list)
            action_string = all_action_list[index]
            if len(action_string) != 0:
                return False, action_string[:my_speed]
            else:
                return False, '' #没有力量道具的蛇下回合肯定死

def block_target_snack(Num_, GameInfo_, snack_id):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])

    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    enemy_last_move = len(GameInfo_["gameinfo"]["Player"][snack_id]['Act'])
    enemy_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id])
    
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的几个位置，需要删除所有已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 包括目标蛇的危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]
    
    # 蛇在墙里或四个位置已被占领
    if len(pos_list) == 0:
        return False, '', 10000

    try_list = [list(i) for i in itertools.permutations(pos_list)]
    w_map = map_generate(GameInfo_)
    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(w_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]
    path_cost = path_cost_list[index]

    if len(all_action)==0:
        return False, '', 10000
    else:
        # 如果满足1.上一轮蛇的移动速度高于体长的1/3 (2.我阻拦所需要的步数小于他上一轮的移动距离的2/3,取消) 3.我阻拦所需要的步数小于我体长的1/5
        a = 1/3
        b = 1/5
        if my_speed >= len(all_action) and enemy_last_move/enemy_length > a and len(all_action) < (my_length*b):
            return True, all_action, path_cost
        else:
            return False, '', 10000

# 如果不能抢，最好能向这目标移动，考虑自己的蛇长执行(to do)
def rob_target_snack(Num_, GameInfo_, snack_id, rob_tool=4):
    my_speed =  GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    if my_speed > my_length * 1/2:
        my_speed = int(my_length * 1/2)

    cost_dict = {0:1, 2:1, 3:1, 4:1, 5:1}
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    _, scale = wall_position(GameInfo_, scale=-1)
    if scale < 19:
        target_list = target_set_advance(GameInfo_, choice=rob_tool, overall=False)
    else:
        target_list = target_set_advance(GameInfo_, choice=rob_tool, overall=True)
    
    my_pool = []
    for pool in GameInfo_['my_pool']:
        my_pool = my_pool + GameInfo_['my_pool'][pool]

    target_list = [tar for tar in target_list if tar in my_pool]

    w_map = map_generate(GameInfo_)
    if len(target_list) != 0:
        enemy_path_cost_list = []
        action_string_list = []
        # 计算cost
        for tar in target_list:
            my_action_string, my_path_cost, _ = a_star_search(w_map, my_head, tar, cost_dict)
            enemy_action_string, enemy_path_cost, _ = a_star_search(w_map, enemy_head, tar, cost_dict)
            if my_path_cost < enemy_path_cost:
                enemy_path_cost_list.append(enemy_path_cost)
                action_string_list.append(my_action_string)

        # 优先选择离敌人最近的并且离自己更近的
        if len(enemy_path_cost_list) != 0:
            index = np.argmin(enemy_path_cost_list)
            action_string = action_string_list[index]
            return action_string[:my_speed]
        else:
            return ''
    
    return ''

# 我的力量满格，他的力量大于40或满格 长度优先
# 我的力量未满格，他的力量大于40，并且我的力量大于他的力量时 长度优先
def strong_or_length_one_enemy(Num_, GameInfo_):
    strong_full = False
    strong_high = False
    
    dead_list, sit_dic = situation(GameInfo_)
    live_enemy_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list and Num_ != i}
    enemy_num = list(live_enemy_dic.keys())[0]
    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    enemy_strong = GameInfo_["gameinfo"]["Player"][enemy_num]['Prop']['strong']

    if (my_strong + rd_num) > 150 and (enemy_strong > 40 or (enemy_strong + rd_num) > 150):
        strong_full = True
    
    if (my_strong + rd_num) <= 150 and enemy_strong > 40 and my_strong > enemy_strong:
        strong_high = True
    
    if strong_full or strong_high:
        return True
    else:
        return False

# 检查分数是否升高
def score_raise_check3(Num_, GameInfo_, kill_snack_id, dead, loss_length):
    # 当前分数,排名
    score_frame = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
        d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']
        d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
        score_frame.append(d)
    score_frame = pd.DataFrame(score_frame).rank(method='average')
    score_frame = (score_frame['kill']*1.5 + score_frame['time'] + score_frame['length'])/3.5
    my_score = score_frame[Num_]
    my_rank = score_frame.rank(ascending=False,method='min')[Num_]

    # 杀蛇后的分数与排名
    score_frame_aft = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        if i==Num_:
            if dead:
                d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
                d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill'] + 2
            else:
                d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len'] - loss_length
                d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill'] + 2
        else:
            d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
            d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']

        if GameInfo_["gameinfo"]["Player"][i]['IsDead'] or i == kill_snack_id:
            d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
        else:
            if i==Num_ and dead:
                d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
            else:
                d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time'] + 1
        score_frame_aft.append(d)
    score_frame_aft = pd.DataFrame(score_frame_aft).rank(method='average')
    score_frame_aft = (score_frame_aft['kill']*1.5 + score_frame_aft['time'] + score_frame_aft['length'])/3.5
    my_score_aft = score_frame_aft[Num_]
    my_rank_aft = score_frame_aft.rank(ascending=False,method='min')[Num_]

    if my_score_aft >= my_score or my_rank_aft <= my_rank:
        return True
    else:
        return False


# 当只有一个敌人时，进攻更为激进+阻塞+掠夺
def only_one_enemy(Num_, GameInfo_):
    # 寻找可进行击杀的蛇
    kill_or_not, action = search_target_snack(Num_, GameInfo_)
    if kill_or_not:
        # print('kill strategy normal ( snack == 2 ) activated ! ')
        return action
    
    dead_list, sit_dic = situation(GameInfo_)
    live_enemy_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list and Num_ != i}
    enemy_num = list(live_enemy_dic.keys())[0]
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    if not ((my_length + my_save) >= 7 and (my_strong > 1 or (my_length + my_save) > live_enemy_dic[enemy_num]['length'])):
        return ''

    if live_enemy_dic[enemy_num]['strong'] <= 1:
        kill_or_not, action_string = kill_snack_fatal(Num_, GameInfo_, enemy_num)
        if kill_or_not:
            # print('kill strategy fatal ( snack == 2 ) activated ! ')
            return action_string
        else:
            # print('kill strategy fatal ( snack == 2 ) moving ! ')
            return action_string
    else:
        # 阻塞战略
        # print('kill cancel, now block?')
        _, action_string, _ = block_target_snack(Num_, GameInfo_, enemy_num)
        if len(action_string) != 0:
            # print(' block! ')
            return action_string
        # 资源掠夺战略-力量道具
        # print('block cancel, now rob?')
        if strong_or_length_one_enemy(Num_, GameInfo_):
            # print(' rob length ')
            action_string = rob_target_snack(Num_, GameInfo_, enemy_num, rob_tool=5)
        else:
            # print(' rob strong ')
            action_string = rob_target_snack(Num_, GameInfo_, enemy_num, rob_tool=4)

        if len(action_string) != 0:
            # print(' rob! ')
            return action_string

        # print(' rob cancel ')
        return ''

#当敌人长度小于某个值在自己附近时，即使他有力量道具也可以kill    
def block_to_kill_snack(Num_, GameInfo_, snack_id):
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])

    enemy_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id][0]
    enemy_last_move = len(GameInfo_["gameinfo"]["Player"][snack_id]['Act'])
    enemy_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][snack_id])
    
    pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
    pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
    pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
    pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))

    # 需要堵截的几个位置，需要删除所有已被堵截的部分
    pos_list = [pos1, pos2, pos3, pos4]

    # 包括目标蛇的危险块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]
    
    # 蛇在墙里或四个位置已被占领
    if len(pos_list) == 0:
        return False, '', 10000

    try_list = [list(i) for i in itertools.permutations(pos_list)]
    w_map = map_generate(GameInfo_)
    all_action = ''
    path_cost_list = []
    all_action_list = []
    for pos_l in try_list:
        all_action, path_cost, _ = multi_a_star_search(w_map, my_head, pos_l, cost={0:1, 2:1, 3:1, 4:1, 5:1})
        path_cost_list.append(path_cost)
        all_action_list.append(all_action)
        
    index = np.argmin(path_cost_list)
    all_action = all_action_list[index]
    path_cost = path_cost_list[index]

    if len(all_action)==0:
        return False, '', 10000
    else:
        # 如果满足1.上一轮蛇的移动速度大于等于体长-1, 3.我阻拦所需要的步数小于我体长的1/4
        a = 1/4
        if my_speed >= len(all_action) and enemy_last_move >= (enemy_length - 1) and len(all_action) < (my_length*a):
            return True, all_action, path_cost
        else:
            return False, '', 10000

# 寻找可以堵杀的蛇
def search_block_kill_snack(Num_, GameInfo_):
    dead_list,sit_dic = situation(GameInfo_)
    live_enemy_dic = {i:sit_dic[i] for i in sit_dic.keys() if i not in dead_list and Num_ != i}
    # cost一致, 优先攻击长蛇
    length_sort_list = list(pd.DataFrame(live_enemy_dic).T.sort_values(by='length', ascending=False).index)

    path_cost_list = []
    action_string_list = []
    action_string = ''
    for snack_id in length_sort_list:
        _, action_string, path_cost = block_to_kill_snack(Num_, GameInfo_, snack_id)

        path_cost_list.append(path_cost)
        action_string_list.append(action_string)

    index = np.argmin(path_cost_list)
    action_string = action_string_list[index]
    return action_string

# 获取目标
# 0为空白块，2为糖，3为速度道具，4为力量道具，5为双倍道具
def target_set_advance(GameInfo_, choice=2, overall=False):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    scale = wall_position(GameInfo_, scale=-1)[1]
    w_map = map_generate(GameInfo_)
    # 计算scale次缩圈后的最内层墙环
    x_boundry = (scale, Length-1-scale)
    y_boundry = (scale, Width-1-scale)

    # 寻找的道具
    if choice==0:
        pos = []
        for x in range(Length):
            for y in range(Width):
                if w_map[x,y] == 0:
                    pos.append([x,y])
    if choice==2:
        pos = GameInfo_['gameinfo']['Map']['SugarPosition']
    if choice==3:
        pos = GameInfo_['gameinfo']['Map']['PropPosition'][0]
    if choice==4:
        pos = GameInfo_['gameinfo']['Map']['PropPosition'][1]
    if choice==5:
        pos = GameInfo_['gameinfo']['Map']['PropPosition'][2]

    if not overall:
        pos = [p for p in pos if p[0]>x_boundry[0] and p[0]<x_boundry[1] and p[1]>y_boundry[0] and p[1]<y_boundry[1]]

    return pos

# 返回水池字典，包括水池包含的所有块列表
def scan_area(GameInfo_):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    w_map = map_generate(GameInfo_)
    water_pool = {}
    count = 0
    for x in range(Length):
        for y in range(Width):
            if w_map[x,y]==1:
                continue
            else:
                # 水池发源地
                stack = [[x,y]]
                water_pool[count] = [[x,y]]
                while len(stack) != 0:
                    take = stack[0]
                    stack.remove(take)
                    w_map[take[0],take[1]] = 1

                    four_pos = [[take[0]-1,take[1]],[take[0]+1,take[1]],[take[0],take[1]-1],[take[0],take[1]+1]]
                    available_pos = [pos for pos in four_pos if pos[0] >= 0 and pos[0] < Length and pos[1] >= 0 and pos[1] < Width]
                    for pos in available_pos:
                        if pos in stack or w_map[pos[0],pos[1]]==1:
                            continue
                        else:
                            stack.append(pos)
                            water_pool[count].append(pos)

                count = count + 1
    
    return water_pool

# 返回一长度策略路径
def hierarchy_target_path(Num_, GameInfo_, careful=True, priority=[4,2,5,3,0], noway=False):
    Length = GameInfo_["gameinfo"]["Map"]['Length']
    Width = GameInfo_["gameinfo"]["Map"]['Width']
    # 只在乎落点，不在乎过程
    w_map = np.array([0]*(Length*Width)).reshape(Length,Width)
    wall_pos, scale = wall_position(GameInfo_, scale=-1)
    cost_dict = {0:1, 2:1, 3:1, 4:1, 5:1}
    action_map = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}
    # 危险块
    level1 = wall_pos + snack_body_position(GameInfo_)
    level1 = [[i[0],i[1]]for i in set([(i[0],i[1]) for i in level1])]

    begin = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    level2 = GameInfo_['level2'] #danger_snack_field(Num_, GameInfo_)
    if scale < 19 and not noway:
        for c in priority:
            target_list = target_set_advance(GameInfo_, choice=c, overall=False)
            if careful:
                target_list = [tar for tar in target_list if tar not in level2]
            
            if len(target_list)==0:
                continue
            
            path_cost_list = []
            action_string_list = []
            # 计算cost
            for tar in target_list:
                action_string, path_cost, _ = a_star_search(w_map, begin, tar, cost_dict)
                path_cost_list.append(path_cost)
                action_string_list.append(action_string)

            # 优先选择cost最低,
            while len(action_string_list) != 0:
                index = np.argmin(path_cost_list)
                action_string = action_string_list[index]

                if len(action_string) == 0:
                    break

                head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
                my_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
                my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']

                for action in action_string[:my_speed]:
                    # 头部移动
                    head = list(np.sum([head, action_map[action]], axis=0))
                    my_body = [head] + my_body
                    # 尾部移动
                    if my_save > 0:
                        my_save = my_save - 1
                    else:
                        my_body.pop()
                
                # 是否落点未死亡
                safe = True
                for i in my_body:
                    if i in level1:
                        safe = False
                        break
                    
                if safe:
                    return action_string[:my_speed]
                else:
                    path_cost_list.pop(index)
                    action_string_list.pop(index) 
    else:
        # 所有水池
        pool = GameInfo_['pool']
        size_of_pool = {num:len(pool[num]) for num in pool.keys()}
        size_of_pool = pd.Series(size_of_pool).sort_values(ascending=False)
        available_pool = list(size_of_pool[size_of_pool>(my_length + my_save)].index)

        for pool_num in available_pool:
            for c in priority:
                target_list = target_set_advance(GameInfo_, choice=c, overall=True)
                target_list = [tar for tar in pool[pool_num] if tar in target_list]

                if len(target_list)==0:
                    continue

                path_cost_list = []
                action_string_list = []
                # 计算cost
                for tar in target_list:
                    action_string, path_cost, _ = a_star_search(w_map, begin, tar, cost_dict)
                    path_cost_list.append(path_cost)
                    action_string_list.append(action_string)

                # 优先选择cost最低,
                while len(action_string_list) != 0:
                    index = np.argmin(path_cost_list)
                    action_string = action_string_list[index]

                    if len(action_string) == 0:
                        break

                    head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
                    my_body = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]
                    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']

                    for action in action_string[:my_speed]:
                        # 头部移动
                        head = list(np.sum([head, action_map[action]], axis=0))
                        my_body = [head] + my_body
                        # 尾部移动
                        if my_save > 0:
                            my_save = my_save - 1
                        else:
                            my_body.pop()
                    
                    # 是否落点未死亡
                    safe = True
                    for i in my_body:
                        if i in level1:
                            safe = False
                            break
                        
                    if safe:
                        return action_string[:my_speed]
                    else:
                        path_cost_list.pop(index)
                        action_string_list.pop(index)

      
    return ''

# 分为两部分，当只剩我一个人时，不需要力量道具
# 当存在更多玩家的时候，在力量道具不唯一的时候，才需要力量道具
# 否则任何时候先选星星
# 对路径不考虑任何障碍，只考虑落点是否安全（一级危险块，二级危险块）->跃迁至下一目标位置
# 在没缩圈完成时，都往中间找目标
# 缩圈完成后，尽量往最大的水池跃迁
# 长度为1时蛇的攻略      
def one_length_strategy(Num_, GameInfo_, live_snack):
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    my_head = GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    # 检查安全性
    pos1 = list(np.sum([my_head, [0,1]], axis=0))
    pos2 = list(np.sum([my_head, [0,-1]], axis=0))
    pos3 = list(np.sum([my_head, [1,0]], axis=0))
    pos4 = list(np.sum([my_head, [-1,0]], axis=0))
    pos_list = [pos1, pos2, pos3, pos4]
    # 蛇头附近的安全块
    wall_pos = wall_position(GameInfo_, scale=-1)[0]
    level1 = wall_pos + snack_body_position(GameInfo_)
    save_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    save_list = [[i[0],i[1]] for i in save_list]

    noway = False
    if len(save_list) == 0:
        noway = True

    if len(live_snack)==1:
        action_string = hierarchy_target_path(Num_, GameInfo_, careful=False, priority=[2,5,3,4,0], noway=noway)
    # elif my_strong <= 1:
    #     action_string = hierarchy_target_path(Num_, GameInfo_, careful=True, priority=[4,2,5,3,0], noway=noway)
    # elif my_strong <= 5:
    #     action_string = hierarchy_target_path(Num_, GameInfo_, careful=True, priority=[4,2,5,3,0], noway=noway)
    else:
        action_string = hierarchy_target_path(Num_, GameInfo_, careful=True, priority=[2,5,4,3,0], noway=noway)
    
    
    ActList = {"w":[0,1],"s":[0,-1],"a":[-1,0],"d":[1,0]}

    if len(action_string)==0:
        if noway:
            for i in ActList:
                PositionMove = list(np.sum([my_head, ActList[i]], axis=0))
                # 撞墙死
                if PositionMove in wall_pos:
                    return i
                else:
                    continue
            
            return 'w'
        else:
            for i in ActList:
                PositionMove = list(np.sum([my_head, ActList[i]], axis=0))
                # 撞墙死
                if PositionMove in level1:
                    continue
                else:
                    return i
    else:
        return action_string

# 力量第一或满格
# 长度达到最大水池的2/3
# 长度达到150，随回合增加速度衰减
# 长度120+，超越第二2倍
# 因为杀人策略仍然要执行，所以这个函数返回什么放在哪需要抉择
# 返回是否开启猥琐策略 + 建议速度
def safe_mode(Num_, GameInfo_):
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i: sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    
    if len(live_sit_dic.keys()) == 1:
        return False, 0, 1

    strong_sit = False
    length_sit = False
    # 回合数 力量条件是否满足
    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    second_strong_val = list(pd.DataFrame(live_sit_dic).T.sort_values(by='strong',ascending=False)['strong'])[1]
    if my_strong > second_strong_val or (my_strong + rd_num) > 150:
        strong_sit = True

    # 所有水池，长度条件是否满足
    pool = scan_area(GameInfo_)
    size_of_pool = {num:len(pool[num]) for num in pool.keys()}
    size_of_max_pool = pd.Series(size_of_pool).max()
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    second_length_val = list(pd.DataFrame(live_sit_dic).T.sort_values(by='length',ascending=False)['length'])[1]
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    if (my_length + my_save) >= 120 or (my_length > second_length_val and my_length >= 4*size_of_max_pool):
        length_sit = True

    
    # 建议速度
    # 同时存在与同一水池的蛇
    rank_size_of_pool = pd.Series(size_of_pool).sort_values(ascending=False)

    my_head= GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    pos1 = list(np.sum([my_head, [0,1]], axis=0))
    pos2 = list(np.sum([my_head, [0,-1]], axis=0))
    pos3 = list(np.sum([my_head, [1,0]], axis=0))
    pos4 = list(np.sum([my_head, [-1,0]], axis=0))

    pos_list = [pos1, pos2, pos3, pos4]
    # 蛇头附近的安全块
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    # 所在水池
    flag = False
    tar_pool_num = -1
    for num in rank_size_of_pool.index:
        for pos in pos_list:
            if pos in pool[num]:
                flag = True
                tar_pool_num = num
                break
        if flag:
            break

    # 检查其他蛇
    other_snack = 1
    for snack_id in live_sit_dic.keys():
        if snack_id == Num_:
            continue
        enemy_head= GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
        pos1 = list(np.sum([enemy_head, [0,1]], axis=0))
        pos2 = list(np.sum([enemy_head, [0,-1]], axis=0))
        pos3 = list(np.sum([enemy_head, [1,0]], axis=0))
        pos4 = list(np.sum([enemy_head, [-1,0]], axis=0))
        pos_list = [pos1, pos2, pos3, pos4]
        pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
        pos_list = [[i[0],i[1]] for i in pos_list]

        for pos in pos_list:
            if pos in pool[tar_pool_num]:
                other_snack = other_snack + 1
                break


    # 1.平均剩余块/剩余回合数 * 同池蛇头
    left_round = 151 - rd_num
    recommend_speed = np.ceil(size_of_max_pool/(left_round * other_snack))
    
    if strong_sit and length_sit:
        # print(' safe mode activate! ')
        return True, int(recommend_speed), other_snack
    else:
        return False, 0, other_snack

def head_in_pool(Num_,GameInfo_):
    pool = GameInfo_['pool']
    my_head= GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_][0]
    pos1 = list(np.sum([my_head, [0,1]], axis=0))
    pos2 = list(np.sum([my_head, [0,-1]], axis=0))
    pos3 = list(np.sum([my_head, [1,0]], axis=0))
    pos4 = list(np.sum([my_head, [-1,0]], axis=0))
    pos_list = [pos1, pos2, pos3, pos4]
    level1 = wall_position(GameInfo_, scale=-1)[0] + snack_body_position(GameInfo_)
    pos_list = set([(i[0],i[1]) for i in pos_list])-set([(i[0],i[1]) for i in level1])
    pos_list = [[i[0],i[1]] for i in pos_list]

    if len(pos_list) == 0:
        return {}

    size_of_pool = {num:len(pool[num]) for num in pool.keys()}
    rank_size_of_pool = pd.Series(size_of_pool).sort_values(ascending=False)

    # 检查可能进入的水池
    d = {}
    pos_list_temp = pos_list.copy()
    now_pos_list = pos_list.copy()
    for index in rank_size_of_pool.index:
        for pos in now_pos_list:
            if pos in pool[index]:
                d[index] = pool[index]
                pos_list_temp.remove(pos)

        if len(pos_list_temp)==0:
            break
        else:
            now_pos_list = pos_list_temp.copy()

    return d

def score_raise_check2(Num_, GameInfo_, loss_length):
    # 当前分数,排名
    score_frame = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']
        d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']
        d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time']
        score_frame.append(d)
    score_frame = pd.DataFrame(score_frame).rank(method='average')
    score_frame = (score_frame['kill']*1.5 + score_frame['time'] + score_frame['length'])/3.5
    # my_score = score_frame[Num_]
    my_rank = score_frame.rank(ascending=False,method='min')[Num_]

    # 杀蛇后的分数与排名
    score_frame_aft = []
    for i in range(len(GameInfo_["gameinfo"]["Player"])):
        d = {}
        if i==Num_:
            d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len'] - loss_length 
        else:
            d['length'] = GameInfo_["gameinfo"]["Player"][i]['Score_len']

        d['kill'] = GameInfo_["gameinfo"]["Player"][i]['Score_kill']
        d['time'] = GameInfo_["gameinfo"]["Player"][i]['Score_time'] 
        score_frame_aft.append(d)

    score_frame_aft = pd.DataFrame(score_frame_aft).rank(method='average')
    score_frame_aft = (score_frame_aft['kill']*1.5 + score_frame_aft['time'] + score_frame_aft['length'])/3.5
    # my_score_aft = score_frame_aft[Num_]
    my_rank_aft = score_frame_aft.rank(ascending=False,method='min')[Num_]

    if my_rank_aft <= my_rank + 1:
        return True
    else:
        return False

# 掉头如果引起名次下降就不掉
def turnover_or_not(Num_, GameInfo_, pool):
    dead_list, sit_dic = situation(GameInfo_)
    live_sit_dic = {i: sit_dic[i] for i in sit_dic.keys() if i not in dead_list}
    
    if len(live_sit_dic.keys()) == 1:
        return True

    strong_sit = False
    length_sit = False
    # 回合数 力量条件是否满足
    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    my_speed_time = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['speed']
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length = len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_])
    my_save = GameInfo_["gameinfo"]["Player"][Num_]['SaveLength']
    second_strong_val = list(pd.DataFrame(live_sit_dic).T.sort_values(by='strong',ascending=False)['strong'])[1]
    # 需要给自己额外三秒缓缓,这个一般只有后期触发
    if my_strong > second_strong_val or (my_strong + rd_num) > 150 or ((my_length + my_save - my_strong + 3) < my_speed and my_speed >= 15 and my_speed_time > (my_length + my_save - my_speed + 3)):
        strong_sit = True

    # 所有水池，长度条件是否满足
    size_of_pool = {num:len(pool[num]) for num in pool.keys()}
    size_of_max_pool = pd.Series(size_of_pool).max()
    second_length_val = list(pd.DataFrame(live_sit_dic).T.sort_values(by='length',ascending=False)['length'])[1]
    if (my_length + my_save) >= 60 or my_length >= int(1.5*size_of_max_pool) or (my_length > second_length_val and my_length >= size_of_max_pool):
        length_sit = True

    if strong_sit or length_sit:
        return False
    
    # 积分判定
    loss_length = my_length - 1 - my_speed
    if score_raise_check2(Num_, GameInfo_, loss_length):
        return True
    else:
        return False

def MyStrategy(Num_, GameInfo_, target=None):
    # 2为糖，3为速度道具，4为力量道具，5为双倍道具
    if(GameInfo_["gameinfo"]["Player"][Num_]["IsDead"]):
        print('already dead now')
        return "w", target

    dead_list, sit_dic = situation(GameInfo_)
    live_snack = [ i for i in sit_dic.keys() if i not in dead_list]
    my_state = sit_dic[Num_].values()
    
    pool = scan_area(GameInfo_)
    GameInfo_['pool'] = pool
    GameInfo_['level2'] = danger_snack_field(Num_, GameInfo_)
    pool_dict = head_in_pool(Num_,GameInfo_)
    GameInfo_['my_pool'] = pool_dict

    # 长度为1单独进策略
    # 也许可以跳到某个地方不动
    if len(GameInfo_["gameinfo"]["Map"]["SnakePosition"][Num_]) == 1:
        print('length one strategy:')
        action = one_length_strategy(Num_, GameInfo_, live_snack)
        return action, target
        
    # 水池检测
    my_speed = GameInfo_["gameinfo"]["Player"][Num_]['Speed']
    my_length =  GameInfo_["gameinfo"]["Player"][Num_]['Score_len']
    my_speedtime = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['speed']
    my_strong = GameInfo_["gameinfo"]["Player"][Num_]['Prop']['strong']
    rd_num = int(GameInfo_["gameinfo"]["Map"]['Time'])
    way_left = True
    if len(pool_dict) ==0:
        way_left = False
        max_pool = []
    else:
        size_of_pool = {num:len(pool_dict[num]) for num in pool_dict.keys()}
        rank_size_of_pool = pd.Series(size_of_pool).sort_values(ascending=False)
        max_pool = pool[rank_size_of_pool.index[0]]
    
    # 只剩一只蛇
    if len(live_snack) == 1:
        length_sort_list = list(pd.DataFrame(sit_dic).T.sort_values(by='length',ascending=False)['length'])
        longest_snack = pd.DataFrame(sit_dic).T.sort_values(by='length',ascending=False).index[0]
        len1 = length_sort_list[0]
        len2 = length_sort_list[1]
        if longest_snack == Num_ and len1 != len2:
            print('now kill myself:')
            action = kill_myself(Num_, GameInfo_)
        else:
            # 迅速生长拿第一
            print('grow strategy:')
            if GameInfo_["gameinfo"]["Player"][Num_]['Prop']['double'] > 1:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=2, backup_target=[5,3,4], cost_dict = {0:1, 2:-0.5, 3:0, 4:0, 5:-0.5}, last_target=target, method=3, useful_list=[2,5])
            else:
                action, _, target = target_strategy_wrap(Num_, GameInfo_, max_pool, target_type=5, backup_target=[2,3,4], cost_dict = {0:1, 2:-0.5, 3:0, 4:0, 5:-0.5}, last_target=target, method=3, useful_list=[2,5])
    
    else:
        if len(live_snack) == 2:
            # 更加激进的进攻策略
            print('kill strategy(snack == 2):')
            action = only_one_enemy(Num_, GameInfo_)
            if len(action)==0 and way_left:
                action, target = prepare_Strategy(Num_, GameInfo_, my_state, max_pool, target=target)

        else:
            # 寻找可进行击杀的蛇
            print('kill strategy ( snack > 2 ) :')
            kill_or_not, action = search_target_snack(Num_, GameInfo_)

            if kill_or_not:
                print('kill activated ! ')
                return action, target
           
            if len(action) != 0:
                # 可接近的条件
                if my_speed > 8 and (my_strong > 10 or rd_num + my_strong > 150) and (my_speedtime > 5 or rd_num + my_speedtime > 150) and my_length > 8:
                    print(' close to the enemy ')
                    return action, target
                else:
                    print(' prepare strategy: ')
                    action, target = prepare_Strategy(Num_, GameInfo_, my_state, max_pool, target=target)

            # 如果有舍命阻塞在这里加
            elif way_left:
                # 如果有非舍命阻塞在这里加
                print(' prepare strategy: ')
                action, target = prepare_Strategy(Num_, GameInfo_, my_state, max_pool, target=target)
            else:
                # 自尽判定
                # 转向判定
                action = AI0(Num_, GameInfo_)

    if len(action) == 0:
        print(' no_way here ')
        action = AI0(Num_, GameInfo_)

    return action, target