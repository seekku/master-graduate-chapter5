训练代码脚本：lat_planner.py
目前的思路就是利用两个FQF网络来进行横纵向规划问题,对应纵向agent和横向lat_agent。

节点存放：
version1.0当中存放的是 6m为一次决策距离，并且
  def _lat_get_reward(self):
    #这里就是看换道路
    if  self.current_L == self.target_L:
      r = 0.5
    elif abs( self.current_L-self.target_L)>1:
      r = -1
    else:
      r = 0

    if self.target_L == 0:
      r += 0.5

    return r 

version1.5存放  没考虑a.x
version2.0存放  reward 考虑a.x  ：

version3.0存放  10m为一次决策距离 ，reward 考虑a.x的


