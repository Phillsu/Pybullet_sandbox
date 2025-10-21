import pybullet as p
import pybullet_data
import time
import numpy as np

# 初始化物理引擎
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# 加载地面
planeId = p.loadURDF("plane.urdf")

# 设置相机位置
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5])

# 加载机器人手臂
robotId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

# 获取关节数量
numJoints = p.getNumJoints(robotId)
print(f"机器人关节数量: {numJoints}")

# 获取关节信息
jointInfo = []
for i in range(numJoints):
    jointInfo.append(p.getJointInfo(robotId, i))
    print(f"关节 {i}: {jointInfo[i][1]}")

# 初始关节位置
initialPositions = [0, 0, 0, 0, 0, 0, 0]

# 设置初始关节位置
def reset_all_joints():
    """重置所有关节到初始位置"""
    for i in range(numJoints):
        p.resetJointState(robotId, i, initialPositions[i])
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=initialPositions[i])
    print("已重置所有关节到初始位置")

# 初始化关节位置
reset_all_joints()

# 关节控制参数
jointPositions = initialPositions.copy()
jointVelocity = 0.5

# 逆运动学参数
targetPosition = [0.5, 0, 0.5]
targetOrientation = p.getQuaternionFromEuler([0, 0, 0])

# 创建可视化目标点
targetVisual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.5])
targetId = p.createMultiBody(baseVisualShapeIndex=targetVisual, basePosition=targetPosition)

# 控制模式
controlMode = "JOINT"  # "JOINT" 或 "IK"

# 选中的关节索引
selectedJoint = 0

try:
    while True:
        # 处理键盘事件
        keys = p.getKeyboardEvents()
        
        for key, state in keys.items():
            if state & p.KEY_WAS_TRIGGERED:
                # 切换控制模式
                if key == ord('m'):
                    controlMode = "IK" if controlMode == "JOINT" else "JOINT"
                    print(f"切换到 {controlMode} 控制模式")
                
                # 重置所有关节
                elif key == ord('r'):
                    reset_all_joints()
                    jointPositions = initialPositions.copy()
                    targetPosition = [0.5, 0, 0.5]
                    p.resetBasePositionAndOrientation(targetId, targetPosition, targetOrientation)
                
                # 退出
                elif key == ord('q'):
                    raise KeyboardInterrupt
                
                # 关节控制模式
                elif controlMode == "JOINT":
                    if key >= ord('1') and key <= ord('7'):
                        selectedJoint = key - ord('1')
                        if selectedJoint < numJoints:
                            print(f"已选择关节 {selectedJoint + 1}")
                        else:
                            print(f"关节 {selectedJoint + 1} 不存在")
                    
                    elif key == ord('w'):
                        if selectedJoint < numJoints:
                            jointPositions[selectedJoint] += jointVelocity * 0.1
                            p.setJointMotorControl2(robotId, selectedJoint, p.POSITION_CONTROL, 
                                                   targetPosition=jointPositions[selectedJoint])
                            print(f"关节 {selectedJoint + 1} 角度: {jointPositions[selectedJoint]:.2f}")
                    
                    elif key == ord('s'):
                        if selectedJoint < numJoints:
                            jointPositions[selectedJoint] -= jointVelocity * 0.1
                            p.setJointMotorControl2(robotId, selectedJoint, p.POSITION_CONTROL, 
                                                   targetPosition=jointPositions[selectedJoint])
                            print(f"关节 {selectedJoint + 1} 角度: {jointPositions[selectedJoint]:.2f}")
                
                # 逆运动学控制模式
                elif controlMode == "IK":
                    # 目标点移动控制
                    move_step = 0.05
                    if key == ord('i'):
                        targetPosition[0] += move_step
                    elif key == ord('k'):
                        targetPosition[0] -= move_step
                    elif key == ord('j'):
                        targetPosition[1] += move_step
                    elif key == ord('l'):
                        targetPosition[1] -= move_step
                    elif key == ord('u'):
                        targetPosition[2] += move_step
                    elif key == ord('o'):
                        targetPosition[2] -= move_step
                    
                    # 更新目标点位置
                    p.resetBasePositionAndOrientation(targetId, targetPosition, targetOrientation)
                    print(f"目标点位置: X={targetPosition[0]:.2f}, Y={targetPosition[1]:.2f}, Z={targetPosition[2]:.2f}")
                    
                    # 计算逆运动学
                    jointPoses = p.calculateInverseKinematics(
                        robotId, 
                        numJoints-1,  # 末端执行器关节
                        targetPosition, 
                        targetOrientation
                    )
                    
                    # 设置关节位置
                    for i in range(numJoints):
                        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=jointPoses[i])
                        jointPositions[i] = jointPoses[i]
        
        # 步进仿真
        p.stepSimulation()
        time.sleep(1./240.)

except KeyboardInterrupt:
    print("退出仿真")

# 断开连接
p.disconnect()