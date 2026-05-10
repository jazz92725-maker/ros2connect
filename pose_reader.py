import rclpy
from rclpy.node import Node
from isaac_ros_tensor_list_interfaces.msg import TensorList
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import zmq
import threading
import json

class PoseServerNode(Node):
    def __init__(self):
        super().__init__('pose_server_node')
        
        # 儲存最新的姿態資料
        self.latest_pose = None
        self.pose_lock = threading.Lock()

        # 1. 啟動 ROS 2 訂閱
        self.sub = self.create_subscription(
            TensorList, 
            '/pose_estimation/pose_matrix_output', 
            self.callback, 
            10)
        
        # 2. 啟動 ZMQ Server (在背景執行緒)
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind("tcp://*:5555")
        
        self.zmq_thread = threading.Thread(target=self.zmq_server_loop, daemon=True)
        self.zmq_thread.start()

        self.get_logger().info("✅ 姿態解碼器與 ZMQ 伺服器 (Port 5555) 已啟動！正在等候數據...")

    def callback(self, msg):
        if not msg.tensors: return
        tensor = msg.tensors[0]
        
        # 解壓縮 16 個 Float32 浮點數
        floats = struct.unpack('<16f', bytes(tensor.data))
        
        # 1. 提取平移向量 (Translation: X, Y, Z) - 注意單位，這裡假設是 meter 轉成 mm 讓 KUKA 好處理
        tx = floats[12] * 1000.0  
        ty = floats[13] * 1000.0
        tz = floats[14] * 1000.0
        
        # 2. 提取 3x3 旋轉矩陣
        rot_matrix = np.array([
            [floats[0], floats[4], floats[8]],
            [floats[1], floats[5], floats[9]],
            [floats[2], floats[6], floats[10]]
        ])
        
        # 3. 轉換為尤拉角 (Roll, Pitch, Yaw)
        try:
            r = R.from_matrix(rot_matrix)
            euler_angles = r.as_euler('xyz', degrees=True)
            roll, pitch, yaw = euler_angles[0], euler_angles[1], euler_angles[2]
            
            # 更新最新姿態 (加鎖保護)
            with self.pose_lock:
                self.latest_pose = {
                    "x": tx, "y": ty, "z": tz,
                    "roll": roll, "pitch": pitch, "yaw": yaw
                }
            
            # 印出 Log 供除錯
            self.get_logger().debug(f"更新姿態: X:{tx:.1f}, Y:{ty:.1f}, Z:{tz:.1f}")
            
        except ValueError:
            pass

    def zmq_server_loop(self):
        """ 背景執行的 ZMQ 伺服器，處理手臂控制程式的請求 """
        while rclpy.ok():
            try:
                # 接收請求 (無阻塞或設 timeout 以免卡死)
                message = self.zmq_socket.recv_string()
                
                if message == "GET_POSE":
                    with self.pose_lock:
                        if self.latest_pose is not None:
                            # 傳送 JSON 格式的姿態資料
                            self.zmq_socket.send_string(json.dumps(self.latest_pose))
                        else:
                            # 如果還沒收到過任何 ROS 訊息
                            self.zmq_socket.send_string(json.dumps({"error": "No pose data yet"}))
            except Exception as e:
                self.get_logger().error(f"ZMQ Error: {e}")

def main():
    rclpy.init()
    node = PoseServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
