import mmap
import ctypes
import time
import random
from ctypes import wintypes, windll, WinError, byref
from ctypes import c_size_t, c_void_p
import tensorflow as tf
import threading
import numpy as np
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import argparse
import os


def plot_reward(rewards, filename="reward.png"):
    try:
        if not rewards:
            print("empty rewards, quiting")
            return
        plt.plot(rewards)
        
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.savefig(filename)
    except Exception as e:
        print(f"plot exception:{str(e)}")
    finally:
        plt.close()

def save_rewards(rewards, filename="reward.txt"):
    try:
        if not rewards:
            print("empty rewards, quiting")
            return
        with open(filename, 'w') as f:
            for reward in rewards:
                f.write(f"{reward}\n")
    except Exception as e:
        print(f"Error saving rewards to {filename}:{e}")

# 显示声明windows API的参数和返回值类型
kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

#openMutexW
kernel32.OpenMutexW.argtypes = [
    wintypes.DWORD,   # dwDesiredAccess 
    wintypes.BOOL,    # bInheritHandle 
    wintypes.LPCWSTR  # lpName 
]
kernel32.OpenMutexW.restype = wintypes.HANDLE  
# OpenFileMappingW
kernel32.OpenFileMappingW.argtypes = [
    wintypes.DWORD,   # dwDesiredAccess
    wintypes.BOOL,    # bInheritHandle
    wintypes.LPCWSTR  # lpName
]
kernel32.OpenFileMappingW.restype = wintypes.HANDLE
# MapViewOfFile
kernel32.MapViewOfFile.argtypes = [
    wintypes.HANDLE,  # hFileMappingObject
    wintypes.DWORD,   # dwDesiredAccess
    wintypes.DWORD,   # dwFileOffsetHigh
    wintypes.DWORD,   # dwFileOffsetLow
    c_size_t          # dwNumberOfBytesToMap
]
kernel32.MapViewOfFile.restype = c_void_p  # 关键：返回64位指针

# CloseHandle
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CloseHandle.restype = wintypes.BOOL

# UnmapViewOfFile
kernel32.UnmapViewOfFile.argtypes = [c_void_p]
kernel32.UnmapViewOfFile.restype = wintypes.BOOL
# WaitForSingleObject
kernel32.WaitForSingleObject.argtypes = [
    wintypes.HANDLE,  # hHandle
    wintypes.DWORD    # dwMilliseconds
]
kernel32.WaitForSingleObject.restype = wintypes.DWORD

# ReleaseMutex
kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
kernel32.ReleaseMutex.restype = wintypes.BOOL

# -------------共享内存结构体---------
LOCAL_BUFFER_SIZE = 256
STATE_DIMENTION = 7
class InteractionData(ctypes.Structure):
    _fields_ = [
        ("state", ctypes.c_float * 7),
        ("reward", ctypes.c_float)
    ]

class CWriteStruct(ctypes.Structure):
    _fields_ = [
        ("states_reward", InteractionData),
        ("has_new", ctypes.c_int)
    ]

class CReadStruct(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_float),
        ("has_new", ctypes.c_int)
    ]
class RLAgent:
    def __init__(self, model_path="rl_model_weights.weights.h5"):
        self.norm_layer = tf.keras.layers.Normalization(axis=-1, input_shape=(STATE_DIMENTION,))
        self.model = tf.keras.Sequential([
            self.norm_layer,
            tf.keras.layers.Dense(64, activation="relu", input_shape=(STATE_DIMENTION,), kernel_initializer='he_normal'),
            tf.keras.layers.Dense(2),
            tf.keras.layers.Lambda(lambda x: tf.concat([
            tf.sigmoid(x[:, 0:1]),  # mu ∈ (-1,1)
            tf.tanh(x[:, 1:2]) * 2.0 - 3.0  # log_var ∈ [-5, -1] → std ∈ [0.05, 0.37]
        ], axis=1))
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        self.buffer = []
        self.lock = threading.Lock()# selfbufferlock
        self.model_lock = threading.Lock()#modellock
        self.model_path = model_path
        self.gamma = 0.9
        self._initialize_model()
    def _initialize_model(self):
        try:
            self.load_weights()
        except:
            with self.model_lock:
                print(f"Model file {self.model_path} not found, initialing new model")
                self.save_weights()
    def save_weights(self):
        try:
            self.model.save_weights(self.model_path)
            print(f"Saved model weights to {self.model_path}")
        except Exception as e:
            print(f"Error saving model weights to {self.model_path}:{e}")   
    def load_weights(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file {self.model_path} not found")
            with self.model_lock:
                self.model.load_weights(self.model_path)
                print(f"Loaded model weights from {self.model_path}")
        except :
            print(f"Failed to load model weights from {self.model_path}")
            raise
    def add_experience(self, state, reward):
        """
        with self.lock:
            if len(self.buffer) < 100:  # 用前100个样本更新归一化参数
                self.norm_layer.adapt(np.array([state]))
            self.buffer.append((state, reward))
            while len(self.buffer) >= LOCAL_BUFFER_SIZE:
                self.buffer.pop(0)        
        """
        with self.lock:
            self.buffer.append((state, reward))
            
            # 维护缓冲区大小
            while len(self.buffer) >= LOCAL_BUFFER_SIZE:
                self.buffer.pop(0)
            
            # 仅在前100个样本时初始化归一化层（关键修改！）
            if not hasattr(self, '_norm_adapted'):
                if len(self.buffer) >= 100:
                    # 提取前100个state作为二维数组（shape=(100,7)）
                    init_states = np.array([s for s, _ in self.buffer[:100]])
                    
                    # 正确初始化归一化层（批量adapt）
                    self.norm_layer.adapt(init_states)  
                    
                    # 打印验证信息
                    print(f"归一化层初始化完成！均值：{self.norm_layer.mean.numpy()}")
                    print(f"标准差：{self.norm_layer.variance.numpy()**0.5}")
                    
                    # 设置标志位避免重复初始化
                    self._norm_adapted = True
            if hasattr(self, '_norm_adapted') and len(self.buffer) % 32 == 0:
                sample_state = self.buffer[-1][0]
                normalized = self.norm_layer(np.array([sample_state])).numpy()[0]
                print(f"归一化样本检查: {normalized} (Max={np.max(normalized):.2f}, Min={np.min(normalized):.2f})")            

    def predict_action(self, state):
        #return np.clip(np.random.normal(0.33, 0.2), 0.1, 0.9)
        with self.model_lock:
            if len(state.shape) == 1:
                state = np.expand_dims(state, axis=0)
            mu_logvar = self.model.predict(state, verbose=0)[0]
            mu, log_var = mu_logvar[0], mu_logvar[1]
            
            # 计算标准差（确保正数）
            log_var = tf.clip_by_value(log_var, -5, 2)  # 对应方差范围: e^-20 ~ e^2
            std = tf.exp(log_var * 0.5)
            
            # 从分布中采样
            action = mu + std * tf.random.normal(shape=())
            
            print(f"raw_action:{action.numpy()}, mu_norm:{mu:.4f}, std:{std.numpy():.4f}")
            action = tf.clip_by_value(action, 0.0, 1.0)
            # 映射到 [0,1] 范围
            return float(action) 
    def _calculate_returns(self, rewards):
        G = 0
        returns = []
        for r in reversed(rewards):  # 反向遍历：从最后一个奖励开始
            G = r + self.gamma * G   # 计算累积奖励
            returns.insert(0, G)     # 插入到列表头部
        return np.array(returns)
    def train(self):
        if not hasattr(self, '_norm_adapted') or len(self.buffer) < 32:
            return
        #非阻塞训练
        def async_train():
            with self.model_lock:
                states = np.array([x[0] for x in self.buffer[0:31]])
                rewards = np.array([x[1] for x in self.buffer[1:32]])
                
                #添加长期奖励的计算
                returns = self._calculate_returns(rewards)
                # 标准化奖励
                returns_mean = np.mean(returns)
                returns_std = np.std(returns)
                if returns_std < 1e-8:
                    returns = returns - returns_mean  # 仅中心化
                else:
                    returns = (returns - returns_mean) / (returns_std + 1e-8)
                print(f"Returns - mean: {returns_mean:.4f}, std: {returns_std:.4f}")
                #REINFORCE算法
                with tf.GradientTape() as tape:
                    # 获取动作分布参数
                    mus_logvars = self.model(states)
                    mus, log_vars = mus_logvars[:, 0], mus_logvars[:, 1]
                    
                    # 构建概率分布
                    log_vars = tf.clip_by_value(log_vars, -20, 2)
                    stds = tf.exp(log_vars * 0.5) 
                    dist = tfp.distributions.Normal(loc=mus, scale=stds)
                    
                    # 重新采样动作
                    sampled_actions = dist.sample()
                    log_probs = dist.log_prob(sampled_actions) 
                    log_probs = tf.clip_by_value(log_probs, -50, 50)

                    # 用长期奖励替代单步奖励
                    loss = -tf.reduce_mean(log_probs * returns)
                    

                    # 计算梯度
                grads = tape.gradient(loss, self.model.trainable_variables)
                grad_norms = [tf.norm(g).numpy() for g in grads]
                print(f"梯度范数: {grad_norms}")  # 监控各层梯度
                grads, _ = tf.clip_by_global_norm(grads, 5.0)
                self.optimizer.apply_gradients(
                        zip(grads,self.model.trainable_variables)
                    )
                with self.lock:
                    self.buffer = self.buffer[32:]
                self.save_weights()
        print("async_training")
        threading.Thread(target=async_train).start()
def print_error(action):
    """打印Windows API错误信息"""
    error_code = ctypes.GetLastError()
    print(f"[Python] 错误 ({action}): 错误码 {error_code} - {WinError(error_code)}")

def force_reload_structure(address,structure_type):
    buffer = (ctypes.c_byte * ctypes.sizeof(structure_type)).from_address(address)
    return structure_type.from_buffer_copy(buffer)

def main(args):
    log_file = open('rl_log.csv', 'w', newline='')
    all_rewards = []
    log_writer = csv.writer(log_file)
    log_writer.writerow(['cwnd','Action', 'Throughput','MaxThroughput','delay','mindelay','minrtt','Reward'])  

    hMapFile_cwrite = hMapFile_cread = None
    cwrite_ptr = cread_ptr = None
    cwrite = cread = None
    cwrite_mutex = cread_mutex = None

    try:
        # 打开cread共享内存
        while True:
            hMapFile_cread = kernel32.OpenFileMappingW(wintypes.DWORD(0xF001F), wintypes.BOOL(False), u"Local\\cread_shm")
            if hMapFile_cread:
                break
            time.sleep(0.5)
            print("[Python] 等待cread共享内存...")  
        cread_ptr = kernel32.MapViewOfFile(hMapFile_cread, wintypes.DWORD(0xF001F), 0, 0, ctypes.sizeof(CReadStruct))
        if not cread_ptr:
            print_error("Map cread pointer")
            return
        # 打开cwrite共享内存
        hMapFile_cwrite = kernel32.OpenFileMappingW(wintypes.DWORD(0xF001F), wintypes.BOOL(False), u"Local\\cwrite_shm")
        if hMapFile_cwrite == 0:
            print_error("open cwrite shared memory")
            return 
        cwrite_ptr = kernel32.MapViewOfFile(hMapFile_cwrite, wintypes.DWORD(0xF001F), 0, 0, ctypes.sizeof(CWriteStruct))
        if not cwrite_ptr:
            print_error("Map cwrite pointer")
            return
        buffer_cwrite = (ctypes.c_byte * ctypes.sizeof(CWriteStruct)).from_address(cwrite_ptr)
        cwrite = CWriteStruct.from_buffer_copy(buffer_cwrite)

        buffer_cread = (ctypes.c_byte * ctypes.sizeof(CReadStruct)).from_address(cread_ptr)
        cread = CReadStruct.from_buffer_copy(buffer_cread)
        #打开mutex
        cwrite_mutex  = kernel32.OpenMutexW(
            0x1F0001, False, u"Local\\cwrite_mutex")
        if cwrite_mutex == 0:
            print_error("open cwrite mutex")
            return
        cread_mutex = kernel32.OpenMutexW(
            0x1F0001, False, u"Local\\cread_mutex")
        if cread_mutex == 0:
            print_error("open cread mutex")
            return
        
        # 主循环
        agent = RLAgent(model_path=args.model)
        states_buffer = []
        print("[Pythons] shared memory connected")
        while True:
            time.sleep(0.5)
            has_new_data = False
            # 保留最新的state用于写入日志。
            latest_state = None
            latest_reward = None
            # Read from cwrite
            wait_result = kernel32.WaitForSingleObject(cwrite_mutex, 2000)
            if wait_result == 0xFFFFFFFF:  # WAIT_FAILED
                print_error("等待cwrite互斥锁")
                break
            elif wait_result == 0x00000102:  # WAIT_TIMEOUT
                print("[Python] 获取cwrite锁超时")
                continue
            try:
                cwrite = force_reload_structure(cwrite_ptr, CWriteStruct)
                has_new = cwrite.has_new

                if has_new > 0:
                    data = cwrite.states_reward
                    latest_state = list(data.state)
                    latest_reward = data.reward 
                    states_buffer.append(latest_state)
                    all_rewards.append(latest_reward)
                    agent.add_experience(list(data.state), data.reward)
                    has_new_data = True  # 标记有新数据

                    # 更新共享内存中的tail
                    updated_cwrite = force_reload_structure(cwrite_ptr, CWriteStruct)
                    updated_cwrite.has_new = 0
                    ctypes.memmove(cwrite_ptr, ctypes.byref(updated_cwrite), ctypes.sizeof(CWriteStruct))
                #else:
                #    print(f"[python] no data in cwrite, cwrite.head = {cwrite.head}, cwrite.tail = {cwrite.tail}")
            finally:
                if not kernel32.ReleaseMutex(cwrite_mutex):
                    print_error("release cwrite mutex")
            while len(states_buffer) >= LOCAL_BUFFER_SIZE:
                states_buffer.pop(0)
            agent.train()
            # Generate c and write to cread
            if has_new_data and latest_state:
                latest_state = states_buffer[-1]
                if not latest_state:
                    print_error("empty state buffer when action needed")
                    return
                #action = agent.predict_action(np.array([latest_state]))[0][0]
                action = agent.predict_action(np.array(latest_state))
                wait_result = kernel32.WaitForSingleObject(cread_mutex, 2000)
                if wait_result == 0xFFFFFFFF:
                    print_error("等待cread互斥锁")
                    break
                elif wait_result == 0x00000102:
                    print("[Python] 获取cread锁超时")
                    continue
                
                try:
                    cread = force_reload_structure(cread_ptr, CReadStruct)
                    cread.action = float(action)
                    cread.has_new = 1
                    ctypes.memmove(cread_ptr, ctypes.byref(cread), ctypes.sizeof(CReadStruct))


                    throughput = latest_state[4]
                    delay = latest_state[1]
                    max_throughput = latest_state[5]
                    min_delay = latest_state[6]
                    ackgap = latest_state[3]
                    cwnd = latest_state[0]
                    minRtt = latest_state[2]
                    log_writer.writerow([f'action:{cread.action}|', f'cwnd:{cwnd}|',f"throughput:{throughput}|", f"max_throughput:{max_throughput}|",f"delay:{delay}us|",f"min_delay:{min_delay}us|", f"minRtt:{minRtt}us|",f"ackgap:{ackgap}|", f"latest_reward:{latest_reward}|"])
                    log_file.flush()
                finally:                
                    if not kernel32.ReleaseMutex(cread_mutex):
                        print_error("release cread mutex")
    except KeyboardInterrupt:
        print("\n[Python] keyboard interrupt, cleaning up")
    except Exception as e:
        print(f"[Python] error:{str(e)}")
    finally:
        plot_reward(all_rewards, filename=args.plot)
        save_rewards(all_rewards, filename=args.reward_file)
        if 'log_file' in locals():
            log_file.close()
            print("[Python] Log file closed")
        if cwrite_ptr:
            kernel32.UnmapViewOfFile(cwrite_ptr)
        if cread_ptr:
            kernel32.UnmapViewOfFile(cread_ptr)        
        if hMapFile_cwrite:
            kernel32.CloseHandle(hMapFile_cwrite)
        if hMapFile_cwrite:
            kernel32.CloseHandle(hMapFile_cread)
        if cwrite_mutex:
            kernel32.CloseHandle(cwrite_mutex)
        if cread_mutex:
            kernel32.CloseHandle(cread_mutex)
        print("[Python] shared memory cleaned up")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Agent with Shared Memory")
    parser.add_argument("--model", type=str, default="rl_model_weights.weights.h5", help="Path to the saved model")
    parser.add_argument("--plot", type=str, default="reward.png", help="Path to save the reward plot")
    parser.add_argument("--reward_file", type=str, default="reward.txt", help="filename to save the reward values")
    args = parser.parse_args()
    main(args)