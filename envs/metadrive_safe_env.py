"""
这是如何使用 SafeMetaDriveEnv 环境的示例。
我们将使用下面的 VALIDATION_CONFIG 来评估训练好的智能体的"基线性能"。
一个隐藏的测试集将用于评估你训练的智能体的"最终性能"。

你可以直接运行这个文件,使用键盘控制在训练环境中的车辆。
"""
import copy
import os
import sys

import numpy as np

DEFAULT_METADRIVE_SOURCE = "/home/ac/@Lyz-Code/safeRL-metadrive/metadrive"


def ensure_metadrive_source(metadrive_source=None):
    """Expose a local MetaDrive checkout if no installed package is available."""
    source = os.path.expanduser(metadrive_source or DEFAULT_METADRIVE_SOURCE)
    if os.path.isdir(os.path.join(source, "metadrive")) and source not in sys.path:
        sys.path.insert(0, source)


try:
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
except ImportError:
    ensure_metadrive_source()
    try:
        from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
    except ImportError:
        from metadrive.metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

from envs.metadrive_risk_field import RiskFieldCalculator

# 默认配置参数
DEFAULT_CONFIG = {
    # ===== 环境难度设置 =====
    "accident_prob": 0.8,        # 事故概率
    "traffic_density": 0.05,     # 交通密度
    
    # ===== 终止条件 =====
    "crash_vehicle_done": False,  # 车辆碰撞后是否结束episode
    "crash_object_done": False,   # 碰撞障碍物后是否结束episode
    # "crash_object_done": True,   # 碰撞障碍物后是否结束episode
    # "crash_vehicle_done": True,  # 车辆碰撞后是否结束episode

    # "out_of_road_done": False,    # 偏离道路后是否结束episode

    # ===== 奖励设置 =====
    "success_reward": 10.0,       # 成功完成任务的奖励
    "driving_reward": 1.0,        # 驾驶行为奖励
    "speed_reward": 0.1,          # 速度奖励
    
    # ===== 惩罚设置(将被取负值并加到奖励中) =====
    "out_of_road_penalty": 5.0,   # 偏离道路的惩罚
    "crash_vehicle_penalty": 1.0, # 碰撞车辆的惩罚
    "crash_object_penalty": 1.0,  # 碰撞障碍物的惩罚
    # "out_of_road_penalty": 0.0,   # 偏离道路的惩罚
    # "crash_vehicle_penalty": 0.0, # 碰撞车辆的惩罚
    # "crash_object_penalty": 0.0,  # 碰撞障碍物的惩罚
    # ===== 成本设置(将在 info["cost"] 中返回,可用于约束优化) =====
    "crash_vehicle_cost": 1.0,    # 碰撞车辆的成本
    "crash_object_cost": 1.0,     # 碰撞障碍物的成本
    "out_of_road_cost": 1.0,      # 偏离道路的成本

    # ===== IDM调试策略设置 =====
    "enable_idm_lane_change": True,     # IDM自车是否允许变道
    "disable_idm_deceleration": False,  # 是否禁用IDM减速逻辑



    #！！！ ===== MetaDrive原生风险场成本配置 =====   ！！！#
    # 控制是否启用风险场（Potential Risk Field）作为安全约束信号
    "use_risk_field_cost": False,          # 是否将风险场计算结果接入 info["cost"]，用于Safe RL的成本约束优化
    
    # 成本尺度映射：风险场原始值是连续势场强度，进入Safe RL的cost需压缩到“碰撞等价”尺度。
    "risk_field_event_cost_weight": 1.0,      # MetaDrive原始离散事件成本的保留权重（碰撞、偏离道路等二进制事件）
    "risk_field_cost_weight": 0.2,            # 原始风险场进入饱和映射前的缩放系数，越大越敏感
    "risk_field_cost_transform": "event_squash",  # event_squash: 1-exp(-x)压缩；linear_clip: 线性裁剪
    "risk_field_collision_equivalent_cost": 1.0,  # 风险场单步最大值，默认等价一次碰撞cost=1
    "risk_field_cost_clip": 1.0,              # 压缩后的风险场成本上限，默认不超过一次碰撞
    "risk_field_cost_combine": "max",         # max避免碰撞步双重计数；sum可恢复旧的事件+风险相加
    
    # 感知范围配置
    "risk_field_max_distance": 40.0,      # 风险场计算的最大感知半径（米），仅统计该范围内的周围车辆和障碍物
    
    # ========== 风险组件权重配置 ==========
    # 控制各个风险组件对总成本的贡献比例，可通过调整权重实现不同的安全策略
    
    # 1. 道路几何风险
    "risk_field_boundary_weight": 1.0,    # 道路边界风险权重（到左右边界的距离惩罚）
    "risk_field_lane_weight": 0.2,        # 车道线风险权重（到车道边缘的距离惩罚，较低以避免过度限制变道）
    "risk_field_offroad_weight": 1.0,     # 偏离道路风险权重（在不可行驶区域的风险带惩罚）
    
    # 2. 动态物体风险
    "risk_field_vehicle_weight": 1.0,     # 周围车辆风险权重（包含静态占据+动态速度风险）
    "risk_field_object_weight": 0.8,      # 静态障碍物风险权重（交通锥、路障等，略低于车辆）
    
    # 3. 预测性安全指标
    "risk_field_headway_weight": 1.0,     # 车头时距风险权重（跟车过近的惩罚，行业标准安全指标）
    "risk_field_ttc_weight": 1.0,         # 碰撞时间风险权重（TTC过低时的预警惩罚，预测性安全指标）
    
    # ========== 风险场形状参数（Sigma）==========
    # 控制高斯/超高斯分布的扩散范围，sigma越大风险场越宽泛
    
    # 道路几何sigma
    "risk_field_boundary_sigma": 0.75,    # 边界风险的sigma（米），默认0.75m表示距离边界0.75m时风险降至exp(-0.5)≈0.61
    "risk_field_lane_edge_sigma": 0.75,   # 车道线风险的sigma（米），与边界类似但权重更低
    
    # 车道线类型惩罚因子（根据交通规则设置不同权重）
    "risk_field_broken_line_factor": 0.05,   # 虚线惩罚因子：允许变道，几乎不惩罚（基准值）
    "risk_field_solid_line_factor": 0.60,    # 实线惩罚因子：禁止变道，中等惩罚（12倍于虚线）
    "risk_field_boundary_line_factor": 1.0,  # 边界线惩罚因子：道路边缘，高惩罚（20倍于虚线）
    "risk_field_oncoming_line_factor": 1.50, # 对向线惩罚因子：黄色分隔线，最高惩罚（30倍于虚线，严禁逆行）
    
    # 偏离道路参数
    "risk_field_offroad_cost": 1.0,       # 偏离道路的基准成本值
    "risk_field_offroad_sigma": 1.0,      # 偏离道路风险带的sigma（米），采用边缘带衰减模型避免整片路外区域标红
    "risk_field_on_lane_margin": 0.05,    # 判定"在道路上"的容差边距（米），考虑数值误差
    
    # ========== 车辆风险参数 ==========
    # 超高斯分布参数：E = exp(-((long/sigma_long)^beta + (lat/sigma_lat)^beta))
    
    "risk_field_vehicle_longitudinal_sigma": 5.0,   # 车辆纵向sigma（米），默认5.0m覆盖前后安全距离
    "risk_field_vehicle_lateral_sigma": 1.6,        # 车辆横向sigma（米），默认1.6m覆盖车身宽度+安全余量
    "risk_field_vehicle_beta": 2.0,                 # 超高斯指数，beta=2为标准高斯，beta>2使风险场更"方形"
    
    # 动态风险参数（与相对速度相关）
    "risk_field_vehicle_dynamic_sigma_scale": 2.0,  # 动态sigma的速度缩放系数：sigma_dynamic = scale * |v_other - v_ego|
    "risk_field_vehicle_dynamic_alpha": 0.9,        # 非对称性参数：控制前车/后车风险差异的Sigmoid函数偏移
    "risk_field_vehicle_min_dynamic_sigma": 0.5,    # 动态sigma的最小值（米），防止速度差接近0时除零或数值不稳定
    
    # ========== 静态障碍物风险参数 ==========
    # 与车辆类似但无动态部分（因为障碍物静止）
    "risk_field_object_longitudinal_sigma": 5.0,    # 障碍物纵向sigma（米）
    "risk_field_object_lateral_sigma": 1.6,         # 障碍物横向sigma（米）
    "risk_field_object_beta": 2.0,                  # 障碍物超高斯指数
    
    # ========== 预测性安全指标阈值 ==========
    # 基于交通安全领域的黄金标准指标
    
    "risk_field_headway_time_threshold": 1.2,   # 车头时距阈值（秒），行业标准最小安全时距，低于此值开始惩罚
    "risk_field_ttc_threshold": 3.0,            # 碰撞时间阈值（秒），常见的TTC预警阈值，低于此值开始惩罚
    "risk_field_min_speed": 0.5,                # 最小速度阈值（m/s），低于此速度时跳过Headway/TTC计算（避免除零）
    
    # 预测性指标的成本裁剪
    "risk_field_headway_cost_clip": 3.0,        # 车头时距成本上限，使用对数惩罚：-ln(t/t_thresh)，裁剪避免极端值
    "risk_field_ttc_cost_clip": 3.0,            # TTC成本上限，同样使用对数惩罚并裁剪
    
    # 总风险裁剪
    "risk_field_raw_clip": 10.0,                # 原始风险场总和的上限（加权前），防止多个风险叠加导致数值爆炸




    # ===== 偏离道路新模式配置 =====
    # legacy: 保持MetaDrive原始逻辑（一旦偏离立即终止）
    # warning_budget: 偏离道路先触发有限次数warning，允许短暂纠正后再终止
    "out_of_road_mode": "legacy",               # 偏离道路处理模式："legacy" 或 "warning_budget"
    # "out_of_road_mode": "warning_budget",     # （备选）宽容模式，允许有限次数的短暂偏离
    "out_of_road_warning_limit": 5,             # warning模式下允许的偏离警告次数上限
    "out_of_road_warning_penalty": 1.0,         # 每次触发warning时的奖励惩罚值
    "out_of_road_warning_cost": 1.0,            # 每次触发warning时的成本值
    "out_of_road_recovery_steps": 15,           # 偏离后允许的自我纠正步数，在此步数内回到道路则不计入warning
    "out_of_road_terminate_after_budget": True, # 超出warning次数上限后是否终止episode
}

# 使用深拷贝避免修改 DEFAULT_CONFIG
TRAINING_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
TRAINING_CONFIG.update(
    {  
        # ===== 训练环境设置 =====
        "num_scenarios": 50,   # 总共有50种可能的地图场景
        "start_seed": 100,     # 使用种子在 [100, 150) 范围内的地图作为默认训练环境
    }
)


def get_training_env(extra_config=None):
    """
    获取训练环境
    
    Args:
        extra_config: 额外的配置参数,将覆盖默认配置
        
    Returns:
        SafeMetaDriveEnv: 配置好的训练环境实例
    """
    config = copy.deepcopy(TRAINING_CONFIG)
    if extra_config:
        config.update(extra_config)
    return SafeMetaDriveEnv_mini(config)


# 验证环境配置
VALIDATION_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
VALIDATION_CONFIG.update(
    {  
        # ===== 验证环境设置 =====
        "num_scenarios": 10,    # 总共有50种可能的地图场景
        "start_seed": 1000,     # 使用种子在 [1000, 1050) 范围内的地图作为默认验证环境
    }
)


def get_validation_env(extra_config=None):
    """
    获取验证环境
    
    Args:
        extra_config: 额外的配置参数,将覆盖默认配置
        
    Returns:
        SafeMetaDriveEnv: 配置好的验证环境实例
    """
    config = copy.deepcopy(VALIDATION_CONFIG)
    if extra_config:
        config.update(extra_config)
    return SafeMetaDriveEnv_mini(config)


class SafeMetaDriveEnv_mini(SafeMetaDriveEnv): 
    """SafeMetaDriveEnv 的简化版本"""
    
    def default_config(self):
        """返回默认配置"""
        config = super(SafeMetaDriveEnv_mini, self).default_config()
        config.update(
            DEFAULT_CONFIG,
            allow_add_new_key=True  # 允许添加新的配置键
        )
        return config

    def _reset_out_of_road_warning_state(self):
        self._out_of_road_warning_count = 0
        self._out_of_road_recovery_remaining = 0
        self._out_of_road_warning_active = False
        self._out_of_road_budget_exhausted = False

    def _is_warning_budget_mode(self):
        return self.config.get("out_of_road_mode", "legacy") == "warning_budget"

    def _get_risk_field_calculator(self):
        calculator = getattr(self, "_risk_field_calculator", None)
        if calculator is None:
            calculator = RiskFieldCalculator(self.config)
            self._risk_field_calculator = calculator
        return calculator

    @staticmethod
    def _safe_float_config(value, default):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _risk_field_collision_equivalent_cost(self):
        """Return the per-step risk cost that represents one collision-equivalent event."""
        return max(
            self._safe_float_config(self.config.get("risk_field_collision_equivalent_cost", 1.0), 1.0),
            0.0,
        )

    def _risk_field_event_equivalent_cost(self, risk_cost):
        """Map raw continuous risk-field potential into bounded event-equivalent cost."""
        collision_equivalent_cost = self._risk_field_collision_equivalent_cost()
        risk_cost_clip = self.config.get("risk_field_cost_clip", collision_equivalent_cost)
        if risk_cost_clip is None:
            risk_cost_clip = collision_equivalent_cost
        risk_cost_clip = max(self._safe_float_config(risk_cost_clip, collision_equivalent_cost), 0.0)
        upper_bound = min(collision_equivalent_cost, risk_cost_clip)
        if upper_bound <= 0.0:
            return 0.0

        raw_risk_cost = max(self._safe_float_config(risk_cost, 0.0), 0.0)
        risk_weight = max(self._safe_float_config(self.config.get("risk_field_cost_weight", 1.0), 1.0), 0.0)
        scaled_risk_cost = raw_risk_cost * risk_weight
        transform = str(self.config.get("risk_field_cost_transform", "event_squash")).lower()

        if transform == "event_squash":
            mapped_cost = upper_bound * (1.0 - np.exp(-scaled_risk_cost / max(upper_bound, 1e-6)))
        elif transform == "linear_clip":
            mapped_cost = scaled_risk_cost
        else:
            raise ValueError(
                "Unsupported risk_field_cost_transform '{}'. Expected 'event_squash' or 'linear_clip'.".format(
                    transform
                )
            )
        return float(np.clip(mapped_cost, 0.0, upper_bound))

    def _combine_event_and_risk_cost(self, event_cost, risk_cost):
        event_cost = float(self.config.get("risk_field_event_cost_weight", 1.0)) * float(event_cost)
        combine_mode = str(self.config.get("risk_field_cost_combine", "max")).lower()
        if combine_mode == "max":
            return max(event_cost, float(risk_cost))
        if combine_mode == "sum":
            return event_cost + float(risk_cost)
        if combine_mode == "risk_only":
            return float(risk_cost)
        if combine_mode == "event_only":
            return event_cost
        raise ValueError(
            "Unsupported risk_field_cost_combine '{}'. Expected 'max', 'sum', 'risk_only', or 'event_only'.".format(
                combine_mode
            )
        )

    def _zero_risk_field_info(self):
        collision_equivalent_cost = self._risk_field_collision_equivalent_cost()
        return {
            "risk_field_cost": 0.0,
            "risk_field_road_cost": 0.0,
            "risk_field_boundary_cost": 0.0,
            "risk_field_lane_cost": 0.0,
            "risk_field_offroad_cost": 0.0,
            "risk_field_vehicle_cost": 0.0,
            "risk_field_object_cost": 0.0,
            "risk_field_headway_cost": 0.0,
            "risk_field_ttc_cost": 0.0,
            "risk_field_weighted_cost": 0.0,
            "risk_field_event_equivalent_cost": 0.0,
            "risk_field_collision_equivalent_cost": collision_equivalent_cost,
            "risk_field_cost_transform": self.config.get("risk_field_cost_transform", "event_squash"),
            "risk_field_cost_combine": self.config.get("risk_field_cost_combine", "max"),
        }

    def cost_function(self, vehicle_id: str):
        event_cost, step_info = super(SafeMetaDriveEnv_mini, self).cost_function(vehicle_id)
        event_cost = float(event_cost)
        step_info["event_cost"] = event_cost

        if not self.config.get("use_risk_field_cost", False):
            step_info.update(self._zero_risk_field_info())
            return event_cost, step_info

        vehicle = self.agents[vehicle_id]
        risk_cost, risk_info = self._get_risk_field_calculator().calculate(self, vehicle)
        weighted_risk_cost = self._risk_field_event_equivalent_cost(risk_cost)
        final_cost = self._combine_event_and_risk_cost(event_cost, weighted_risk_cost)
        self.episode_cost += final_cost - event_cost

        step_info.update(risk_info)
        step_info["risk_field_weighted_cost"] = weighted_risk_cost
        step_info["risk_field_event_equivalent_cost"] = weighted_risk_cost
        step_info["risk_field_collision_equivalent_cost"] = self._risk_field_collision_equivalent_cost()
        step_info["risk_field_cost_transform"] = self.config.get("risk_field_cost_transform", "event_squash")
        step_info["risk_field_cost_combine"] = self.config.get("risk_field_cost_combine", "max")
        step_info["cost"] = final_cost
        step_info["total_cost"] = self.episode_cost
        return final_cost, step_info

    @staticmethod
    def _has_severe_crash(info):
        return bool(
            info.get("crash_vehicle", False) or info.get("crash_object", False) or
            info.get("crash_building", False) or info.get("crash_human", False)
        )

    def _annotate_out_of_road_info(
        self,
        info,
        *,
        warning_triggered=False,
        timeout_terminated=False,
    ):
        info["out_of_road_mode"] = self.config.get("out_of_road_mode", "legacy")
        info["out_of_road_warning_count"] = int(self._out_of_road_warning_count)
        info["out_of_road_recovery_remaining"] = int(self._out_of_road_recovery_remaining)
        info["out_of_road_budget_exhausted"] = bool(self._out_of_road_budget_exhausted)
        info["out_of_road_warning_triggered"] = bool(warning_triggered)
        info["out_of_road_warning_active"] = bool(self._out_of_road_warning_active)
        info["out_of_road_timeout_terminated"] = bool(timeout_terminated)
        return info

    def reset(self, *args, **kwargs):
        obs, info = super(SafeMetaDriveEnv_mini, self).reset(*args, **kwargs)
        # SafeMetaDriveEnv内部会累计episode_cost，这里显式清零，确保debug信息按episode统计。
        self.episode_cost = 0
        self._reset_out_of_road_warning_state()
        info["total_cost"] = 0.0
        return obs, self._annotate_out_of_road_info(info)

    def step(self, actions):
        if not self._is_warning_budget_mode():
            obs, reward, terminated, truncated, info = super(SafeMetaDriveEnv_mini, self).step(actions)
            info = self._annotate_out_of_road_info(info)
            return obs, reward, terminated, truncated, info

        original_out_of_road_done = self.config["out_of_road_done"]
        self.config["out_of_road_done"] = False
        try:
            obs, reward, terminated, truncated, info = super(SafeMetaDriveEnv_mini, self).step(actions)
        finally:
            self.config["out_of_road_done"] = original_out_of_road_done

        warning_triggered = False
        timeout_terminated = False
        original_reward = float(reward)
        original_cost = float(info.get("cost", 0.0))
        new_cost = original_cost
        new_reward = original_reward

        raw_out_of_road = bool(info.get("out_of_road", False))
        warning_applicable = raw_out_of_road and (not self._has_severe_crash(info))

        if warning_applicable:
            if not self._out_of_road_warning_active:
                self._out_of_road_warning_active = True
                self._out_of_road_warning_count += 1
                self._out_of_road_recovery_remaining = int(self.config["out_of_road_recovery_steps"])
                warning_triggered = True

            self._out_of_road_budget_exhausted = (
                self._out_of_road_warning_count > int(self.config["out_of_road_warning_limit"])
            )

            if self._out_of_road_budget_exhausted and self.config["out_of_road_terminate_after_budget"]:
                terminated = True
                new_reward = -float(self.config["out_of_road_penalty"])
                new_cost = float(self.config["out_of_road_cost"])
            elif self._out_of_road_recovery_remaining <= 0:
                terminated = True
                timeout_terminated = True
                new_reward = -float(self.config["out_of_road_penalty"])
                new_cost = float(self.config["out_of_road_cost"])
            else:
                if warning_triggered and (self._out_of_road_warning_count <= int(self.config["out_of_road_warning_limit"])):
                    new_reward = float(info.get("step_reward", reward)) - float(
                        self.config["out_of_road_warning_penalty"]
                    )
                    new_cost = float(self.config["out_of_road_warning_cost"])
                else:
                    new_reward = float(info.get("step_reward", reward))
                    new_cost = 0.0
                self._out_of_road_recovery_remaining -= 1
        elif not raw_out_of_road:
            self._out_of_road_warning_active = False
            self._out_of_road_recovery_remaining = 0

        if new_cost != original_cost:
            self.episode_cost += new_cost - original_cost
            info["cost"] = new_cost
            info["total_cost"] = self.episode_cost

        if new_reward != original_reward:
            agent_id = next(iter(self.agents.keys()))
            self.episode_rewards[agent_id] += new_reward - original_reward
            info["episode_reward"] = self.episode_rewards[agent_id]

        reward = new_reward
        info = self._annotate_out_of_road_info(
            info,
            warning_triggered=warning_triggered,
            timeout_terminated=timeout_terminated,
        )
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # 创建训练环境,启用手控和渲染模式
    env = get_training_env({
        "manual_control": True,  # 启用手控模式
        "use_render": True,      # 启用渲染
    })
    env.reset()
    env.engine.toggle_help_message()  # 在渲染窗口中显示帮助信息
    
    # 主循环
    while True:
        _, _, tm, tc, _ = env.step([0, 0])  # 执行动作(这里使用空动作,等待手控输入)
        env.render(mode="topdown", target_agent_heading_up=True)  # 以俯视角度渲染,目标智能体朝上
        done = tm or tc  # 判断是否结束(truncated 或 terminated)
        if done:
            env.reset()  # 重置环境
