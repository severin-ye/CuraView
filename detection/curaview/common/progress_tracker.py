"""
进度跟踪和时间统计模块

提供详细的步骤时间戳记录、预期时间计算和进度条显示
"""

import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import threading

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠️  tqdm未安装，将使用简单进度显示。安装: pip install tqdm")


@dataclass
class StepTiming:
    """单个步骤的时间记录"""
    step_name: str
    step_index: int
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    status: str = "running"  # running, completed, failed

    def complete(self, status: str = "completed"):
        """标记步骤完成"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status

    def to_dict(self):
        """转换为字典"""
        return {
            "step_name": self.step_name,
            "step_index": self.step_index,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration_seconds": round(self.duration, 2) if self.duration else None,
            "status": self.status
        }


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self,
                 task_name: str,
                 total_steps: int,
                 step_names: List[str],
                 stats_file: Optional[str] = None,
                 show_progress_bar: bool = True):
        """
        初始化进度跟踪器

        Args:
            task_name: 任务名称（如"patient_2_generation"）
            total_steps: 总步骤数
            step_names: 步骤名称列表
            stats_file: 历史统计文件路径
            show_progress_bar: 是否显示进度条
        """
        self.task_name = task_name
        self.total_steps = total_steps
        self.step_names = step_names
        self.stats_file = stats_file
        self.show_progress_bar = show_progress_bar and TQDM_AVAILABLE

        # 当前任务的步骤记录
        self.step_timings: List[StepTiming] = []
        self.current_step_index = 0
        self.task_start_time = time.time()

        # 加载历史统计数据
        self.historical_stats = self._load_historical_stats()

        # 进度条
        self.pbar = None
        if self.show_progress_bar:
            self.pbar = tqdm(
                total=total_steps,
                desc=f"🚀 {task_name}",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]",
                ncols=100
            )

        # 线程锁（如果并发使用）
        self.lock = threading.Lock()

    def _load_historical_stats(self) -> Dict:
        """加载历史统计数据"""
        if not self.stats_file or not os.path.exists(self.stats_file):
            return defaultdict(lambda: {"count": 0, "total_time": 0, "avg_time": 0})

        try:
            with open(self.stats_file, 'r') as f:
                data = json.load(f)
                return defaultdict(lambda: {"count": 0, "total_time": 0, "avg_time": 0}, data)
        except:
            return defaultdict(lambda: {"count": 0, "total_time": 0, "avg_time": 0})

    def _save_historical_stats(self):
        """保存历史统计数据"""
        if not self.stats_file:
            return

        # 确保目录存在
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        try:
            with open(self.stats_file, 'w') as f:
                json.dump(dict(self.historical_stats), f, indent=2)
        except Exception as e:
            print(f"⚠️  保存统计数据失败: {e}")

    def get_expected_time(self, step_name: str) -> float:
        """获取步骤的预期耗时（秒）"""
        stats = self.historical_stats.get(step_name, {"avg_time": 0})
        return stats.get("avg_time", 0)

    def get_remaining_time(self) -> float:
        """计算预期剩余时间（秒）"""
        remaining_steps = self.step_names[self.current_step_index:]
        return sum(self.get_expected_time(step) for step in remaining_steps)

    def get_expected_completion_time(self) -> str:
        """获取预期完成时间（格式化字符串）"""
        remaining_seconds = self.get_remaining_time()
        if remaining_seconds == 0:
            return "未知"

        completion_time = datetime.now() + timedelta(seconds=remaining_seconds)
        return completion_time.strftime("%H:%M:%S")

    def start_step(self, step_name: str, step_index: Optional[int] = None):
        """开始一个步骤"""
        with self.lock:
            if step_index is None:
                step_index = self.current_step_index

            timing = StepTiming(
                step_name=step_name,
                step_index=step_index,
                start_time=time.time()
            )
            self.step_timings.append(timing)

            # 显示步骤信息
            expected_time = self.get_expected_time(step_name)
            expected_completion = self.get_expected_completion_time()

            if self.pbar:
                self.pbar.set_description(
                    f"🚀 {self.task_name} | Step {step_index+1}/{self.total_steps}: {step_name}"
                )
                self.pbar.set_postfix_str(f"预计完成: {expected_completion}")
            else:
                # 简单文本显示
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step {step_index+1}/{self.total_steps}: {step_name}")
                if expected_time > 0:
                    print(f"  预计耗时: {expected_time:.1f}秒 | 预计完成: {expected_completion}")

            return timing

    def complete_step(self, status: str = "completed"):
        """完成当前步骤"""
        with self.lock:
            if not self.step_timings:
                return

            timing = self.step_timings[-1]
            timing.complete(status)

            # 更新历史统计
            step_name = timing.step_name
            self.historical_stats[step_name]["count"] += 1
            self.historical_stats[step_name]["total_time"] += timing.duration
            self.historical_stats[step_name]["avg_time"] = (
                self.historical_stats[step_name]["total_time"] /
                self.historical_stats[step_name]["count"]
            )

            # 更新进度条
            if self.pbar:
                self.pbar.update(1)
            else:
                print(f"  ✓ 完成 (耗时: {timing.duration:.2f}秒)")

            self.current_step_index += 1

    def finish(self):
        """完成整个任务"""
        with self.lock:
            total_duration = time.time() - self.task_start_time

            # 关闭进度条
            if self.pbar:
                self.pbar.close()

            # 保存统计数据
            self._save_historical_stats()

            # 显示总结
            print(f"\n{'='*80}")
            print(f"✅ 任务完成: {self.task_name}")
            print(f"总耗时: {total_duration:.2f}秒 ({timedelta(seconds=int(total_duration))})")
            print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

    def get_summary(self) -> Dict:
        """获取任务摘要"""
        total_duration = time.time() - self.task_start_time

        return {
            "task_name": self.task_name,
            "total_steps": self.total_steps,
            "completed_steps": self.current_step_index,
            "total_duration_seconds": round(total_duration, 2),
            "start_time": datetime.fromtimestamp(self.task_start_time).isoformat(),
            "end_time": datetime.now().isoformat(),
            "steps": [timing.to_dict() for timing in self.step_timings]
        }

    def save_summary(self, output_file: str):
        """保存任务摘要到文件"""
        summary = self.get_summary()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"📊 任务摘要已保存: {output_file}")


class SimpleProgressDisplay:
    """简单的进度显示（当tqdm不可用时）"""

    def __init__(self, total: int, desc: str = ""):
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time

        # 简单的进度条
        bar_length = 40
        filled_length = int(bar_length * self.current / self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        print(f"\r{self.desc}: |{bar}| {percentage:.1f}% ({self.current}/{self.total}) {elapsed:.1f}s", end='')

        if self.current >= self.total:
            print()  # 换行

    def close(self):
        """关闭进度显示"""
        print()


# 使用示例
