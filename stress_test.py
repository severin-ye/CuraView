#!/usr/bin/env python3
"""
高强度机器学习压力测试脚本
专门用于测试服务器在高负载下的性能表现
"""

import time
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class StressTestRunner:
    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024**3)  # GB
        
    def get_system_status(self):
        """获取当前系统状态"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used': psutil.virtual_memory().used / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3),  # GB
        }
    
    def cpu_intensive_task(self, task_id, duration=30):
        """CPU密集型任务 - 矩阵计算"""
        print(f"启动CPU密集型任务 {task_id}")
        start_time = time.time()
        
        results = []
        while time.time() - start_time < duration:
            # 大规模矩阵乘法
            size = 500
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            C = np.dot(A, B)
            
            # 特征值分解
            eigenvals, eigenvecs = np.linalg.eig(C[:100, :100])
            
            # SVD分解
            U, s, Vh = np.linalg.svd(C[:200, :200])
            
            results.append({
                'task_id': task_id,
                'matrix_sum': np.sum(C),
                'eigenvals_sum': np.sum(eigenvals.real),
                'svd_sum': np.sum(s)
            })
        
        return {
            'task_id': task_id,
            'duration': time.time() - start_time,
            'iterations': len(results),
            'avg_iterations_per_sec': len(results) / (time.time() - start_time)
        }
    
    def memory_intensive_task(self, task_id, duration=30):
        """内存密集型任务 - 大数据处理"""
        print(f"启动内存密集型任务 {task_id}")
        start_time = time.time()
        
        results = []
        large_arrays = []
        
        while time.time() - start_time < duration:
            # 创建大型数组
            size = 50000
            data = np.random.rand(size, 100)
            large_arrays.append(data)
            
            # 数据处理操作
            mean_vals = np.mean(data, axis=0)
            std_vals = np.std(data, axis=0)
            corr_matrix = np.corrcoef(data.T)
            
            results.append({
                'task_id': task_id,
                'data_shape': data.shape,
                'mean_sum': np.sum(mean_vals),
                'std_sum': np.sum(std_vals),
                'corr_trace': np.trace(corr_matrix)
            })
            
            # 控制内存使用，定期清理
            if len(large_arrays) > 10:
                large_arrays = large_arrays[-5:]
        
        return {
            'task_id': task_id,
            'duration': time.time() - start_time,
            'iterations': len(results),
            'peak_arrays': len(large_arrays),
            'avg_iterations_per_sec': len(results) / (time.time() - start_time)
        }
    
    def ml_training_task(self, task_id, duration=30):
        """机器学习训练任务"""
        print(f"启动ML训练任务 {task_id}")
        start_time = time.time()
        
        results = []
        
        while time.time() - start_time < duration:
            # 生成数据
            n_samples = np.random.randint(5000, 15000)
            n_features = np.random.randint(20, 100)
            
            X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                     n_informative=max(5, n_features//3), 
                                     random_state=np.random.randint(1000))
            
            # 训练随机森林
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            train_start = time.time()
            rf.fit(X, y)
            train_time = time.time() - train_start
            
            # 预测
            pred_start = time.time()
            predictions = rf.predict(X[:1000])
            pred_time = time.time() - pred_start
            
            results.append({
                'task_id': task_id,
                'n_samples': n_samples,
                'n_features': n_features,
                'train_time': train_time,
                'pred_time': pred_time,
                'accuracy': rf.score(X[:1000], y[:1000])
            })
        
        return {
            'task_id': task_id,
            'duration': time.time() - start_time,
            'models_trained': len(results),
            'avg_train_time': np.mean([r['train_time'] for r in results]),
            'avg_accuracy': np.mean([r['accuracy'] for r in results])
        }
    
    def mixed_workload_task(self, task_id, duration=30):
        """混合工作负载任务"""
        print(f"启动混合工作负载任务 {task_id}")
        start_time = time.time()
        
        results = []
        
        while time.time() - start_time < duration:
            # 随机选择任务类型
            task_type = np.random.choice(['cpu', 'memory', 'ml'])
            
            if task_type == 'cpu':
                # CPU计算
                size = 200
                A = np.random.rand(size, size)
                B = np.random.rand(size, size)
                C = np.dot(A, B)
                result_val = np.sum(C)
                
            elif task_type == 'memory':
                # 内存操作
                data = np.random.rand(10000, 50)
                processed = np.sort(data, axis=0)
                result_val = np.sum(processed)
                
            else:  # ml
                # ML训练
                X, y = make_regression(n_samples=2000, n_features=20, random_state=42)
                model = LinearRegression()
                model.fit(X, y)
                result_val = model.score(X, y)
            
            results.append({
                'task_id': task_id,
                'task_type': task_type,
                'result': result_val
            })
        
        return {
            'task_id': task_id,
            'duration': time.time() - start_time,
            'operations': len(results),
            'ops_per_sec': len(results) / (time.time() - start_time),
            'task_distribution': pd.Series([r['task_type'] for r in results]).value_counts().to_dict()
        }
    
    def run_stress_test(self, test_duration=60, max_workers=None):
        """运行压力测试"""
        if max_workers is None:
            max_workers = self.cpu_count
        
        print("=" * 60)
        print("机器学习高强度压力测试")
        print("=" * 60)
        print(f"CPU核心数: {self.cpu_count}")
        print(f"总内存: {self.memory_total:.2f} GB")
        print(f"测试时长: {test_duration} 秒")
        print(f"并发任务数: {max_workers}")
        print("=" * 60)
        
        # 记录测试开始状态
        initial_status = self.get_system_status()
        print(f"测试开始时系统状态:")
        print(f"  CPU使用率: {initial_status['cpu_percent']:.1f}%")
        print(f"  内存使用率: {initial_status['memory_percent']:.1f}%")
        print(f"  可用内存: {initial_status['memory_available']:.2f} GB")
        
        # 创建不同类型的任务
        tasks = []
        
        # CPU密集型任务 (使用部分CPU)
        cpu_workers = max(1, max_workers // 4)
        for i in range(cpu_workers):
            tasks.append(('cpu', i, test_duration))
        
        # 内存密集型任务
        memory_workers = max(1, max_workers // 4)
        for i in range(memory_workers):
            tasks.append(('memory', i, test_duration))
        
        # ML训练任务
        ml_workers = max(1, max_workers // 4)
        for i in range(ml_workers):
            tasks.append(('ml', i, test_duration))
        
        # 混合工作负载任务
        mixed_workers = max_workers - cpu_workers - memory_workers - ml_workers
        for i in range(mixed_workers):
            tasks.append(('mixed', i, test_duration))
        
        print(f"\n任务分配:")
        print(f"  CPU密集型任务: {cpu_workers}")
        print(f"  内存密集型任务: {memory_workers}")
        print(f"  ML训练任务: {ml_workers}")
        print(f"  混合工作负载任务: {mixed_workers}")
        print(f"  总任务数: {len(tasks)}")
        
        # 启动监控线程
        monitor_results = []
        
        def monitor_system():
            while True:
                status = self.get_system_status()
                status['timestamp'] = time.time()
                monitor_results.append(status)
                time.sleep(2)
                if len(monitor_results) > test_duration // 2:
                    break
        
        # 开始压力测试
        print(f"\n开始压力测试... (预计用时: {test_duration}秒)")
        start_time = time.time()
        
        # 启动监控
        import threading
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
        
        # 执行任务
        task_results = []
        
        def run_task(task_info):
            task_type, task_id, duration = task_info
            if task_type == 'cpu':
                return self.cpu_intensive_task(task_id, duration)
            elif task_type == 'memory':
                return self.memory_intensive_task(task_id, duration)
            elif task_type == 'ml':
                return self.ml_training_task(task_id, duration)
            else:  # mixed
                return self.mixed_workload_task(task_id, duration)
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_task, task) for task in tasks]
            
            for future in futures:
                try:
                    result = future.result(timeout=test_duration + 30)
                    task_results.append(result)
                except Exception as e:
                    print(f"任务执行错误: {e}")
        
        total_time = time.time() - start_time
        
        # 分析结果
        print(f"\n压力测试完成! 实际用时: {total_time:.2f}秒")
        self.analyze_stress_test_results(task_results, monitor_results, initial_status)
        
        return task_results, monitor_results
    
    def analyze_stress_test_results(self, task_results, monitor_results, initial_status):
        """分析压力测试结果"""
        print("\n" + "="*60)
        print("压力测试结果分析")
        print("="*60)
        
        # 最终系统状态
        final_status = self.get_system_status()
        print(f"\n1. 系统状态对比:")
        print(f"  测试前CPU使用率: {initial_status['cpu_percent']:.1f}%")
        print(f"  测试后CPU使用率: {final_status['cpu_percent']:.1f}%")
        print(f"  测试前内存使用率: {initial_status['memory_percent']:.1f}%")
        print(f"  测试后内存使用率: {final_status['memory_percent']:.1f}%")
        
        # 任务执行统计
        if task_results:
            print(f"\n2. 任务执行统计:")
            print(f"  完成任务数: {len(task_results)}")
            
            # 按任务类型统计
            cpu_tasks = [r for r in task_results if 'iterations' in r]
            memory_tasks = [r for r in task_results if 'peak_arrays' in r]
            ml_tasks = [r for r in task_results if 'models_trained' in r]
            mixed_tasks = [r for r in task_results if 'ops_per_sec' in r]
            
            if cpu_tasks:
                avg_cpu_perf = np.mean([r['avg_iterations_per_sec'] for r in cpu_tasks])
                print(f"  CPU任务平均性能: {avg_cpu_perf:.2f} 迭代/秒")
            
            if memory_tasks:
                avg_mem_perf = np.mean([r['avg_iterations_per_sec'] for r in memory_tasks])
                print(f"  内存任务平均性能: {avg_mem_perf:.2f} 迭代/秒")
            
            if ml_tasks:
                total_models = sum([r['models_trained'] for r in ml_tasks])
                avg_accuracy = np.mean([r['avg_accuracy'] for r in ml_tasks])
                print(f"  ML任务训练模型总数: {total_models}")
                print(f"  ML任务平均准确率: {avg_accuracy:.3f}")
            
            if mixed_tasks:
                avg_mixed_perf = np.mean([r['ops_per_sec'] for r in mixed_tasks])
                print(f"  混合任务平均性能: {avg_mixed_perf:.2f} 操作/秒")
        
        # 系统监控数据分析
        if monitor_results:
            print(f"\n3. 系统性能监控:")
            df_monitor = pd.DataFrame(monitor_results)
            
            print(f"  监控数据点数: {len(df_monitor)}")
            print(f"  平均CPU使用率: {df_monitor['cpu_percent'].mean():.1f}%")
            print(f"  峰值CPU使用率: {df_monitor['cpu_percent'].max():.1f}%")
            print(f"  平均内存使用率: {df_monitor['memory_percent'].mean():.1f}%")
            print(f"  峰值内存使用率: {df_monitor['memory_percent'].max():.1f}%")
            print(f"  最小可用内存: {df_monitor['memory_available'].min():.2f} GB")
        
        # 性能评级
        print(f"\n4. 服务器性能评级:")
        
        # 基于CPU使用率评级
        if monitor_results:
            avg_cpu = df_monitor['cpu_percent'].mean()
            if avg_cpu < 50:
                cpu_grade = "优秀"
            elif avg_cpu < 70:
                cpu_grade = "良好"
            elif avg_cpu < 85:
                cpu_grade = "一般"
            else:
                cpu_grade = "需要优化"
            
            print(f"  CPU性能: {cpu_grade} (平均使用率: {avg_cpu:.1f}%)")
            
            # 基于内存使用率评级
            avg_memory = df_monitor['memory_percent'].mean()
            if avg_memory < 60:
                memory_grade = "优秀"
            elif avg_memory < 75:
                memory_grade = "良好"
            elif avg_memory < 90:
                memory_grade = "一般"
            else:
                memory_grade = "需要优化"
            
            print(f"  内存性能: {memory_grade} (平均使用率: {avg_memory:.1f}%)")
        
        # 保存结果
        try:
            if task_results:
                pd.DataFrame(task_results).to_csv('/home/work/hd/stress_test_tasks.csv', index=False)
            if monitor_results:
                pd.DataFrame(monitor_results).to_csv('/home/work/hd/stress_test_monitoring.csv', index=False)
            print(f"\n结果已保存到:")
            print(f"  任务结果: /home/work/hd/stress_test_tasks.csv")
            print(f"  监控数据: /home/work/hd/stress_test_monitoring.csv")
        except Exception as e:
            print(f"保存结果时出错: {e}")

def main():
    tester = StressTestRunner()
    
    print("机器学习压力测试工具")
    print("注意: 此测试会对系统造成高负载，请确保系统稳定")
    
    # 获取测试参数
    duration = input("请输入测试时长(秒，默认60): ").strip()
    duration = int(duration) if duration.isdigit() else 60
    
    workers = input(f"请输入并发任务数(默认{tester.cpu_count}): ").strip()
    workers = int(workers) if workers.isdigit() else None
    
    confirm = input(f"即将开始{duration}秒的高强度压力测试，是否继续? (y/N): ").strip().lower()
    
    if confirm == 'y':
        task_results, monitor_results = tester.run_stress_test(duration, workers)
        print("\n压力测试完成！")
    else:
        print("测试已取消。")

if __name__ == "__main__":
    main()