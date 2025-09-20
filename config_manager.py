#!/usr/bin/env python3
"""
配置文件读取器
用于管理项目的各种配置参数
"""

import configparser
import os
from typing import List, Dict, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.ini"):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            self.config.read(self.config_file)
        else:
            print(f"配置文件 {self.config_file} 不存在，使用默认配置")
    
    def get_test_config(self) -> Dict[str, Any]:
        """获取测试配置"""
        section = self.config['test_config'] if 'test_config' in self.config else {}
        
        quick_samples = section.get('quick_test_samples', '1000,5000')
        quick_features = section.get('quick_test_features', '10,50')
        full_samples = section.get('full_test_samples', '1000,5000,10000,25000')
        full_features = section.get('full_test_features', '10,50,100')
        
        return {
            'quick_test_samples': [int(x.strip()) for x in quick_samples.split(',')],
            'quick_test_features': [int(x.strip()) for x in quick_features.split(',')],
            'full_test_samples': [int(x.strip()) for x in full_samples.split(',')],
            'full_test_features': [int(x.strip()) for x in full_features.split(',')]
        }
    
    def get_performance_thresholds(self) -> Dict[str, float]:
        """获取性能阈值配置"""
        section = self.config['performance_thresholds'] if 'performance_thresholds' in self.config else {}
        
        return {
            'excellent_time_threshold': float(section.get('excellent_time_threshold', '0.5')),
            'good_time_threshold': float(section.get('good_time_threshold', '1.0')),
            'fair_time_threshold': float(section.get('fair_time_threshold', '2.0')),
            'excellent_memory_threshold': float(section.get('excellent_memory_threshold', '5.0')),
            'good_memory_threshold': float(section.get('good_memory_threshold', '15.0')),
            'fair_memory_threshold': float(section.get('fair_memory_threshold', '30.0')),
            'excellent_cpu_threshold': float(section.get('excellent_cpu_threshold', '30.0')),
            'good_cpu_threshold': float(section.get('good_cpu_threshold', '60.0')),
            'high_cpu_threshold': float(section.get('high_cpu_threshold', '80.0'))
        }
    
    def get_stress_test_config(self) -> Dict[str, Any]:
        """获取压力测试配置"""
        section = self.config['stress_test_config'] if 'stress_test_config' in self.config else {}
        
        return {
            'default_duration': int(section.get('default_duration', '60')),
            'cpu_task_ratio': float(section.get('cpu_task_ratio', '0.25')),
            'memory_task_ratio': float(section.get('memory_task_ratio', '0.25')),
            'ml_task_ratio': float(section.get('ml_task_ratio', '0.25')),
            'mixed_task_ratio': float(section.get('mixed_task_ratio', '0.25'))
        }
    
    def get_output_config(self) -> Dict[str, str]:
        """获取输出文件配置"""
        section = self.config['output_config'] if 'output_config' in self.config else {}
        
        return {
            'results_csv': section.get('results_csv', 'ml_performance_results.csv'),
            'report_png': section.get('report_png', 'ml_performance_report.png'),
            'stress_tasks_csv': section.get('stress_tasks_csv', 'stress_test_tasks.csv'),
            'stress_monitoring_csv': section.get('stress_monitoring_csv', 'stress_test_monitoring.csv')
        }
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """获取算法配置"""
        section = self.config['algorithms'] if 'algorithms' in self.config else {}
        
        enabled_algorithms = section.get('enabled_algorithms', 
            'Linear Regression,Logistic Regression,Random Forest Classifier,Random Forest Regressor,SVM Classifier,K-Means Clustering,Neural Network')
        
        return {
            'enabled_algorithms': [x.strip() for x in enabled_algorithms.split(',')],
            'svm_sample_limit': int(section.get('svm_sample_limit', '5000')),
            'neural_network_sample_limit': int(section.get('neural_network_sample_limit', '10000'))
        }
    
    def get_system_requirements(self) -> Dict[str, Any]:
        """获取系统要求配置"""
        section = self.config['system_requirements'] if 'system_requirements' in self.config else {}
        
        return {
            'min_python_version': section.get('min_python_version', '3.8'),
            'min_memory_gb': int(section.get('min_memory_gb', '4')),
            'recommended_cpu_cores': int(section.get('recommended_cpu_cores', '4'))
        }
    
    def performance_grade(self, value: float, metric_type: str) -> str:
        """根据配置的阈值给出性能等级"""
        thresholds = self.get_performance_thresholds()
        
        if metric_type == 'time':
            if value < thresholds['excellent_time_threshold']:
                return "优秀"
            elif value < thresholds['good_time_threshold']:
                return "良好"
            elif value < thresholds['fair_time_threshold']:
                return "一般"
            else:
                return "需要优化"
        
        elif metric_type == 'memory':
            if value < thresholds['excellent_memory_threshold']:
                return "优秀"
            elif value < thresholds['good_memory_threshold']:
                return "良好"
            elif value < thresholds['fair_memory_threshold']:
                return "一般"
            else:
                return "需要优化"
        
        elif metric_type == 'cpu':
            if value < thresholds['excellent_cpu_threshold']:
                return "利用率偏低"
            elif value < thresholds['good_cpu_threshold']:
                return "利用率良好"
            elif value < thresholds['high_cpu_threshold']:
                return "利用率较高"
            else:
                return "利用率很高"
        
        return "未知"

# 全局配置实例
config_manager = ConfigManager()

if __name__ == "__main__":
    # 测试配置读取
    print("测试配置管理器...")
    
    test_config = config_manager.get_test_config()
    print(f"快速测试样本数: {test_config['quick_test_samples']}")
    print(f"完整测试特征数: {test_config['full_test_features']}")
    
    thresholds = config_manager.get_performance_thresholds()
    print(f"时间阈值: {thresholds['excellent_time_threshold']}")
    
    print(f"时间性能等级 (0.3秒): {config_manager.performance_grade(0.3, 'time')}")
    print(f"内存性能等级 (10MB): {config_manager.performance_grade(10, 'memory')}")
    print(f"CPU性能等级 (45%): {config_manager.performance_grade(45, 'cpu')}")