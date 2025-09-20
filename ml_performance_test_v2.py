#!/usr/bin/env python3
"""
机器学习算法服务器性能测试脚本
测试多种ML算法在不同数据规模下的性能表现
"""

import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class MLPerformanceTester:
    def __init__(self):
        self.results = []
        self.system_info = self.get_system_info()
        
    def get_system_info(self):
        """获取系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq(),
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3)  # GB
        }
    
    def measure_performance(self, func, *args, **kwargs):
        """测量函数执行性能"""
        # 记录开始状态
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        start_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 记录结束状态
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)  # MB
        end_cpu_percent = psutil.cpu_percent(interval=0.1)
        
        performance_data = {
            'execution_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'cpu_usage': (start_cpu_percent + end_cpu_percent) / 2,
            'peak_memory': end_memory
        }
        
        return result, performance_data
    
    def generate_classification_data(self, n_samples, n_features):
        """生成分类数据集"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features//3),
            n_redundant=max(1, n_features//4),
            n_classes=min(5, max(2, n_features//10)),
            random_state=42
        )
        return X, y
    
    def generate_regression_data(self, n_samples, n_features):
        """生成回归数据集"""
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=42
        )
        return X, y
    
    def generate_clustering_data(self, n_samples, n_features):
        """生成聚类数据集"""
        X, _ = make_blobs(
            n_samples=n_samples,
            centers=min(10, max(3, n_features//5)),
            n_features=n_features,
            random_state=42
        )
        return X
    
    def test_linear_regression(self, X, y):
        """测试线性回归"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        
        return {'algorithm': 'Linear Regression', 'score': score, 'type': 'regression'}
    
    def test_logistic_regression(self, X, y):
        """测试逻辑回归"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        score = accuracy_score(y_test, predictions)
        
        return {'algorithm': 'Logistic Regression', 'score': score, 'type': 'classification'}
    
    def test_random_forest_classifier(self, X, y):
        """测试随机森林分类器"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        
        return {'algorithm': 'Random Forest Classifier', 'score': score, 'type': 'classification'}
    
    def test_random_forest_regressor(self, X, y):
        """测试随机森林回归器"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        score = r2_score(y_test, predictions)
        
        return {'algorithm': 'Random Forest Regressor', 'score': score, 'type': 'regression'}
    
    def test_svm_classifier(self, X, y):
        """测试SVM分类器"""
        # 对于大数据集，限制样本数量以避免过长的训练时间
        if X.shape[0] > 5000:
            X_sample = X[:5000]
            y_sample = y[:5000]
        else:
            X_sample, y_sample = X, y
            
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        score = accuracy_score(y_test, predictions)
        
        return {'algorithm': 'SVM Classifier', 'score': score, 'type': 'classification'}
    
    def test_kmeans_clustering(self, X):
        """测试K-means聚类"""
        n_clusters = min(10, max(3, X.shape[1]//5))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        
        # 使用惯性作为评分
        score = -model.inertia_  # 负值，因为惯性越小越好
        
        return {'algorithm': 'K-Means Clustering', 'score': score, 'type': 'clustering'}
    
    def test_neural_network(self, X, y):
        """测试神经网络"""
        # 对于大数据集，限制样本数量
        if X.shape[0] > 10000:
            X_sample = X[:10000]
            y_sample = y[:10000]
        else:
            X_sample, y_sample = X, y
            
        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        score = accuracy_score(y_test, predictions)
        
        return {'algorithm': 'Neural Network', 'score': score, 'type': 'classification'}
    
    def run_comprehensive_test(self, data_sizes=None, feature_sizes=None):
        """运行综合性能测试"""
        if data_sizes is None:
            data_sizes = [1000, 5000, 10000, 25000]
        if feature_sizes is None:
            feature_sizes = [10, 50, 100]
            
        print("=" * 60)
        print("机器学习算法服务器性能测试")
        print("=" * 60)
        print(f"CPU核心数: {self.system_info['cpu_count']}")
        print(f"总内存: {self.system_info['memory_total']:.2f} GB")
        print(f"可用内存: {self.system_info['memory_available']:.2f} GB")
        print("=" * 60)
        
        total_tests = len(data_sizes) * len(feature_sizes) * 7  # 7个算法
        current_test = 0
        
        for n_samples in data_sizes:
            for n_features in feature_sizes:
                print(f"\n测试数据规模: {n_samples} 样本, {n_features} 特征")
                
                # 生成数据
                print("生成测试数据...")
                X_class, y_class = self.generate_classification_data(n_samples, n_features)
                X_reg, y_reg = self.generate_regression_data(n_samples, n_features)
                X_cluster = self.generate_clustering_data(n_samples, n_features)
                
                # 测试各种算法
                algorithms = [
                    ('Linear Regression', self.test_linear_regression, X_reg, y_reg),
                    ('Logistic Regression', self.test_logistic_regression, X_class, y_class),
                    ('Random Forest Classifier', self.test_random_forest_classifier, X_class, y_class),
                    ('Random Forest Regressor', self.test_random_forest_regressor, X_reg, y_reg),
                    ('SVM Classifier', self.test_svm_classifier, X_class, y_class),
                    ('K-Means Clustering', self.test_kmeans_clustering, X_cluster),
                    ('Neural Network', self.test_neural_network, X_class, y_class)
                ]
                
                for algo_name, algo_func, *args in algorithms:
                    current_test += 1
                    try:
                        print(f"  [{current_test}/{total_tests}] 测试 {algo_name}...")
                        
                        if len(args) == 1:  # 聚类算法只需要X
                            result, perf = self.measure_performance(algo_func, args[0])
                        else:  # 监督学习算法需要X和y
                            result, perf = self.measure_performance(algo_func, args[0], args[1])
                        
                        # 保存结果
                        self.results.append({
                            'algorithm': result['algorithm'],
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'execution_time': perf['execution_time'],
                            'memory_usage': perf['memory_usage'],
                            'cpu_usage': perf['cpu_usage'],
                            'peak_memory': perf['peak_memory'],
                            'score': result['score'],
                            'type': result['type']
                        })
                        
                        print(f"    完成 - 用时: {perf['execution_time']:.2f}s, 内存: {perf['memory_usage']:.2f}MB")
                        
                    except Exception as e:
                        print(f"    错误: {str(e)}")
                        continue
        
        print("\n测试完成！正在生成报告...")
        return self.generate_report()
    
    def generate_report(self):
        """生成性能测试报告"""
        if not self.results:
            print("没有测试结果可用于生成报告")
            return
        
        df = pd.DataFrame(self.results)
        
        # 创建可视化报告
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 执行时间对比
        ax1 = axes[0, 0]
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            ax1.plot(algo_data['n_samples'], algo_data['execution_time'], 
                    marker='o', label=algo, linewidth=2)
        ax1.set_xlabel('样本数量')
        ax1.set_ylabel('执行时间 (秒)')
        ax1.set_title('算法执行时间对比')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 2. 内存使用情况
        ax2 = axes[0, 1]
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            ax2.plot(algo_data['n_samples'], algo_data['memory_usage'], 
                    marker='s', label=algo, linewidth=2)
        ax2.set_xlabel('样本数量')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.set_title('算法内存使用对比')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # 3. CPU使用率
        ax3 = axes[1, 0]
        cpu_by_algo = df.groupby('algorithm')['cpu_usage'].mean().sort_values(ascending=False)
        bars = cpu_by_algo.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_xlabel('算法')
        ax3.set_ylabel('平均CPU使用率 (%)')
        ax3.set_title('算法CPU使用率对比')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. 性能效率 (执行时间 vs 数据规模)
        ax4 = axes[1, 1]
        for algo in df['algorithm'].unique()[:5]:  # 只显示前5个算法避免图表过于复杂
            algo_data = df[df['algorithm'] == algo]
            ax4.scatter(algo_data['n_samples'], algo_data['execution_time'], 
                       label=algo, alpha=0.7, s=60)
        ax4.set_xlabel('样本数量')
        ax4.set_ylabel('执行时间 (秒)')
        ax4.set_title('数据规模 vs 执行时间')
        ax4.legend()
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        try:
            plt.savefig('/home/work/hd/ml_performance_report.png', dpi=300, bbox_inches='tight')
            print(f"可视化报告已保存到: /home/work/hd/ml_performance_report.png")
        except Exception as e:
            print(f"保存图片时出错: {e}")
        
        # 生成统计报告
        print("\n" + "="*60)
        print("性能测试统计报告")
        print("="*60)
        
        print(f"\n系统信息:")
        print(f"  CPU核心数: {self.system_info['cpu_count']}")
        print(f"  总内存: {self.system_info['memory_total']:.2f} GB")
        print(f"  可用内存: {self.system_info['memory_available']:.2f} GB")
        
        print("\n1. 算法执行时间统计 (秒):")
        time_stats = df.groupby('algorithm')['execution_time'].agg(['mean', 'std', 'min', 'max'])
        print(time_stats.round(3))
        
        print("\n2. 算法内存使用统计 (MB):")
        memory_stats = df.groupby('algorithm')['memory_usage'].agg(['mean', 'std', 'min', 'max'])
        print(memory_stats.round(2))
        
        print("\n3. 算法CPU使用率统计 (%):")
        cpu_stats = df.groupby('algorithm')['cpu_usage'].agg(['mean', 'std', 'min', 'max'])
        print(cpu_stats.round(2))
        
        print("\n4. 最快的算法 (按平均执行时间):")
        fastest = time_stats['mean'].sort_values().head(3)
        for i, (algo, time_val) in enumerate(fastest.items(), 1):
            print(f"  {i}. {algo}: {time_val:.3f}秒")
        
        print("\n5. 最省内存的算法 (按平均内存使用):")
        most_efficient = memory_stats['mean'].sort_values().head(3)
        for i, (algo, mem_val) in enumerate(most_efficient.items(), 1):
            print(f"  {i}. {algo}: {mem_val:.2f}MB")
        
        print("\n6. 数据规模影响分析:")
        large_data = df[df['n_samples'] >= 10000]
        small_data = df[df['n_samples'] < 10000]
        
        if not large_data.empty and not small_data.empty:
            print(f"  大数据集平均执行时间: {large_data['execution_time'].mean():.2f}秒")
            print(f"  小数据集平均执行时间: {small_data['execution_time'].mean():.2f}秒")
            print(f"  执行时间增长倍数: {large_data['execution_time'].mean() / small_data['execution_time'].mean():.2f}x")
        
        # 保存详细结果到CSV
        try:
            df.to_csv('/home/work/hd/ml_performance_results.csv', index=False)
            print(f"\n详细结果已保存到: /home/work/hd/ml_performance_results.csv")
        except Exception as e:
            print(f"保存CSV文件时出错: {e}")
        
        print("\n" + "="*60)
        print("测试总结:")
        print(f"  总测试数量: {len(df)}")
        print(f"  测试算法数量: {df['algorithm'].nunique()}")
        print(f"  总执行时间: {df['execution_time'].sum():.2f}秒")
        print(f"  总内存使用: {df['memory_usage'].sum():.2f}MB")
        print("="*60)
        
        return df

def run_quick_test():
    """运行快速测试版本"""
    print("运行快速性能测试...")
    tester = MLPerformanceTester()
    
    # 较小的数据规模用于快速测试
    data_sizes = [1000, 5000]
    feature_sizes = [10, 50]
    
    return tester.run_comprehensive_test(data_sizes, feature_sizes)

def run_full_test():
    """运行完整测试版本"""
    print("运行完整性能测试...")
    tester = MLPerformanceTester()
    
    # 完整的数据规模
    data_sizes = [1000, 5000, 10000, 25000]
    feature_sizes = [10, 50, 100]
    
    return tester.run_comprehensive_test(data_sizes, feature_sizes)

if __name__ == "__main__":
    print("机器学习性能测试工具")
    print("1. 快速测试 (推荐)")
    print("2. 完整测试 (耗时较长)")
    
    # 默认运行快速测试
    choice = input("\n请选择测试类型 (1/2, 默认为1): ").strip()
    
    if choice == "2":
        results = run_full_test()
    else:
        results = run_quick_test()
    
    print("\n测试完成！检查生成的报告文件。")