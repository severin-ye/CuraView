#!/usr/bin/env python3
"""
机器学习性能测试结果查看器
"""

import pandas as pd
import numpy as np

def analyze_results():
    """分析测试结果"""
    try:
        # 读取结果文件
        df = pd.read_csv('/home/work/hd/ml_performance_results.csv')
        
        print("=" * 60)
        print("机器学习性能测试结果分析")
        print("=" * 60)
        
        print(f"总测试数量: {len(df)}")
        print(f"测试算法: {', '.join(df['algorithm'].unique())}")
        print(f"数据规模: {df['n_samples'].unique()}")
        print(f"特征数量: {df['n_features'].unique()}")
        
        print("\n1. 算法性能排名 (按执行时间)")
        print("-" * 40)
        time_ranking = df.groupby('algorithm')['execution_time'].mean().sort_values()
        for i, (algo, time_val) in enumerate(time_ranking.items(), 1):
            print(f"{i:2d}. {algo:<25} {time_val:.3f}秒")
        
        print("\n2. 内存使用效率排名")
        print("-" * 40)
        memory_ranking = df.groupby('algorithm')['memory_usage'].mean().sort_values()
        for i, (algo, mem_val) in enumerate(memory_ranking.items(), 1):
            print(f"{i:2d}. {algo:<25} {mem_val:.2f}MB")
        
        print("\n3. 数据规模扩展性分析")
        print("-" * 40)
        for algo in df['algorithm'].unique():
            algo_data = df[df['algorithm'] == algo]
            if len(algo_data) > 1:
                small_data = algo_data[algo_data['n_samples'] == algo_data['n_samples'].min()]
                large_data = algo_data[algo_data['n_samples'] == algo_data['n_samples'].max()]
                
                if not small_data.empty and not large_data.empty:
                    scale_factor = large_data['n_samples'].iloc[0] / small_data['n_samples'].iloc[0]
                    time_factor = large_data['execution_time'].mean() / small_data['execution_time'].mean()
                    
                    print(f"{algo:<25} 数据规模{scale_factor:.1f}x -> 时间{time_factor:.1f}x")
        
        print("\n4. 最佳算法推荐")
        print("-" * 40)
        print(f"最快算法: {time_ranking.index[0]} ({time_ranking.iloc[0]:.3f}秒)")
        print(f"最省内存: {memory_ranking.index[0]} ({memory_ranking.iloc[0]:.2f}MB)")
        
        # 综合性能评分 (时间权重0.6, 内存权重0.4)
        normalized_time = (df.groupby('algorithm')['execution_time'].mean() - df.groupby('algorithm')['execution_time'].mean().min()) / (df.groupby('algorithm')['execution_time'].mean().max() - df.groupby('algorithm')['execution_time'].mean().min())
        normalized_memory = (df.groupby('algorithm')['memory_usage'].mean() - df.groupby('algorithm')['memory_usage'].mean().min()) / (df.groupby('algorithm')['memory_usage'].mean().max() - df.groupby('algorithm')['memory_usage'].mean().min())
        
        # 处理可能的负值
        normalized_memory = np.maximum(normalized_memory, 0)
        
        composite_score = 0.6 * normalized_time + 0.4 * normalized_memory
        best_overall = composite_score.sort_values().index[0]
        
        print(f"综合最佳: {best_overall}")
        
        print("\n5. 服务器性能评估")
        print("-" * 40)
        avg_time = df['execution_time'].mean()
        avg_memory = df['memory_usage'].mean()
        
        if avg_time < 0.5:
            time_grade = "优秀"
        elif avg_time < 1.0:
            time_grade = "良好"
        elif avg_time < 2.0:
            time_grade = "一般"
        else:
            time_grade = "需要优化"
        
        if avg_memory < 5:
            memory_grade = "优秀"
        elif avg_memory < 15:
            memory_grade = "良好"
        elif avg_memory < 30:
            memory_grade = "一般"
        else:
            memory_grade = "需要优化"
        
        print(f"执行速度: {time_grade} (平均 {avg_time:.2f}秒)")
        print(f"内存效率: {memory_grade} (平均 {avg_memory:.2f}MB)")
        
        # CPU使用率分析
        avg_cpu = df['cpu_usage'].mean()
        if avg_cpu < 30:
            cpu_grade = "利用率偏低"
        elif avg_cpu < 60:
            cpu_grade = "利用率良好"
        elif avg_cpu < 80:
            cpu_grade = "利用率较高"
        else:
            cpu_grade = "利用率很高"
        
        print(f"CPU利用率: {cpu_grade} (平均 {avg_cpu:.1f}%)")
        
        return df
        
    except FileNotFoundError:
        print("错误: 找不到测试结果文件 ml_performance_results.csv")
        print("请先运行 ml_performance_test_v2.py")
        return None
    except Exception as e:
        print(f"分析结果时出错: {e}")
        return None

def show_detailed_results():
    """显示详细结果"""
    try:
        df = pd.read_csv('/home/work/hd/ml_performance_results.csv')
        
        print("\n详细测试结果:")
        print("=" * 80)
        
        for algo in sorted(df['algorithm'].unique()):
            print(f"\n{algo}:")
            algo_data = df[df['algorithm'] == algo]
            
            for _, row in algo_data.iterrows():
                print(f"  样本:{row['n_samples']:5d} 特征:{row['n_features']:3d} "
                      f"时间:{row['execution_time']:6.3f}s "
                      f"内存:{row['memory_usage']:6.2f}MB "
                      f"CPU:{row['cpu_usage']:5.1f}% "
                      f"得分:{row['score']:8.3f}")
        
        return df
        
    except Exception as e:
        print(f"显示详细结果时出错: {e}")
        return None

if __name__ == "__main__":
    print("机器学习性能测试结果分析工具")
    print("1. 基本分析")
    print("2. 详细结果")
    
    choice = input("\n请选择 (1/2, 默认1): ").strip()
    
    if choice == "2":
        show_detailed_results()
    else:
        analyze_results()