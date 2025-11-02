#!/usr/bin/env python3
"""
重试API失败的任务
读取api_failures.json，重新执行失败的API调用
"""

from google import genai
from google.genai import types
from pathlib import Path
import json
import argparse
import time
from tqdm import tqdm


class FailedTaskRetry:
    def __init__(self, failures_file):
        """
        初始化失败任务重试器
        
        Args:
            failures_file: api_failures.json 文件路径
        """
        self.client = genai.Client()
        self.failures_file = Path(failures_file)
        
        if not self.failures_file.exists():
            raise FileNotFoundError(f"Failures file not found: {self.failures_file}")
        
        # 加载失败记录
        with open(self.failures_file, 'r') as f:
            self.failures = json.load(f)
        
        # 输出目录（和failures文件在同一目录）
        self.output_dir = self.failures_file.parent
        self.retry_log = self.output_dir / "retry_log.json"
        
        # 加载已重试记录
        self.retry_results = self.load_retry_log()
    
    def load_retry_log(self):
        """加载重试日志"""
        if self.retry_log.exists():
            with open(self.retry_log, 'r') as f:
                return json.load(f)
        return []
    
    def save_retry_log(self):
        """保存重试日志"""
        with open(self.retry_log, 'w') as f:
            json.dump(self.retry_results, f, indent=2, ensure_ascii=False)
    
    def api_call_with_retry(self, func, max_retries=3):
        """API调用带重试"""
        for attempt in range(max_retries):
            try:
                return func(), None
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    time.sleep(wait_time)
                else:
                    return None, str(e)
        return None, "Max retries exceeded"
    
    def retry_single_failure(self, failure_record):
        """重试单个失败记录"""
        task = failure_record.get("task")
        step = failure_record.get("step")
        error = failure_record.get("error")
        
        print(f"\n重试任务: {task}")
        print(f"  步骤: {step}")
        print(f"  原错误: {error}")
        
        # 这里只是验证API是否恢复
        # 实际的重新处理需要在主程序中手动删除progress记录后重新运行
        
        # 简单测试：调用一次API看是否成功
        def test_call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents="Hello, are you available?",
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
        
        response, retry_error = self.api_call_with_retry(test_call)
        
        result = {
            "task": task,
            "step": step,
            "original_error": error,
            "retry_success": retry_error is None,
            "retry_error": retry_error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.retry_results.append(result)
        
        return result
    
    def run(self, filter_task=None, filter_step=None):
        """
        运行重试
        
        Args:
            filter_task: 仅重试指定任务（可选）
            filter_step: 仅重试指定步骤（可选）
        """
        print(f"\n{'='*80}")
        print(f"API失败任务重试")
        print(f"{'='*80}")
        print(f"失败记录文件: {self.failures_file}")
        print(f"总失败数: {len(self.failures)}")
        
        # 过滤
        tasks_to_retry = self.failures
        if filter_task:
            tasks_to_retry = [f for f in tasks_to_retry if filter_task in f.get("task", "")]
        if filter_step:
            tasks_to_retry = [f for f in tasks_to_retry if f.get("step") == filter_step]
        
        print(f"待重试: {len(tasks_to_retry)}")
        print(f"{'='*80}\n")
        
        # 统计
        success_count = 0
        failed_count = 0
        
        # 逐个重试
        for failure in tqdm(tasks_to_retry, desc="重试中"):
            result = self.retry_single_failure(failure)
            
            if result["retry_success"]:
                success_count += 1
                print(f"  ✓ 成功")
            else:
                failed_count += 1
                print(f"  ✗ 仍然失败: {result['retry_error']}")
        
        # 保存重试日志
        self.save_retry_log()
        
        # 打印统计
        print(f"\n{'='*80}")
        print(f"重试完成！")
        print(f"{'='*80}")
        print(f"成功: {success_count}")
        print(f"仍失败: {failed_count}")
        print(f"\n重试日志: {self.retry_log}")
        print(f"{'='*80}\n")
        
        if success_count > 0:
            print("提示: API已恢复的任务，需要在主程序的progress.json中删除对应记录，")
            print("      然后重新运行主程序以完成处理。\n")


def main():
    parser = argparse.ArgumentParser(
        description='重试API失败的任务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 重试所有失败任务
  python retry_failed.py --failures-json path/to/api_failures.json
  
  # 仅重试特定任务
  python retry_failed.py --failures-json path/to/api_failures.json --task "s50000230-Pneumothorax"
  
  # 仅重试特定步骤
  python retry_failed.py --failures-json path/to/api_failures.json --step "edit_image"
        """
    )
    parser.add_argument('--failures-json', type=str, required=True,
                       help='api_failures.json 文件路径')
    parser.add_argument('--task', type=str, default=None,
                       help='仅重试包含此字符串的任务（可选）')
    parser.add_argument('--step', type=str, default=None,
                       choices=['generate_prompt', 'update_prompt', 'edit_image', 'verify_image'],
                       help='仅重试指定步骤（可选）')
    
    args = parser.parse_args()
    
    # 创建重试器
    retrier = FailedTaskRetry(args.failures_json)
    
    # 运行
    retrier.run(filter_task=args.task, filter_step=args.step)


if __name__ == '__main__':
    main()

