#!/usr/bin/env python3
"""
添加病症：将正常图像编辑成病症图像
支持多线程、断点续传、API失败重试
"""

from google import genai
from google.genai import types
from pathlib import Path
from PIL import Image
from io import BytesIO
import json
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class DiseaseAdder:
    def __init__(self, dataset_path, output_base, max_workers=20, max_rounds=5):
        """
        初始化病症添加器
        
        Args:
            dataset_path: 数据集路径（如 full-data/MIMIC_single_disease_selection_dim1024_1k_per_class）
            output_base: 输出基础路径（如 img_gen）
            max_workers: 并发线程数
            max_rounds: 最多尝试轮次
        """
        self.client = genai.Client()
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        
        # 输出目录
        self.output_dir = self.output_base / f"{self.dataset_name}-edit"
        self.failed_dir = self.output_base / f"{self.dataset_name}-edit-failed"
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        
        # 进度和失败记录
        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"
        
        # 加载进度
        self.progress = self.load_progress()
        self.api_failures = self.load_api_failures()
        self.failed_summary = self.load_failed_summary()
        self.final_prompts = self.load_final_prompts()
        self.all_conversations = self.load_all_conversations()
        
        # 线程锁
        self.lock = threading.Lock()
        
    def load_progress(self):
        """加载进度"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """保存进度（线程安全）"""
        with self.lock:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def load_api_failures(self):
        """加载API失败记录"""
        if self.api_failures_file.exists():
            with open(self.api_failures_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_api_failures(self):
        """保存API失败记录（线程安全）"""
        with self.lock:
            with open(self.api_failures_file, 'w') as f:
                json.dump(self.api_failures, f, indent=2, ensure_ascii=False)
    
    def load_failed_summary(self):
        """加载失败任务摘要"""
        if self.failed_summary_file.exists():
            with open(self.failed_summary_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_failed_summary(self):
        """保存失败任务摘要（线程安全）"""
        with self.lock:
            with open(self.failed_summary_file, 'w') as f:
                json.dump(self.failed_summary, f, indent=2, ensure_ascii=False)
    
    def load_final_prompts(self):
        """加载最终prompts"""
        if self.final_prompts_file.exists():
            with open(self.final_prompts_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_final_prompts(self):
        """保存最终prompts（线程安全）"""
        with self.lock:
            with open(self.final_prompts_file, 'w') as f:
                json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)
    
    def load_all_conversations(self):
        """加载所有对话记录"""
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_all_conversations(self):
        """保存所有对话记录（线程安全）"""
        with self.lock:
            with open(self.conversations_file, 'w') as f:
                json.dump(self.all_conversations, f, indent=2, ensure_ascii=False)
    
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
    
    def extract_thinking_and_content(self, response):
        """提取思考过程和内容"""
        thinking = ""
        content = ""
        
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thinking += part.text + "\n"
            else:
                content += part.text + "\n"
        
        return thinking.strip(), content.strip()
    
    def generate_initial_prompt(self, disease_name, dataset_type):
        """生成初始编辑prompt"""
        system_instruction = f"""You are a medical image editing expert. Generate a clear, concise prompt to edit a normal {dataset_type} image to show {disease_name}.

Requirements:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

Return ONLY the editing prompt in English, no explanations."""
        
        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=system_instruction,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None
    
    def update_prompt(self, original_image_path, disease_name, prompt_history, dataset_type):
        """
        更新编辑prompt
        
        Args:
            original_image_path: 原始图像路径
            disease_name: 疾病名称
            prompt_history: 历史prompt列表，格式为 [{"round": 1, "prompt": "...", "verification": {...}}, ...]
            dataset_type: 数据集类型
        """
        with open(original_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 构建历史失败记录
        history_text = ""
        for i, history in enumerate(prompt_history, 1):
            history_text += f"""
Attempt {i}:
  Prompt: {history['prompt']}
  Verification Result:
    - Has disease: {history['verification']['has_disease']}
    - Structure reasonable: {history['verification']['structure_reasonable']}
    - Looks realistic: {history['verification']['looks_realistic']}
    - Reason: {history['verification']['reason']}
"""
        
        system_instruction = f"""You are a medical image editing expert. Multiple previous editing attempts have failed. You need to analyze ALL previous attempts and generate a BETTER prompt.

HISTORY OF ALL PREVIOUS ATTEMPTS:
{history_text}

Looking at the ORIGINAL image and analyzing the patterns of failures above, generate an IMPROVED editing prompt to add {disease_name} to this {dataset_type} image.

ANALYSIS REQUIREMENTS:
1. Identify common issues across multiple attempts
2. Learn from what didn't work in previous rounds
3. Avoid repeating the same mistakes
4. Address ALL verification issues mentioned in the history

BASIC REQUIREMENTS:
1. Easy for non-medical personnel to understand
2. Must look natural and realistic, not detectable as deepfake
3. Should follow common sense and medical plausibility
4. Be specific about what to add/modify

CRITICAL CONSTRAINTS - Fidelity Preservation:
- Preserve original image noise, grain texture, and contrast
- Do NOT alter device artifacts, watermarks, grid lines, or annotations
- Keep the same imaging characteristics (brightness, sharpness, field of view)

CRITICAL CONSTRAINTS - Negative Rules:
- NO adding text, labels, or annotations
- NO sharp unnatural edges or boundaries
- NO repetitive/duplicated structures
- NO non-target pathologies
- NO obvious deepfake artifacts

CRITICAL CONSTRAINTS - Minimal Change Principle (Counterfactual Minimality):
- ONLY modify areas directly related to {disease_name}
- Keep all other anatomical structures, textures, and background UNCHANGED
- Minimal intervention: change as little as possible to show the disease

Return ONLY the improved editing prompt in English, no explanations."""
        
        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg'),
                    system_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, prompt = self.extract_thinking_and_content(response)
        return thinking, prompt, None
    
    def edit_image(self, image_path, edit_prompt):
        """编辑图像"""
        image = Image.open(image_path)
        
        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[edit_prompt, image]
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, error
        
        # 提取生成的图像
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image, None
        
        return None, "No image generated"
    
    def verify_edited_image(self, edited_image, disease_name, dataset_type):
        """验证编辑后的图像"""
        # 转换为字节
        img_byte_arr = BytesIO()
        edited_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        verification_instruction = f"""You are a medical image verification expert. Evaluate if this {dataset_type} image shows {disease_name}.

IMPORTANT: Take your time to think carefully. Medical image editing is challenging, and minor imperfections are acceptable as long as the overall goal is achieved. Be thoughtful and balanced in your evaluation - don't reject an image for trivial issues.

Check these aspects:
1. Has disease: Does the image show signs of {disease_name}? (Consider: Are the disease features visible and recognizable, even if not perfect?)
2. Structure reasonable: Are the anatomical structures reasonable and correct? (Consider: Minor artifacts are acceptable if the overall anatomy is preserved)
3. Looks realistic: Does it look like a real medical image? (Consider: Some editing traces are inevitable; focus on whether it could pass as a real medical image to non-experts)

Additional verification for image fidelity:
- Check if the image preserves natural noise/grain texture (minor changes are acceptable)
- Check if there are unnatural sharp edges or boundaries (slight artifacts are tolerable if not obvious)
- Check if there are added text, repetitive structures, or deepfake artifacts (focus on major issues, not minor imperfections)
- Check if modifications are minimal (only disease-related changes; some collateral changes are acceptable)

Before deciding, ask yourself:
- Would this image be useful for the intended purpose despite minor flaws?
- Are the issues critical or just cosmetic?
- Does the image achieve the main goal of showing {disease_name}?

Return your evaluation in this JSON format:
{{
    "qualified": true/false,
    "has_disease": true/false,
    "structure_reasonable": true/false,
    "looks_realistic": true/false,
    "reason": "detailed explanation (mention both strengths and weaknesses; explain your reasoning for acceptance or rejection)"
}}

Only qualified if ALL three aspects are true AND no MAJOR fidelity issues detected. Minor imperfections are acceptable."""
        
        def call():
            return self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    verification_instruction
                ],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, None, error
        
        thinking, content = self.extract_thinking_and_content(response)
        
        # 解析JSON
        try:
            # 提取JSON
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            return thinking, result, None
        except json.JSONDecodeError as e:
            return thinking, None, f"JSON parse error: {str(e)}"
    
    def get_dataset_type(self):
        """获取数据集类型描述"""
        name = self.dataset_name.lower()
        if 'mimic' in name or 'chest' in name:
            return "chest X-ray"
        elif 'brain' in name or 'tumor' in name:
            return "brain MRI"
        elif 'odir' in name or 'fundus' in name:
            return "fundus"
        return "medical"
    
    def process_single_task(self, normal_image_path, disease_name):
        """处理单个编辑任务"""
        image_name = Path(normal_image_path).stem
        task_key = f"{image_name}-{disease_name}"
        
        # 检查是否已完成
        if task_key in self.progress:
            return {"status": "skipped", "task": task_key}
        
        dataset_type = self.get_dataset_type()
        
        # 对话记录
        conversation = {
            "image": Path(normal_image_path).name,
            "disease": disease_name,
            "status": "failed",
            "rounds": []
        }
        
        current_prompt = None
        
        for round_num in range(1, self.max_rounds + 1):
            round_data = {"round": round_num}
            
            # 1. 生成或更新prompt
            if round_num == 1:
                thinking, prompt, error = self.generate_initial_prompt(disease_name, dataset_type)
                if error:
                    # API失败 - 记录并标记为已处理（跳过）
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key,
                            "step": "generate_prompt",
                            "round": round_num,
                            "error": error,
                            "image": Path(normal_image_path).name,
                            "disease": disease_name
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}
                
                round_data["generate_prompt"] = {
                    "thinking_summary": thinking,
                    "prompt": prompt
                }
                current_prompt = prompt
            else:
                # 构建完整的历史记录供LLM分析
                prompt_history = []
                for prev_round in conversation["rounds"]:
                    if "generate_prompt" in prev_round and "verification" in prev_round:
                        prompt_history.append({
                            "round": prev_round["round"],
                            "prompt": prev_round["generate_prompt"]["prompt"],
                            "verification": prev_round["verification"]
                        })
                
                # 使用完整历史更新prompt
                thinking, prompt, error = self.update_prompt(
                    normal_image_path, disease_name, prompt_history, dataset_type
                )
                if error:
                    # API失败 - 记录并标记为已处理（跳过）
                    with self.lock:
                        self.api_failures.append({
                            "task": task_key,
                            "step": "update_prompt",
                            "round": round_num,
                            "error": error,
                            "image": Path(normal_image_path).name,
                            "disease": disease_name
                        })
                        self.progress[task_key] = "api_failed"
                    self.save_api_failures()
                    self.save_progress()
                    return {"status": "api_failed", "task": task_key, "error": error}
                
                round_data["generate_prompt"] = {
                    "thinking_summary": thinking,
                    "prompt": prompt
                }
                current_prompt = prompt
            
            # 2. 编辑图像
            edited_image, error = self.edit_image(normal_image_path, current_prompt)
            if error:
                round_data["edit_result"] = {"success": False, "error": error}
                conversation["rounds"].append(round_data)
                
                # API失败 - 记录并标记为已处理（跳过）
                with self.lock:
                    self.api_failures.append({
                        "task": task_key,
                        "step": "edit_image",
                        "round": round_num,
                        "error": error,
                        "image": Path(normal_image_path).name,
                        "disease": disease_name
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}
            
            round_data["edit_result"] = {"success": True}
            
            # 3. 验证图像
            thinking, verification, error = self.verify_edited_image(edited_image, disease_name, dataset_type)
            if error:
                round_data["verification"] = {"error": error}
                conversation["rounds"].append(round_data)
                
                # API失败 - 记录并标记为已处理（跳过）
                with self.lock:
                    self.api_failures.append({
                        "task": task_key,
                        "step": "verify_image",
                        "round": round_num,
                        "error": error,
                        "image": Path(normal_image_path).name,
                        "disease": disease_name
                    })
                    self.progress[task_key] = "api_failed"
                self.save_api_failures()
                self.save_progress()
                return {"status": "api_failed", "task": task_key, "error": error}
            
            round_data["verification"] = verification
            round_data["verification"]["thinking_summary"] = thinking
            conversation["rounds"].append(round_data)
            
            # 4. 检查是否合格
            if verification.get("qualified", False):
                # 成功！保存图像
                output_subdir = self.output_dir / disease_name
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{image_name}-{disease_name}-edit.jpeg"
                edited_image.save(output_path, 'JPEG', quality=95)
                
                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_image_path"] = str(output_path)
                
                # 保存到集中的对话记录
                with self.lock:
                    self.all_conversations[task_key] = conversation
                self.save_all_conversations()
                
                # 保存最终prompt
                with self.lock:
                    self.final_prompts[task_key] = {
                        "image": Path(normal_image_path).name,
                        "disease": disease_name,
                        "status": "success",
                        "final_prompt": current_prompt,
                        "rounds": round_num
                    }
                self.save_final_prompts()
                
                # 更新进度
                with self.lock:
                    self.progress[task_key] = "success"
                self.save_progress()
                
                return {"status": "success", "task": task_key, "rounds": round_num}
            else:
                # 保存失败的中间图像
                failed_subdir = self.failed_dir / disease_name
                failed_subdir.mkdir(parents=True, exist_ok=True)
                failed_path = failed_subdir / f"{image_name}-{disease_name}-edit-failed-{round_num}.jpeg"
                edited_image.save(failed_path, 'JPEG', quality=95)
        
        # 所有轮次都失败 - 图像仅保存在failed目录
        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        conversation["final_image_path"] = str(self.failed_dir / disease_name / f"{image_name}-{disease_name}-edit-failed-{self.max_rounds}.jpeg")
        
        # 保存到集中的对话记录
        with self.lock:
            self.all_conversations[task_key] = conversation
        self.save_all_conversations()
        
        # 保存到失败摘要（包含完整对话历史）
        with self.lock:
            self.failed_summary.append({
                "task": task_key,
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "final_prompt": current_prompt,
                "rounds": self.max_rounds,
                "last_verification": conversation["rounds"][-1]["verification"],
                "final_image_path": conversation["final_image_path"],
                "full_conversation": conversation  # 保存完整对话历史
            })
        self.save_failed_summary()
        
        # 保存最终prompt
        with self.lock:
            self.final_prompts[task_key] = {
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "status": "failed",
                "final_prompt": current_prompt,
                "rounds": self.max_rounds
            }
        self.save_final_prompts()
        
        # 更新进度（标记为已处理，避免重复）
        with self.lock:
            self.progress[task_key] = "failed"
        self.save_progress()
        
        return {"status": "failed", "task": task_key, "rounds": self.max_rounds}
    
    def get_all_tasks(self):
        """获取所有待处理任务"""
        tasks = []
        
        # 找到正常图像目录
        normal_dir = self.dataset_path / "normal"
        if not normal_dir.exists():
            raise ValueError(f"Normal directory not found: {normal_dir}")
        
        # 找到所有病症类别
        disease_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name != "normal"]
        disease_names = [d.name for d in disease_dirs]
        
        # 找到所有正常图像
        normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
        
        # 为每个正常图像生成所有病症的编辑任务
        for normal_img in normal_images:
            for disease in disease_names:
                tasks.append((str(normal_img), disease))
        
        return tasks
    
    def run(self):
        """运行完整工作流"""
        tasks = self.get_all_tasks()
        
        print(f"\n{'='*80}")
        print(f"添加病症 - {self.dataset_name}")
        print(f"{'='*80}")
        print(f"数据集路径: {self.dataset_path}")
        print(f"输出目录: {self.output_dir}")
        print(f"总任务数: {len(tasks)}")
        print(f"已完成: {len(self.progress)}")
        print(f"待处理: {len(tasks) - len(self.progress)}")
        print(f"并发线程: {self.max_workers}")
        print(f"最大轮次: {self.max_rounds}")
        print(f"{'='*80}\n")
        
        # 统计
        stats = {
            "success": 0,
            "failed": 0,
            "api_failed": 0,
            "skipped": 0
        }
        
        # 多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_task, img_path, disease): (img_path, disease)
                for img_path, disease in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理中"):
                try:
                    result = future.result()
                    stats[result["status"]] += 1
                except Exception as e:
                    print(f"\n任务异常: {e}")
                    stats["api_failed"] += 1
        
        # 打印最终统计
        print(f"\n{'='*80}")
        print(f"处理完成！")
        print(f"{'='*80}")
        print(f"成功: {stats['success']}")
        print(f"失败: {stats['failed']}")
        print(f"API失败: {stats['api_failed']}")
        print(f"跳过: {stats['skipped']}")
        print(f"\n对话记录: {self.conv_dir}")
        print(f"最终prompts: {self.final_prompts_file}")
        print(f"失败摘要: {self.failed_summary_file}")
        if self.api_failures:
            print(f"API失败记录: {self.api_failures_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='添加病症：将正常图像编辑成病症图像')
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集名称（如 Chest-XRay, Brain-MRI, Fundus）')
    parser.add_argument('--max-workers', type=int, default=20,
                       help='并发线程数（默认: 20）')
    parser.add_argument('--max-rounds', type=int, default=5,
                       help='最大尝试轮次（默认: 5）')
    
    args = parser.parse_args()
    
    # 路径设置
    BASE_DIR = Path(__file__).parent
    dataset_path = BASE_DIR / "full-data" / args.dataset
    
    if not dataset_path.exists():
        print(f"错误: 数据集不存在 {dataset_path}")
        return
    
    # 创建处理器
    adder = DiseaseAdder(
        dataset_path=dataset_path,
        output_base=BASE_DIR,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds
    )
    
    # 运行
    adder.run()


if __name__ == '__main__':
    main()

