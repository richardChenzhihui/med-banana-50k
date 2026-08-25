#!/usr/bin/env python3
"""
Add pathology: edit normal medical images into diseased ones.
Multi-threaded, resumable across runs, with retries on API failure.
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


# Default Gemini model IDs used to build Med-Banana-80K. `gemini-2.5-flash-image-preview`
# is the API id of the image editor released under the "Nano Banana" name; the reasoning
# model drives prompt generation, prompt refinement and LLM-as-Judge verification.
# Override on the command line with --editor-model / --verifier-model.
DEFAULT_EDITOR_MODEL = "gemini-2.5-flash-image-preview"
DEFAULT_VERIFIER_MODEL = "gemini-2.5-pro"

class DiseaseAdder:
    def __init__(self, dataset_path, output_base, max_workers=20, max_rounds=5,
                 editor_model=DEFAULT_EDITOR_MODEL, verifier_model=DEFAULT_VERIFIER_MODEL):
        """
        Initialise the disease adder.
        
        Args:
            dataset_path: dataset directory, e.g. full-data/Chest-XRay
            output_base: base directory for generated output
            max_workers: number of concurrent worker threads
            max_rounds: maximum edit-verify-refine rounds per image
        """
        self.client = genai.Client()
        self.dataset_path = Path(dataset_path)
        self.dataset_name = self.dataset_path.name
        self.output_base = Path(output_base)
        self.max_workers = max_workers
        self.max_rounds = max_rounds
        self.editor_model = editor_model
        self.verifier_model = verifier_model
        
        # output directories
        self.output_dir = self.output_base / f"{self.dataset_name}-edit"
        self.failed_dir = self.output_base / f"{self.dataset_name}-edit-failed"
        
        # create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        
        # progress and failure logs
        self.progress_file = self.output_dir / "progress.json"
        self.api_failures_file = self.output_dir / "api_failures.json"
        self.failed_summary_file = self.failed_dir / "failed_summary.json"
        self.final_prompts_file = self.output_dir / "final_prompts.json"
        self.conversations_file = self.output_dir / "all_conversations.json"
        
        # load progress for resume
        self.progress = self.load_progress()
        self.api_failures = self.load_api_failures()
        self.failed_summary = self.load_failed_summary()
        self.final_prompts = self.load_final_prompts()
        self.all_conversations = self.load_all_conversations()
        
        # thread locks
        self.lock = threading.Lock()
        
    def load_progress(self):
        """Load progress, enabling resume across runs."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Persist progress (thread-safe)."""
        with self.lock:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
    
    def load_api_failures(self):
        """Load the API-failure log."""
        if self.api_failures_file.exists():
            with open(self.api_failures_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_api_failures(self):
        """Persist the API-failure log (thread-safe)."""
        with self.lock:
            with open(self.api_failures_file, 'w') as f:
                json.dump(self.api_failures, f, indent=2, ensure_ascii=False)
    
    def load_failed_summary(self):
        """Load the failed-task summary."""
        if self.failed_summary_file.exists():
            with open(self.failed_summary_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_failed_summary(self):
        """Persist the failed-task summary (thread-safe)."""
        with self.lock:
            with open(self.failed_summary_file, 'w') as f:
                json.dump(self.failed_summary, f, indent=2, ensure_ascii=False)
    
    def load_final_prompts(self):
        """Load the accepted prompts."""
        if self.final_prompts_file.exists():
            with open(self.final_prompts_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_final_prompts(self):
        """Persist the accepted prompts (thread-safe)."""
        with self.lock:
            with open(self.final_prompts_file, 'w') as f:
                json.dump(self.final_prompts, f, indent=2, ensure_ascii=False)
    
    def load_all_conversations(self):
        """Load all conversation records."""
        if self.conversations_file.exists():
            with open(self.conversations_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_all_conversations(self):
        """Persist all conversation records (thread-safe)."""
        with self.lock:
            with open(self.conversations_file, 'w') as f:
                json.dump(self.all_conversations, f, indent=2, ensure_ascii=False)
    
    def api_call_with_retry(self, func, max_retries=3):
        """Call the API with exponential-backoff retries."""
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
        """Split the response into its thinking trace and its content."""
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
        """Generate the round-1 editing prompt."""
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
                model=self.verifier_model,
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
        Refine the editing prompt from the attempt history.
        
        Args:
            original_image_path: path to the source image
            disease_name: target pathology
            prompt_history: prior attempts, as [{"round": 1, "prompt": "...", "verification": {...}}, ...]
            dataset_type: imaging-modality description
        """
        with open(original_image_path, 'rb') as f:
            image_bytes = f.read()
        
        # build the failure history
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
                model=self.verifier_model,
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
        """Apply the editing prompt to the source image."""
        image = Image.open(image_path)
        
        def call():
            return self.client.models.generate_content(
                model=self.editor_model,
                contents=[edit_prompt, image]
            )
        
        response, error = self.api_call_with_retry(call)
        if error:
            return None, error
        
        # extract the generated image
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                edited_image = Image.open(BytesIO(part.inline_data.data))
                return edited_image, None
        
        return None, "No image generated"
    
    def verify_edited_image(self, edited_image, disease_name, dataset_type):
        """Verify the edited image with the LLM-as-Judge."""
        # convert to bytes
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
                model=self.verifier_model,
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
        
        # parse the JSON verdict
        try:
            # extract the JSON block
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            result = json.loads(content)
            return thinking, result, None
        except json.JSONDecodeError as e:
            return thinking, None, f"JSON parse error: {str(e)}"
    
    def get_dataset_type(self):
        """Describe the imaging modality, keyword-matched from the dataset name."""
        name = self.dataset_name.lower()
        if 'mimic' in name or 'chest' in name:
            return "chest X-ray"
        elif 'brain' in name or 'tumor' in name:
            return "brain MRI"
        elif 'odir' in name or 'fundus' in name:
            return "fundus"
        return "medical"
    
    def process_single_task(self, normal_image_path, disease_name):
        """Run the edit-verify-refine loop for one addition task."""
        image_name = Path(normal_image_path).stem
        task_key = f"{image_name}-{disease_name}"
        
        # skip tasks already completed
        if task_key in self.progress:
            return {"status": "skipped", "task": task_key}
        
        dataset_type = self.get_dataset_type()
        
        # conversation records
        conversation = {
            "image": Path(normal_image_path).name,
            "disease": disease_name,
            "status": "failed",
            "rounds": []
        }
        
        current_prompt = None
        
        for round_num in range(1, self.max_rounds + 1):
            round_data = {"round": round_num}
            
            # 1. generate or refine the prompt
            if round_num == 1:
                thinking, prompt, error = self.generate_initial_prompt(disease_name, dataset_type)
                if error:
                    # API failure: record it and mark the task processed so the run can move on
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
                # build the full history for the model to analyse
                prompt_history = []
                for prev_round in conversation["rounds"]:
                    if "generate_prompt" in prev_round and "verification" in prev_round:
                        prompt_history.append({
                            "round": prev_round["round"],
                            "prompt": prev_round["generate_prompt"]["prompt"],
                            "verification": prev_round["verification"]
                        })
                
                # refine the prompt using the full attempt history
                thinking, prompt, error = self.update_prompt(
                    normal_image_path, disease_name, prompt_history, dataset_type
                )
                if error:
                    # API failure: record it and mark the task processed so the run can move on
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
            
            # 2. edit the image
            edited_image, error = self.edit_image(normal_image_path, current_prompt)
            if error:
                round_data["edit_result"] = {"success": False, "error": error}
                conversation["rounds"].append(round_data)
                
                # API failure: record it and mark the task processed so the run can move on
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
            
            # 3. verify the result
            thinking, verification, error = self.verify_edited_image(edited_image, disease_name, dataset_type)
            if error:
                round_data["verification"] = {"error": error}
                conversation["rounds"].append(round_data)
                
                # API failure: record it and mark the task processed so the run can move on
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
            
            # 4. accept or reject
            if verification.get("qualified", False):
                # accepted: write the image
                output_subdir = self.output_dir / disease_name
                output_subdir.mkdir(parents=True, exist_ok=True)
                output_path = output_subdir / f"{image_name}-{disease_name}-edit.jpeg"
                edited_image.save(output_path, 'JPEG', quality=95)
                
                conversation["status"] = "success"
                conversation["final_prompt"] = current_prompt
                conversation["final_image_path"] = str(output_path)
                
                # append to the central conversation log
                with self.lock:
                    self.all_conversations[task_key] = conversation
                self.save_all_conversations()
                
                # record the accepted prompt
                with self.lock:
                    self.final_prompts[task_key] = {
                        "image": Path(normal_image_path).name,
                        "disease": disease_name,
                        "status": "success",
                        "final_prompt": current_prompt,
                        "rounds": round_num
                    }
                self.save_final_prompts()
                
                # update progress
                with self.lock:
                    self.progress[task_key] = "success"
                self.save_progress()
                
                return {"status": "success", "task": task_key, "rounds": round_num}
            else:
                # keep the rejected candidate
                failed_subdir = self.failed_dir / disease_name
                failed_subdir.mkdir(parents=True, exist_ok=True)
                failed_path = failed_subdir / f"{image_name}-{disease_name}-edit-failed-{round_num}.jpeg"
                edited_image.save(failed_path, 'JPEG', quality=95)
        
        # every round was rejected: the images live only in the failed directory
        conversation["status"] = "failed"
        conversation["final_prompt"] = current_prompt
        conversation["final_image_path"] = str(self.failed_dir / disease_name / f"{image_name}-{disease_name}-edit-failed-{self.max_rounds}.jpeg")
        
        # append to the central conversation log
        with self.lock:
            self.all_conversations[task_key] = conversation
        self.save_all_conversations()
        
        # append to the failure summary, with the full conversation
        with self.lock:
            self.failed_summary.append({
                "task": task_key,
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "final_prompt": current_prompt,
                "rounds": self.max_rounds,
                "last_verification": conversation["rounds"][-1]["verification"],
                "final_image_path": conversation["final_image_path"],
                "full_conversation": conversation  # full conversation history
            })
        self.save_failed_summary()
        
        # record the accepted prompt
        with self.lock:
            self.final_prompts[task_key] = {
                "image": Path(normal_image_path).name,
                "disease": disease_name,
                "status": "failed",
                "final_prompt": current_prompt,
                "rounds": self.max_rounds
            }
        self.save_final_prompts()
        
        # mark processed so a resumed run does not repeat it
        with self.lock:
            self.progress[task_key] = "failed"
        self.save_progress()
        
        return {"status": "failed", "task": task_key, "rounds": self.max_rounds}
    
    def get_all_tasks(self):
        """Enumerate every task to process."""
        tasks = []
        
        # locate the normal/ directory
        normal_dir = self.dataset_path / "normal"
        if not normal_dir.exists():
            raise ValueError(f"Normal directory not found: {normal_dir}")
        
        # collect the disease categories
        disease_dirs = [d for d in self.dataset_path.iterdir() 
                       if d.is_dir() and d.name != "normal"]
        disease_names = [d.name for d in disease_dirs]
        
        # collect the normal source images
        normal_images = list(normal_dir.glob("*.jpeg")) + list(normal_dir.glob("*.jpg"))
        
        # cross product: every normal image x every disease directory
        for normal_img in normal_images:
            for disease in disease_names:
                tasks.append((str(normal_img), disease))
        
        return tasks
    
    def run(self):
        """Run the full workflow over every task."""
        tasks = self.get_all_tasks()
        
        print(f"\n{'='*80}")
        print(f"Add pathology - {self.dataset_name}")
        print(f"{'='*80}")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total tasks: {len(tasks)}")
        print(f"Already done: {len(self.progress)}")
        print(f"Pending: {len(tasks) - len(self.progress)}")
        print(f"Worker threads: {self.max_workers}")
        print(f"Max rounds: {self.max_rounds}")
        print(f"{'='*80}\n")
        
        # statistics
        stats = {
            "success": 0,
            "failed": 0,
            "api_failed": 0,
            "skipped": 0
        }
        
        # run tasks concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.process_single_task, img_path, disease): (img_path, disease)
                for img_path, disease in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="processing"):
                try:
                    result = future.result()
                    stats[result["status"]] += 1
                except Exception as e:
                    print(f"\nTask raised: {e}")
                    stats["api_failed"] += 1
        
        # print final statistics
        print(f"\n{'='*80}")
        print("Done.")
        print(f"{'='*80}")
        print(f"Accepted: {stats['success']}")
        print(f"Rejected: {stats['failed']}")
        print(f"API failed: {stats['api_failed']}")
        print(f"Skipped: {stats['skipped']}")
        print(f"\nConversation records: {self.conv_dir}")
        print(f"Accepted prompts: {self.final_prompts_file}")
        print(f"Failure summary: {self.failed_summary_file}")
        if self.api_failures:
            print(f"API failure log: {self.api_failures_file}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Add pathology: edit normal medical images into diseased ones (Med-Banana-80K construction)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset folder name under dataset_construction/full-data/ '
                            '(e.g. Chest-XRay, Brain-MRI, Fundus)')
    parser.add_argument('--max-workers', type=int, default=20,
                       help='Number of concurrent worker threads (default: 20)')
    parser.add_argument('--max-rounds', type=int, default=5,
                       help='Maximum edit-verify-refine rounds per image (default: 5)')
    parser.add_argument('--editor-model', type=str, default=DEFAULT_EDITOR_MODEL,
                       help=f'Gemini image-editing model (default: {DEFAULT_EDITOR_MODEL})')
    parser.add_argument('--verifier-model', type=str, default=DEFAULT_VERIFIER_MODEL,
                       help='Gemini reasoning model used for prompt generation, prompt '
                            f'refinement and verification (default: {DEFAULT_VERIFIER_MODEL})')

    args = parser.parse_args()

    BASE_DIR = Path(__file__).parent
    dataset_path = BASE_DIR / "full-data" / args.dataset

    if not dataset_path.exists():
        raise SystemExit(
            f"Dataset not found: {dataset_path}\n"
            f"Expected layout: dataset_construction/full-data/{args.dataset}/"
            "{normal,<disease-1>,<disease-2>,...}/"
        )

    adder = DiseaseAdder(
        dataset_path=dataset_path,
        output_base=BASE_DIR,
        max_workers=args.max_workers,
        max_rounds=args.max_rounds,
        editor_model=args.editor_model,
        verifier_model=args.verifier_model,
    )

    adder.run()


if __name__ == '__main__':
    main()

