"""
Multi-Target Speaker Extraction (MTSE)
多目标说话人提取工具 - 批量提取多个目标说话人，保持原始音质和声道

Batch speaker extraction with voice activity detection and GPU acceleration.
"""

import os
import json
import numpy as np
import torch
import soundfile as sf
import librosa
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    raise ImportError("Please install NeMo: pip install nemo_toolkit[asr] | 请安装NeMo: pip install nemo_toolkit[asr]")

from speaker_state_manager import SpeakerManager


class MultiSpeakerVerification:
    """多说话人验证系统"""
    
    def __init__(
        self,
        reference_dir: str,
        dataset_dir: str,
        output_dir: str,
        similarity_threshold: float = 0.70,
        min_duration: float = 0.5,
        merge_gap: float = 0.3,
        speaker_config: Dict = None,
        batch_size: int = 8,
        prefetch_workers: int = 2
    ):
        """
        初始化多说话人验证系统
        
        Args:
            reference_dir: 参考音频根目录（包含多个说话人子文件夹）
            dataset_dir: 待处理数据集目录
            output_dir: 输出根目录
            similarity_threshold: 余弦相似度阈值
            min_duration: 最小片段时长（秒）
            merge_gap: 合并间隔（秒）
            speaker_config: speaker管理配置
            batch_size: 批量推理大小
            prefetch_workers: 预加载线程数
        """
        self.reference_dir = Path(reference_dir)
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.similarity_threshold = similarity_threshold
        self.min_duration = min_duration
        self.merge_gap = merge_gap
        self.speaker_config = speaker_config or {}
        self.batch_size = batch_size
        self.prefetch_workers = prefetch_workers
        
        # Set device | 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device | 使用设备: {self.device}")
        
        # 预加载缓存
        self._prefetch_cache = {}
        self._prefetch_lock = threading.Lock()
        
        # 异步保存队列
        self._save_executor = ThreadPoolExecutor(max_workers=2)
        self._pending_saves = []
        self._files_processed = 0
        self._gc_interval = 5  # 每处理N个文件清理一次GPU缓存（太大会导致显存累积）
        
        # 加载模型
        self._load_models()
        
        # 确定要处理的speaker
        self.speakers_to_process, self.skip_info = self._determine_speakers_to_process()
        
        # 提取要处理的说话人的embedding
        self.speaker_embeddings = self._extract_all_speaker_embeddings(self.speakers_to_process)
        
    def _load_models(self):
        """Load pretrained models | 加载预训练模型"""
        print("Loading models... | 正在加载模型...")
        
        # Speaker Embedding model (on GPU)
        print(f"  - Speaker Embedding Model → {self.device}")
        self.speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            "nvidia/speakerverification_en_titanet_large"
        )
        self.speaker_model.to(self.device)
        self.speaker_model.eval()
        
        # Silero VAD model (ONNX version for speed)
        print("  - Silero VAD Model → ONNX (CPU)")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        self.get_speech_timestamps, _, self.read_audio, _, _ = self.vad_utils
        
        print(f"  - Prefetch workers | 预加载线程数: {self.prefetch_workers}")
        print("✓ Models loaded | 模型加载完成")
    
    def _determine_speakers_to_process(self) -> tuple[List[str], Dict[str, str]]:
        """Determine which speakers to process | 确定要处理的speaker列表"""
        # Get all speakers
        speaker_dirs = [d for d in self.reference_dir.iterdir() if d.is_dir()]
        all_speakers = [d.name for d in speaker_dirs]
        
        # Use SpeakerManager to filter
        speakers_to_process, skip_info = SpeakerManager.get_speakers_to_process(
            all_speakers, 
            self.speaker_config
        )
        
        # Print skip info
        if skip_info:
            print("\n⊘ Skipped speakers | 跳过的speaker:")
            for speaker, reason in skip_info.items():
                print(f"  - {speaker}: {reason}")
        
        return speakers_to_process, skip_info
    
    def _extract_all_speaker_embeddings(self, speakers_to_process: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for all speakers
        提取所有说话人的embedding
        
        Args:
            speakers_to_process: List of speakers to process, None means all
        """
        print("\nExtracting speaker embeddings... | 正在提取说话人特征...")
        embeddings = {}
        
        # Iterate through all subfolders in reference directory
        speaker_dirs = [d for d in self.reference_dir.iterdir() if d.is_dir()]
        
        if not speaker_dirs:
            raise ValueError(f"No speaker folders found in {self.reference_dir} | 在 {self.reference_dir} 中未找到说话人子文件夹")
        
        for speaker_dir in speaker_dirs:
            speaker_name = speaker_dir.name
            
            # Check if this speaker should be skipped
            if speakers_to_process is not None and speaker_name not in speakers_to_process:
                print(f"  ○ {speaker_name}: Skipped | 跳过")
                continue
            
            audio_files = list(speaker_dir.glob("*.wav"))
            
            if not audio_files:
                print(f"  Warning | 警告: {speaker_name} has no audio files | 文件夹中无音频文件")
                continue
            
            print(f"  - {speaker_name}: {len(audio_files)} samples | 个样本")
            
            # Extract all embeddings for this speaker
            speaker_embs = []
            for audio_file in tqdm(audio_files, desc=f"    Extracting | 提取 {speaker_name}", leave=False):
                try:
                    audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)
                    if len(audio) < sr * 0.5:
                        continue
                    
                    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
                    audio_len = torch.tensor([len(audio)]).to(self.device)
                    
                    with torch.no_grad():
                        _, emb = self.speaker_model(input_signal=audio_tensor, input_signal_length=audio_len)
                        speaker_embs.append(emb.cpu().numpy())
                except Exception as e:
                    print(f"    Warning | 警告: Failed to process | 处理失败 {audio_file.name}: {e}")
            
            if speaker_embs:
                # Compute average embedding and normalize
                avg_emb = np.mean(speaker_embs, axis=0)
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                embeddings[speaker_name] = avg_emb
                print(f"    ✓ {speaker_name} embedding extracted | 提取完成")
            else:
                print(f"    ✗ {speaker_name} no valid audio | 无有效音频")
        
        if not embeddings:
            raise ValueError("Failed to extract any speaker embeddings | 未能提取任何说话人的embedding")
        
        active_speakers = len(embeddings)
        total_speakers = len(speaker_dirs)
        skipped = total_speakers - active_speakers
        
        if skipped > 0:
            print(f"\n✓ Extracted {active_speakers} speaker(s) (skipped {skipped}) | 提取 {active_speakers} 个说话人的特征 (跳过 {skipped} 个)")
        else:
            print(f"\n✓ Extracted {active_speakers} speaker(s) | 共提取 {active_speakers} 个说话人的特征")
        
        return embeddings
    
    def _get_vad_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """使用Silero VAD获取语音活动片段"""
        wav = self.read_audio(audio_path, sampling_rate=16000)
        speech_timestamps = self.get_speech_timestamps(
            wav, 
            self.vad_model,
            sampling_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )
        
        segments = []
        for ts in speech_timestamps:
            start = ts['start'] / 16000.0
            end = ts['end'] / 16000.0
            segments.append((start, end))
        
        return segments
    
    def _prefetch_file(self, audio_path: Path) -> Dict:
        """预加载单个文件的VAD和音频数据（在后台线程运行）"""
        try:
            # VAD检测
            vad_segments = self._get_vad_segments(str(audio_path))
            
            if not vad_segments:
                return {'path': audio_path, 'status': 'no_speech'}
            
            # 预读取音频
            audio_16k, sr_16k = librosa.load(str(audio_path), sr=16000, mono=True)
            audio_orig, sr_orig = sf.read(str(audio_path))
            
            return {
                'path': audio_path,
                'status': 'ready',
                'vad_segments': vad_segments,
                'audio_16k': audio_16k,
                'sr_16k': sr_16k,
                'audio_orig': audio_orig,
                'sr_orig': sr_orig
            }
        except Exception as e:
            return {'path': audio_path, 'status': 'error', 'error': str(e)}
    
    def _get_prefetched(self, audio_path: Path) -> Optional[Dict]:
        """获取预加载的数据"""
        with self._prefetch_lock:
            return self._prefetch_cache.pop(str(audio_path), None)
    
    def _store_prefetched(self, audio_path: Path, data: Dict):
        """存储预加载的数据"""
        with self._prefetch_lock:
            self._prefetch_cache[str(audio_path)] = data
    
    def _save_segment_async(self, output_path: str, segment: np.ndarray, sr: int):
        """异步保存音频片段"""
        sf.write(output_path, segment, sr)
    
    def _save_metadata_async(self, metadata_file: str, result: Dict):
        """异步保存元数据"""
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    def _wait_pending_saves(self):
        """Wait for all pending save tasks to complete | 等待所有挂起的保存任务完成"""
        for future in self._pending_saves:
            try:
                future.result()
            except Exception as e:
                print(f"  ⚠ Save warning | 保存警告: {e}")
        self._pending_saves.clear()
    
    def _cleanup_completed_saves(self):
        """清理已完成的保存任务，释放内存"""
        remaining = []
        for future in self._pending_saves:
            if future.done():
                try:
                    future.result()  # 获取结果以释放资源
                except Exception:
                    pass
            else:
                remaining.append(future)
        self._pending_saves = remaining
    
    def _extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """提取单个音频片段的embedding"""
        if len(audio) < sr * 0.1:
            return None
        
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_len = torch.tensor([len(audio)]).to(self.device)
        
        with torch.no_grad():
            _, emb = self.speaker_model(input_signal=audio_tensor, input_signal_length=audio_len)
            embedding = emb.cpu().numpy()[0]
        
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _extract_embeddings_batch(self, audio_segments: List[np.ndarray], sr: int) -> List[np.ndarray]:
        """批量提取多个音频片段的embedding（GPU加速）"""
        if not audio_segments:
            return []
        
        # 过滤过短的片段，记录有效索引
        valid_indices = []
        valid_segments = []
        min_samples = int(sr * 0.1)
        
        for i, audio in enumerate(audio_segments):
            if len(audio) >= min_samples:
                valid_indices.append(i)
                valid_segments.append(audio)
        
        if not valid_segments:
            return [None] * len(audio_segments)
        
        # Padding到相同长度
        max_len = max(len(s) for s in valid_segments)
        
        # 安全检查：限制最大padding长度（防止显存爆炸）
        # 16000 * 60 = 60秒，超过的片段单独处理
        MAX_SAMPLES = sr * 60
        if max_len > MAX_SAMPLES:
            # 分离超长片段，单独处理
            normal_indices = []
            normal_segments = []
            long_indices = []
            long_segments = []
            
            for i, seg in zip(valid_indices, valid_segments):
                if len(seg) > MAX_SAMPLES:
                    long_indices.append(i)
                    long_segments.append(seg)
                else:
                    normal_indices.append(i)
                    normal_segments.append(seg)
            
            results = [None] * len(audio_segments)
            
            # 处理正常片段
            if normal_segments:
                normal_embs = self._extract_embeddings_batch_internal(normal_segments, sr)
                for i, idx in enumerate(normal_indices):
                    results[idx] = normal_embs[i]
            
            # 单独处理超长片段
            for i, seg in zip(long_indices, long_segments):
                emb = self._extract_embedding(seg, sr)
                results[i] = emb
            
            return results
        
        return self._extract_embeddings_batch_internal(valid_segments, sr, valid_indices, len(audio_segments))
    
    def _extract_embeddings_batch_internal(self, valid_segments: List[np.ndarray], sr: int, 
                                            valid_indices: List[int] = None, total_len: int = None) -> List[np.ndarray]:
        """内部批量处理（已过滤的片段）"""
        if not valid_segments:
            return []
        
        # Padding到相同长度
        max_len = max(len(s) for s in valid_segments)
        padded = np.zeros((len(valid_segments), max_len), dtype=np.float32)
        lengths = []
        
        for i, seg in enumerate(valid_segments):
            padded[i, :len(seg)] = seg
            lengths.append(len(seg))
        
        # 转为tensor
        audio_tensor = torch.tensor(padded, dtype=torch.float32).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(self.device)
        
        # 批量推理
        with torch.no_grad():
            _, embs = self.speaker_model(input_signal=audio_tensor, input_signal_length=lengths_tensor)
            embs_np = embs.cpu().numpy()
        
        # 显式释放GPU张量
        del audio_tensor, lengths_tensor, embs
        
        # L2归一化
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        embs_np = embs_np / norms
        
        # 如果需要重建完整结果列表
        if valid_indices is not None and total_len is not None:
            results = [None] * total_len
            for i, valid_idx in enumerate(valid_indices):
                results[valid_idx] = embs_np[i]
            return results
        
        return list(embs_np)
    
    def _identify_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """识别说话人（返回最匹配的说话人和相似度）"""
        best_speaker = None
        best_similarity = -1.0
        
        for speaker_name, speaker_emb in self.speaker_embeddings.items():
            similarity = float(np.dot(embedding.flatten(), speaker_emb.flatten()))
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_name
        
        return best_speaker, float(best_similarity)
    
    def _identify_speakers_batch(self, embeddings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """批量识别说话人（向量化加速）"""
        if not embeddings:
            return []
        
        # 构建speaker矩阵（一次性）
        speaker_names = list(self.speaker_embeddings.keys())
        speaker_matrix = np.vstack([self.speaker_embeddings[name].flatten() for name in speaker_names])
        
        # 过滤有效的embeddings
        valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
        if not valid_indices:
            return [(None, 0.0)] * len(embeddings)
        
        # 构建embedding矩阵
        valid_embs = np.vstack([embeddings[i].flatten() for i in valid_indices])
        
        # 矩阵乘法计算所有相似度 (N_segments x N_speakers)
        similarities = np.dot(valid_embs, speaker_matrix.T)
        
        # 找到每个segment的最佳匹配
        best_indices = np.argmax(similarities, axis=1)
        best_similarities = np.max(similarities, axis=1)
        
        # 构建结果
        results = [(None, 0.0)] * len(embeddings)
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = (speaker_names[best_indices[i]], float(best_similarities[i]))
        
        return results
    
    def _merge_segments(self, segments: List[Dict]) -> List[Dict]:
        """合并相邻的同一说话人片段"""
        if not segments:
            return []
        
        # 按时间排序
        segments = sorted(segments, key=lambda x: x['start'])
        
        # 过滤短片段
        filtered = [s for s in segments if s['duration'] >= self.min_duration]
        
        if not filtered:
            return []
        
        # 合并相邻片段
        merged = [filtered[0]]
        for current in filtered[1:]:
            last = merged[-1]
            gap = current['start'] - last['end']
            
            # 如果是同一说话人且间隔很短，则合并
            if current['speaker'] == last['speaker'] and gap < self.merge_gap:
                last['end'] = current['end']
                last['duration'] = last['end'] - last['start']
                last['similarity'] = float(max(last['similarity'], current['similarity']))
            else:
                merged.append(current)
        
        return merged
    
    def process_file(self, audio_path: Path, prefetched: Dict = None) -> Dict:
        """Process a single audio file | 处理单个音频文件"""
        print(f"\nProcessing | 处理: {audio_path.name}")
        
        try:
            # Check for prefetched data
            if prefetched and prefetched.get('status') == 'ready':
                print("  - Using prefetched data ✓ | 使用预加载数据 ✓")
                vad_segments = prefetched['vad_segments']
                audio_16k = prefetched['audio_16k']
                sr_16k = prefetched['sr_16k']
                audio_orig = prefetched['audio_orig']
                sr_orig = prefetched['sr_orig']
                print(f"    Detected {len(vad_segments)} speech segments | 检测到 {len(vad_segments)} 个语音片段")
            elif prefetched and prefetched.get('status') == 'no_speech':
                return {'file': audio_path.name, 'status': 'no_speech'}
            elif prefetched and prefetched.get('status') == 'error':
                raise Exception(prefetched.get('error', 'Unknown error'))
            else:
                # No prefetch, process normally
                # 1. VAD detection
                print("  - VAD detection | VAD检测...")
                vad_segments = self._get_vad_segments(str(audio_path))
                print(f"    Detected {len(vad_segments)} speech segments | 检测到 {len(vad_segments)} 个语音片段")
                
                if not vad_segments:
                    return {'file': audio_path.name, 'status': 'no_speech'}
                
                # 2. Load audio
                audio_16k, sr_16k = librosa.load(str(audio_path), sr=16000, mono=True)
                audio_orig, sr_orig = sf.read(str(audio_path))
            
            n_channels = audio_orig.shape[1] if audio_orig.ndim == 2 else 1
            print(f"    Original audio | 原始音频: {sr_orig}Hz, {n_channels} channel(s) | 声道")
            
            # 3. Speaker identification (batch processing)
            print(f"  - Speaker identification | 说话人识别 (batch_size={self.batch_size})...")
            segments_info = []
            
            # 预提取所有片段音频
            segment_audios = []
            segment_indices = []
            for i, (start, end) in enumerate(vad_segments):
                start_sample = int(start * sr_16k)
                end_sample = int(end * sr_16k)
                segment_audios.append(audio_16k[start_sample:end_sample])
                segment_indices.append(i)
            
            # 按长度排序，让相似长度的片段在同一batch（减少padding浪费）
            sorted_indices = sorted(range(len(segment_audios)), key=lambda i: len(segment_audios[i]))
            sorted_audios = [segment_audios[i] for i in sorted_indices]
            
            # 批量处理（使用排序后的顺序）
            sorted_embeddings = []
            num_batches = (len(sorted_audios) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in tqdm(range(num_batches), desc="    Batch inference | 批量推理", leave=False):
                batch_start = batch_idx * self.batch_size
                batch_end = min(batch_start + self.batch_size, len(sorted_audios))
                batch_audios = sorted_audios[batch_start:batch_end]
                
                # 批量提取embedding
                batch_embs = self._extract_embeddings_batch(batch_audios, sr_16k)
                sorted_embeddings.extend(batch_embs)
            
            # 恢复原始顺序
            all_embeddings = [None] * len(segment_audios)
            for i, orig_idx in enumerate(sorted_indices):
                all_embeddings[orig_idx] = sorted_embeddings[i]
            
            # 批量识别说话人（向量化加速）
            speaker_results = self._identify_speakers_batch(all_embeddings)
            
            for i, (start, end) in enumerate(vad_segments):
                speaker, similarity = speaker_results[i]
                if speaker is None:
                    continue
                
                segments_info.append({
                    'segment_id': i,
                    'start': float(start),
                    'end': float(end),
                    'duration': float(end - start),
                    'speaker': speaker,
                    'similarity': float(similarity),
                    'matched': bool(similarity >= self.similarity_threshold)
                })
            
            # 4. Filter and merge
            print("  - Filtering and merging | 过滤和合并...")
            matched = [s for s in segments_info if s['matched']]
            merged = self._merge_segments(matched)
            
            # 5. 按说话人分组保存
            speaker_stats = {}
            for seg in merged:
                speaker = seg['speaker']
                if speaker not in speaker_stats:
                    speaker_stats[speaker] = []
                speaker_stats[speaker].append(seg)
            
            print(f"    Identified speakers | 识别到说话人: {list(speaker_stats.keys())}")
            
            # 6. Async save segments (non-blocking)
            saved_files = {}
            for speaker, segs in speaker_stats.items():
                print(f"    - {speaker}: {len(segs)} segments | 个片段")
                
                # 创建说话人输出目录
                speaker_dir = self.output_dir / speaker / "segments"
                speaker_dir.mkdir(parents=True, exist_ok=True)
                
                saved_files[speaker] = []
                
                for seg in segs:
                    # 在文件名前添加相似度，格式为0.XX，便于按相似度排序
                    similarity_prefix = f"{seg['similarity']:.2f}"
                    filename = f"{similarity_prefix}_{audio_path.stem}_seg_{seg['segment_id']}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
                    output_path = speaker_dir / filename
                    
                    # 从原始音频裁切
                    start_sample = int(seg['start'] * sr_orig)
                    end_sample = int(seg['end'] * sr_orig)
                    
                    if audio_orig.ndim == 2:
                        segment = audio_orig[start_sample:end_sample, :].copy()
                    else:
                        segment = audio_orig[start_sample:end_sample].copy()
                    
                    # 异步保存
                    future = self._save_executor.submit(
                        self._save_segment_async, str(output_path), segment, sr_orig
                    )
                    self._pending_saves.append(future)
                    saved_files[speaker].append(filename)
            
            result = {
                'file': audio_path.name,
                'status': 'success',
                'total_segments': len(segments_info),
                'matched_segments': len(merged),
                'speakers': {sp: len(segs) for sp, segs in speaker_stats.items()},
                'segments': merged,
                'saved_files': saved_files,
                'audio_info': {
                    'sample_rate': int(sr_orig),
                    'channels': int(n_channels)
                }
            }
            
            # 异步保存元数据
            metadata_dir = self.output_dir / "metadata"
            metadata_dir.mkdir(exist_ok=True)
            metadata_file = metadata_dir / f"{audio_path.stem}.json"
            future = self._save_executor.submit(
                self._save_metadata_async, str(metadata_file), result
            )
            self._pending_saves.append(future)
            
            # 清理已完成的保存任务，防止内存累积（非阻塞）
            self._cleanup_completed_saves()
            
            # 定期清理GPU缓存（不等待IO，保持流水线）
            self._files_processed += 1
            if self._files_processed % self._gc_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"  ✗ Error | 错误: {e}")
            import traceback
            traceback.print_exc()
            return {'file': audio_path.name, 'status': 'error', 'error': str(e)}
    
    def process_dataset(self):
        """Process entire dataset with prefetch optimization | 处理整个数据集（支持预加载优化）"""
        audio_files = list(self.dataset_dir.glob("*.wav"))
        print(f"\nFound {len(audio_files)} audio files | 找到 {len(audio_files)} 个音频文件")
        
        results = []
        
        if self.prefetch_workers > 0 and len(audio_files) > 1:
            # Use prefetch mode
            print(f"Prefetch mode enabled (workers={self.prefetch_workers}) | 启用预加载模式")
            
            with ThreadPoolExecutor(max_workers=self.prefetch_workers) as executor:
                # 预提交前几个文件的预加载任务
                pending_futures = {}
                prefetch_idx = 0
                
                # 初始预加载
                for i in range(min(self.prefetch_workers + 1, len(audio_files))):
                    future = executor.submit(self._prefetch_file, audio_files[i])
                    pending_futures[future] = audio_files[i]
                    prefetch_idx = i + 1
                
                # 按顺序处理文件
                for i, audio_file in enumerate(audio_files):
                    # 等待当前文件的预加载完成
                    prefetched = None
                    for future in list(pending_futures.keys()):
                        if pending_futures[future] == audio_file:
                            prefetched = future.result()
                            del pending_futures[future]
                            break
                    
                    # 提交下一个预加载任务
                    if prefetch_idx < len(audio_files):
                        future = executor.submit(self._prefetch_file, audio_files[prefetch_idx])
                        pending_futures[future] = audio_files[prefetch_idx]
                        prefetch_idx += 1
                    
                    # 处理当前文件
                    result = self.process_file(audio_file, prefetched)
                    results.append(result)
        else:
            # 普通模式
            for audio_file in audio_files:
                result = self.process_file(audio_file)
                results.append(result)
        
        # Wait for all async saves to complete
        print("\nWaiting for file saves... | 等待文件保存完成...")
        self._wait_pending_saves()
        
        # Final GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate summary report
        summary = {
            'total_files': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'error']),
            'no_speech': len([r for r in results if r['status'] == 'no_speech']),
            'speakers': list(self.speaker_embeddings.keys()),
            'skipped_speakers': list(self.skip_info.keys()) if self.skip_info else [],
            'settings': {
                'similarity_threshold': self.similarity_threshold,
                'min_duration': self.min_duration,
                'merge_gap': self.merge_gap
            }
        }
        
        # 统计每个说话人的片段数（只统计实际处理的）
        speaker_counts = {sp: 0 for sp in self.speaker_embeddings.keys()}
        for r in results:
            if r['status'] == 'success' and 'speakers' in r:
                for sp, count in r['speakers'].items():
                    if sp in speaker_counts:  # 只统计处理过的speaker
                        speaker_counts[sp] += count
        
        summary['speaker_segments'] = speaker_counts
        
        # 保存汇总
        with open(self.output_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print("\n" + "="*60)
        print("Processing complete! | 处理完成!")
        print(f"Success | 成功: {summary['successful']}, Failed | 失败: {summary['failed']}, No speech | 无语音: {summary['no_speech']}")
        print("\nSpeaker segment statistics | 说话人片段统计:")
        for speaker, count in speaker_counts.items():
            print(f"  - {speaker}: {count} segments | 个片段")
        print(f"\nOutput directory | 输出目录: {self.output_dir}")
        print("="*60)
        
        return results


def main():
    """Main function | 主函数"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Multi-Target Speaker Extraction (MTSE) | 多目标说话人提取工具")
    parser.add_argument('--config', default='config.yaml', help='Config file path | 配置文件路径')
    parser.add_argument('--reference-dir', help='Reference audio directory | 参考音频根目录')
    parser.add_argument('--dataset-dir', help='Dataset directory | 数据集目录')
    parser.add_argument('--output-dir', help='Output directory | 输出目录')
    parser.add_argument('--threshold', type=float, help='Similarity threshold | 相似度阈值')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = Path(__file__).parent / args.config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 命令行参数覆盖配置文件
    reference_dir = args.reference_dir or config.get('paths', {}).get('reference_dir', './enrollment_audio')
    dataset_dir = args.dataset_dir or config.get('paths', {}).get('dataset_dir', './input_audio')
    output_dir = args.output_dir or config.get('paths', {}).get('output_dir', './output')
    threshold = args.threshold or config.get('verification', {}).get('similarity_threshold', 0.70)
    min_duration = config.get('verification', {}).get('min_duration', 0.5)
    merge_gap = config.get('verification', {}).get('merge_gap', 0.3)
    speaker_config = config.get('speaker_management', {})
    
    print("="*60)
    print("Multi-Target Speaker Extraction (MTSE)")
    print("多目标说话人提取工具")
    print("="*60)
    print(f"Reference audio | 参考音频: {reference_dir}")
    print(f"Dataset | 数据集: {dataset_dir}")
    print(f"Output | 输出: {output_dir}")
    print(f"Threshold | 阈值: {threshold}")
    
    # Display speaker management config
    if speaker_config.get('include_only'):
        print(f"Include only | 仅处理: {speaker_config['include_only']}")
    elif speaker_config.get('skip_speakers'):
        print(f"Skip | 跳过: {speaker_config['skip_speakers']}")
    
    print("="*60)
    
    system = MultiSpeakerVerification(
        reference_dir=reference_dir,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        similarity_threshold=threshold,
        min_duration=min_duration,
        merge_gap=merge_gap,
        speaker_config=speaker_config
    )
    
    system.process_dataset()


if __name__ == "__main__":
    main()



