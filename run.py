"""
Multi-Target Speaker Extraction (MTSE) - Entry Point
多目标说话人提取工具 - 启动脚本
"""

import yaml
from pathlib import Path
from speaker_verification import MultiSpeakerVerification


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration file | 加载配置文件"""
    script_dir = Path(__file__).parent
    config_file = script_dir / config_path
    
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main function | 主函数"""
    print("="*60)
    print("Multi-Target Speaker Extraction (MTSE)")
    print("多目标说话人提取工具")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Parse paths
    script_dir = Path(__file__).parent
    reference_dir = script_dir / config['paths']['reference_dir']
    dataset_dir = script_dir / config['paths']['dataset_dir']
    output_dir = script_dir / config['paths']['output_dir']
    
    # Performance config
    perf_config = config.get('performance', {})
    batch_size = perf_config.get('batch_size', 8)
    prefetch_workers = perf_config.get('prefetch_workers', 2)
    
    print(f"Reference audio | 参考音频: {reference_dir}")
    print(f"Dataset | 数据集: {dataset_dir}")
    print(f"Output | 输出: {output_dir}")
    print(f"Similarity threshold | 相似度阈值: {config['verification']['similarity_threshold']}")
    print(f"Batch size | 批量大小: {batch_size}")
    print(f"Prefetch workers | 预加载线程: {prefetch_workers}")
    
    # Display speaker management config
    speaker_config = config.get('speaker_management', {})
    if speaker_config.get('include_only'):
        print(f"Include only | 仅处理speaker: {speaker_config['include_only']}")
    elif speaker_config.get('skip_speakers'):
        print(f"Skip | 跳过speaker: {speaker_config['skip_speakers']}")
    
    print("="*60 + "\n")
    
    # Create system instance
    system = MultiSpeakerVerification(
        reference_dir=str(reference_dir),
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        similarity_threshold=config['verification']['similarity_threshold'],
        min_duration=config['verification']['min_duration'],
        merge_gap=config['verification']['merge_gap'],
        speaker_config=speaker_config,
        batch_size=batch_size,
        prefetch_workers=prefetch_workers
    )
    
    # Process dataset
    system.process_dataset()


if __name__ == "__main__":
    main()



