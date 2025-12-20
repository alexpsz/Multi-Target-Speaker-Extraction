"""
Speaker Manager - Manual control of which speakers to process
Speaker管理器 - 手动控制要处理的speaker
"""

from typing import Dict, List


class SpeakerManager:
    """Simple speaker filtering manager | 简单的speaker过滤管理器"""
    
    @staticmethod
    def get_speakers_to_process(
        all_speakers: List[str],
        config: Dict
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Get list of speakers to process based on config
        根据配置获取需要处理的speaker列表
        
        Args:
            all_speakers: List of all available speakers | 所有可用的speaker列表
            config: speaker_management configuration | speaker_management配置
        
        Returns:
            (speakers_to_process: List[str], skip_info: Dict[str, str])
        """
        speakers_to_process = []
        skip_info = {}
        
        # include_only has highest priority
        include_only = config.get('include_only', [])
        if include_only:
            for speaker in all_speakers:
                if speaker in include_only:
                    speakers_to_process.append(speaker)
                else:
                    skip_info[speaker] = "Not in include_only list | 不在include_only列表中"
            return speakers_to_process, skip_info
        
        # Use skip_speakers
        skip_speakers = config.get('skip_speakers', [])
        for speaker in all_speakers:
            if speaker in skip_speakers:
                skip_info[speaker] = "In skip_speakers list | 在skip_speakers列表中"
            else:
                speakers_to_process.append(speaker)
        
        return speakers_to_process, skip_info



