import numpy as np
import soundfile as sf
from pystoi import stoi
import librosa

class AudioQualityEvaluator:
    def __init__(self, target_sr=16000, align_method='dynamic'):
        """
        参数说明：
        target_sr: 统一采样率（STOI标准建议16kHz）
        align_method: 对齐方式 ['dynamic', 'truncate', 'pad']
        """
        self.target_sr = target_sr
        self.align_method = align_method

    def _load_audio(self, path):
        """加载并预处理音频"""
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        return audio

    def _align_audios(self, ref, deg):
        """音频对齐策略"""
        len_diff = len(ref) - len(deg)
        
        if self.align_method == 'dynamic':
            # 动态时间规整对齐
            ref, deg = self._dtw_alignment(ref, deg)
        elif self.align_method == 'truncate':
            # 截断较长音频
            min_len = min(len(ref), len(deg))
            ref = ref[:min_len]
            deg = deg[:min_len]
        elif self.align_method == 'pad':
            # 填充较短音频
            if len_diff > 0:
                deg = np.pad(deg, (0, len_diff), mode='constant')
            else:
                ref = np.pad(ref, (0, -len_diff), mode='constant')
        return ref, deg

    def _dtw_alignment(self, ref, deg):
        """基于动态时间规整的精确对齐"""
        from dtw import dtw
        alignment = dtw(ref, deg, keep_internals=True)
        return ref[alignment.index1], deg[alignment.index2]

    def calculate_stoi(self, ref_path, deg_path):
        """
        计算单对音频的STOI
        返回：
        stoi_score: 可懂度分数（0-1）
        aligned_ratio: 实际用于计算的数据比例
        """
        # 加载音频
        ref_audio = self._load_audio(ref_path)
        deg_audio = self._load_audio(deg_path)

        # 幅度归一化
        ref_audio = ref_audio / np.max(np.abs(ref_audio)) + 1e-8
        deg_audio = deg_audio / np.max(np.abs(deg_audio)) + 1e-8

        # 执行对齐
        ref_aligned, deg_aligned = self._align_audios(ref_audio, deg_audio)
        
        # 计算有效数据比例
        aligned_ratio = min(len(ref_audio), len(deg_audio)) / max(len(ref_audio), len(deg_audio))
        
        # 计算STOI
        return stoi(ref_aligned, deg_aligned, self.target_sr), aligned_ratio

    def batch_stoi(self, ref_paths, deg_paths):
        """批量计算STOI"""
        scores = []
        ratios = []
        for ref_p, deg_p in zip(ref_paths, deg_paths):
            score, ratio = self.calculate_stoi(ref_p, deg_p)
            scores.append(score)
            ratios.append(ratio)
        return {
            'mean_stoi': np.mean(scores),
            'std_stoi': np.std(scores),
            'min_stoi': np.min(scores),
            'max_stoi': np.max(scores),
            'alignment_ratio': np.mean(ratios)
        }

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 初始化评估器
    evaluator = AudioQualityEvaluator(target_sr=16000, align_method='dynamic')

    # 单文件测试
    ref_path = "original.wav"
    gen_path = "generated.wav"
    stoi_score, align_ratio = evaluator.calculate_stoi(ref_path, gen_path)
    print(f"STOI: {stoi_score:.3f}, 有效数据比例: {align_ratio:.1%}")

    # 批量评估
    ref_list = ["ref1.wav", "ref2.wav", "ref3.wav"]
    gen_list = ["gen1.wav", "gen2.wav", "gen3.wav"]
    batch_result = evaluator.batch_stoi(ref_list, gen_list)
    print(f"""
    批量评估结果：
    平均STOI: {batch_result['mean_stoi']:.3f} ± {batch_result['std_stoi']:.3f}
    波动范围: [{batch_result['min_stoi']:.3f}, {batch_result['max_stoi']:.3f}]
    平均对齐率: {batch_result['alignment_ratio']:.1%}
    """)