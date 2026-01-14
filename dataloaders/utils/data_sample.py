import math
import numpy as np
from collections import defaultdict
from loguru import logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


def sample_from_clip(
    lmdb_manager, audio_file, audio_each_file, raw_audio_each_file, pose_each_file, trans_each_file, 
    trans_v_each_file, shape_each_file, facial_each_file, word_each_file,
    vid_each_file, emo_each_file, sem_each_file, args, ori_stride, ori_length,
    disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
    n_out_samples):
    """Sample clips from the data according to specified parameters."""
    
    round_seconds_skeleton = pose_each_file.shape[0] // args.pose_fps

    # Calculate timing information  计算audio、pose、facial的时间（秒），取最短的
    timing_info = calculate_timing_info(
        audio_each_file, facial_each_file, round_seconds_skeleton, 
        args.audio_fps, args.pose_fps, args.audio_sr, args.audio_rep
    )
    
    round_seconds_skeleton = timing_info['final_seconds']
    
    # Calculate clip boundaries    audio、pose的起始采样数、帧数
    clip_info = calculate_clip_boundaries(
        round_seconds_skeleton, clean_first_seconds, clean_final_seconds,
        args.audio_fps, args.pose_fps
    )
    
    n_filtered_out = defaultdict(int)
    
    # Process each training length ratio
    for ratio in args.multi_length_training:
        processed_data = process_data_with_ratio(
            ori_stride, ori_length, ratio, clip_info, args, is_test,
            audio_each_file, raw_audio_each_file, pose_each_file, trans_each_file, trans_v_each_file,
            shape_each_file, facial_each_file, word_each_file, vid_each_file,
            emo_each_file, sem_each_file, audio_file,
            lmdb_manager, n_out_samples
        )
        
        for type_key, count in processed_data['filtered_counts'].items():
            n_filtered_out[type_key] += count
            
        n_out_samples = processed_data['n_out_samples']
    
    return n_filtered_out, n_out_samples

def calculate_timing_info(audio_data, facial_data, round_seconds_skeleton, 
                         audio_fps, pose_fps, audio_sr, audio_rep):
    """Calculate timing information for the data."""
    if audio_data is not None:
        if audio_rep != "wave16k":
            round_seconds_audio = len(audio_data) // audio_fps
        elif audio_rep == "mfcc":
            round_seconds_audio = audio_data.shape[0] // audio_fps
        else:
            round_seconds_audio = audio_data.shape[0] // audio_sr
            
        if facial_data is not None:
            round_seconds_facial = facial_data.shape[0] // pose_fps
            logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
            final_seconds = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
            max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
            if final_seconds != max_round:
                logger.warning(f"reduce to {final_seconds}s, ignore {max_round-final_seconds}s")
        else:
            logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
            final_seconds = min(round_seconds_audio, round_seconds_skeleton)
            max_round = max(round_seconds_audio, round_seconds_skeleton)
            if final_seconds != max_round:
                logger.warning(f"reduce to {final_seconds}s, ignore {max_round-final_seconds}s")
    else:
        final_seconds = round_seconds_skeleton
        
    return {
        'final_seconds': final_seconds
    }

def calculate_clip_boundaries(round_seconds, clean_first_seconds, clean_final_seconds,
                            audio_fps, pose_fps):
    """Calculate the boundaries for clip sampling."""
    clip_s_t = clean_first_seconds
    clip_e_t = round_seconds - clean_final_seconds
    
    return {
        'clip_s_t': clip_s_t,
        'clip_e_t': clip_e_t,
        'clip_s_f_audio': audio_fps * clip_s_t,
        'clip_e_f_audio': clip_e_t * audio_fps,
        'clip_s_f_pose': clip_s_t * pose_fps,
        'clip_e_f_pose': clip_e_t * pose_fps
    }

def process_data_with_ratio(ori_stride, ori_length, ratio, clip_info, args, is_test,
                           audio_data, raw_audio_data, pose_data, trans_data, trans_v_data,
                           shape_data, facial_data, word_data, vid_data,
                           emo_data, sem_data, audio_file,
                           lmdb_manager, n_out_samples):
    """Process data with a specific training length ratio."""
    
    if is_test:
        cut_length = clip_info['clip_e_f_pose'] - clip_info['clip_s_f_pose']
        args.stride = cut_length
        max_length = cut_length
    else:
        args.stride = int(ratio * ori_stride)
        cut_length = int(ori_length * ratio)
        
    num_subdivision = math.floor(
        (clip_info['clip_e_f_pose'] - clip_info['clip_s_f_pose'] - cut_length) / args.stride
    ) + 1
    
    logger.info(f"pose from frame {clip_info['clip_s_f_pose']} to {clip_info['clip_e_f_pose']}, length {cut_length}")
    logger.info(f"{num_subdivision} clips is expected with stride {args.stride}")
    
    if audio_data is not None:
        audio_short_length = math.floor(cut_length / args.pose_fps * args.audio_fps)
        logger.info(f"audio from frame {clip_info['clip_s_f_audio']} to {clip_info['clip_e_f_audio']}, length {audio_short_length}")
    
    # Process subdivisions
    filtered_counts = defaultdict(int)
    for i in range(num_subdivision):
        sample_data = extract_sample_data(
            i, clip_info, cut_length, args,
            audio_data, raw_audio_data, pose_data, trans_data, trans_v_data,
            shape_data, facial_data, word_data, vid_data,
            emo_data, sem_data,  audio_file,
            audio_short_length
        )
        
        if sample_data['pose'].any() is not None:
            lmdb_manager.add_sample([
                sample_data['pose'], sample_data['audio'], sample_data['emotion'],sample_data['facial'],
                sample_data['shape'], sample_data['word'], sample_data['emo'],
                sample_data['sem'], sample_data['vid'], sample_data['trans'],
                sample_data['trans_v'], sample_data['audio_name']
            ])
            n_out_samples += 1
    
    return {
        'filtered_counts': filtered_counts,
        'n_out_samples': n_out_samples
    }

def extract_sample_data(idx, clip_info, cut_length, args,
                       audio_data, raw_audio_data, pose_data, trans_data, trans_v_data,
                       shape_data, facial_data, word_data, vid_data,
                       emo_data, sem_data, audio_file,
                       audio_short_length):
    """Extract a single sample from the data."""
    start_idx = clip_info['clip_s_f_pose'] + idx * args.stride
    fin_idx = start_idx + cut_length
    
    sample_data = {
        'pose': pose_data[start_idx:fin_idx],
        'trans': trans_data[start_idx:fin_idx],
        'trans_v': trans_v_data[start_idx:fin_idx],
        'shape': shape_data[start_idx:fin_idx],
        'facial': facial_data[start_idx:fin_idx] if args.facial_rep is not None else np.array([-1]),
        'word': word_data[start_idx:fin_idx] if args.word_rep is not None else np.array([-1]),
        'emo': emo_data[start_idx:fin_idx] if args.emo_rep is not None else np.array([-1]),
        'sem': sem_data[start_idx:fin_idx] if args.sem_rep is not None else np.array([-1]),
        
        'vid': vid_data[start_idx:fin_idx] if args.id_rep is not None else np.array([-1]),
        'audio_name': audio_file
    }
    
    if audio_data is not None:
        audio_start = clip_info['clip_s_f_audio'] + math.floor(idx * args.stride * args.audio_fps / args.pose_fps)
        audio_end = audio_start + audio_short_length
        sample_data['audio'] = audio_data[audio_start:audio_end]
    else:
        sample_data['audio'] = np.array([-1])

    if raw_audio_data is not None:
        sample_rate = raw_audio_data.frame_rate
        start_ms = audio_start * 1000 / sample_rate
        end_ms = audio_end * 1000 / sample_rate
        raw = raw_audio_data[start_ms:end_ms]
        raw.export("/sata/public/lixr/EmoDiffGes/save_emo/temp.wav", format="wav")
        sample_data['emotion'] = inference_pipeline('/sata/public/lixr/EmoDiffGes/save_emo/temp.wav', output_dir="./save_emo", granularity="utterance", extract_embedding=True)
    
    return sample_data