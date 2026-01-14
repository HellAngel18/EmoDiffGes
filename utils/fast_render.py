import os
import time
import numpy as np
import pyrender
import trimesh
import queue
import imageio
import threading
import multiprocessing
import utils.media
import glob
from loguru import logger

def deg_to_rad(degrees):
    return degrees * np.pi / 180

def create_pose_camera(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 1.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_pose_light(angle_deg):
    angle_rad = deg_to_rad(angle_deg)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [0.0, np.sin(angle_rad), np.cos(angle_rad), 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light):
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=uniform_color)
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.OrthographicCamera(xmag=1.0, ymag=1.0)
    scene.add(camera, pose=pose_camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0)
    scene.add(light, pose=pose_light)
    return scene

def do_render_one_frame(renderer, frame_idx, vertices, vertices1, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')

    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)

    figs = []
    for vtx in [vertices, vertices1]:
        # print(vtx.shape)
        scene = create_scene_with_mesh(vtx, faces, uniform_color, pose_camera, pose_light)
        fig, _ = renderer.render(scene)
        figs.append(fig)
  
    return figs[0], figs[1]

def do_render_one_frame_no_gt(renderer, frame_idx, vertices, faces):
    if frame_idx % 100 == 0:
        print('processed', frame_idx, 'frames')

    uniform_color = [220, 220, 220, 255]
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)

    figs = []
    # for vtx in [vertices]:
    #     print(vtx.shape)
    # print(vertices.shape)
    scene = create_scene_with_mesh(vertices, faces, uniform_color, pose_camera, pose_light)
    fig, _ = renderer.render(scene)
    figs.append(fig)
  
    return figs[0]

def write_images_from_queue(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1, fig2 = e
        filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        merged_fig = np.hstack((fig1, fig2))
        try:
            imageio.imwrite(filename, merged_fig)
        except Exception as ex:
            print(f"Error writing image {filename}: {ex}")
            raise ex
        
def write_images_from_queue_no_gt(fig_queue, output_dir, img_filetype):
    while True:
        e = fig_queue.get()
        if e is None:
            break
        fid, fig1 = e
        filename = os.path.join(output_dir, f"frame_{fid}.{img_filetype}")
        try:
            imageio.imwrite(filename, fig1)
            logger.debug(f"Wrote image: {filename}")
        except Exception as ex:
            logger.error(f"Error writing image {filename}: {ex}")
            raise
    
def render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width // 2, render_height)
    renderer = pyrender.OffscreenRenderer(*fig_resolution)

    for idx, fid in enumerate(fids):
        fig1, fig2 = do_render_one_frame(renderer, fid, frame_vertex_pairs[idx][0], frame_vertex_pairs[idx][1], faces)
        fig_queue.put((fid, fig1, fig2))
    
    renderer.delete()

def render_frames_and_enqueue_no_gt(fids, frame_vertices, faces, render_width, render_height, fig_queue):
    fig_resolution = (render_width, render_height)  # 使用完整分辨率
    try:
        renderer = pyrender.OffscreenRenderer(viewport_width=fig_resolution[0], viewport_height=fig_resolution[1])
        if renderer is None:
            raise RuntimeError("Renderer initialization returned None")
        logger.debug(f"Renderer initialized with resolution: {fig_resolution}")
    except Exception as e:
        logger.error(f"Failed to initialize OffscreenRenderer: {str(e)}")
        raise RuntimeError(f"Cannot initialize renderer: {str(e)}")

    for idx, fid in enumerate(fids):
        try:
            fig1 = do_render_one_frame_no_gt(renderer, fid, frame_vertices[idx], faces)
            if fig1 is None:
                logger.error(f"Rendering failed for frame {fid}: Returned None")
                raise ValueError(f"Render returned None for frame {fid}")
            fig_queue.put((fid, fig1))
            logger.debug(f"Enqueued frame {fid}")
        except Exception as e:
            logger.error(f"Failed to render frame {fid}: {str(e)}")
            raise
    renderer.delete()

def sub_process_process_frame(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertex_pairs, faces, output_dir):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    fig_queue = queue.Queue()
    render_frames_and_enqueue(fids, frame_vertex_pairs, faces, render_video_width, render_video_height, fig_queue)
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(target=write_images_from_queue, args=(fig_queue, output_dir, render_tmp_img_filetype))
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )
    
def sub_process_process_frame_no_gt(subprocess_index, render_video_width, render_video_height, render_tmp_img_filetype, fids, frame_vertices, faces, output_dir):
    begin_ts = time.time()
    print(f"subprocess_index={subprocess_index} begin_ts={begin_ts}")

    logger.debug(f"Subprocess {subprocess_index}: Frame IDs: {len(fids)}, Vertices shape: {frame_vertices.shape}")
    logger.debug(f"Subprocess {subprocess_index}: PYOPENGL_PLATFORM: {os.getenv('PYOPENGL_PLATFORM')}")

    fig_queue = queue.Queue()
    try:
        render_frames_and_enqueue_no_gt(fids, frame_vertices, faces, render_video_width, render_video_height, fig_queue)
        logger.debug(f"Subprocess {subprocess_index}: Render completed")
    except Exception as e:
        logger.error(f"Subprocess {subprocess_index}: Render failed: {str(e)}")
        raise
    fig_queue.put(None)
    render_end_ts = time.time()

    image_writer_thread = threading.Thread(target=write_images_from_queue_no_gt, args=(fig_queue, output_dir, render_tmp_img_filetype))
    image_writer_thread.start()
    image_writer_thread.join()

    write_end_ts = time.time()
    print(
        f"subprocess_index={subprocess_index} "
        f"render={render_end_ts - begin_ts:.2f} "
        f"all={write_end_ts - begin_ts:.2f} "
        f"begin_ts={begin_ts:.2f} "
        f"render_end_ts={render_end_ts:.2f} "
        f"write_end_ts={write_end_ts:.2f}"
    )

def distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append((vertices_all[i], vertices1_all[i]))
        sampled_frame_id += 1

    return subproc_frame_ids, subproc_vertices

def distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all):
    sample_interval = max(1, int(30 // render_video_fps))
    subproc_frame_ids = [[] for _ in range(render_concurent_nums)]
    subproc_vertices = [[] for _ in range(render_concurent_nums)]
    sampled_frame_id = 0

    for i in range(frames):
        if i % sample_interval != 0:
            continue
        subprocess_index = sampled_frame_id % render_concurent_nums
        subproc_frame_ids[subprocess_index].append(sampled_frame_id)
        subproc_vertices[subprocess_index].append(vertices_all[i])  # 仅存储单帧顶点数组
        sampled_frame_id += 1

    # 将每组顶点列表转换为 NumPy 数组
    subproc_vertices = [np.array(v) for v in subproc_vertices]
    logger.debug(f"Subproc vertices shapes: {[v.shape for v in subproc_vertices]}")
    return subproc_frame_ids, subproc_vertices

def generate_silent_videos(render_video_fps, 
                           render_video_width,
                           render_video_height,
                           render_concurent_nums,
                           render_tmp_img_filetype,
                           frames, 
                           vertices_all,
                           vertices1_all,
                           faces,
                           output_dir):

    subproc_frame_ids, subproc_vertices = distribute_frames(frames, render_video_fps, render_concurent_nums, vertices_all, vertices1_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame, 
            [
                (subprocess_index,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[subprocess_index],  subproc_vertices[subprocess_index], faces, output_dir) 
                    for subprocess_index in range(render_concurent_nums)
            ]
        )

    output_file = os.path.join(output_dir, "silence_video.mp4")
    utils.media.convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"), output_file, render_video_fps)
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file

'''def generate_silent_videos_no_gt(render_video_fps, 
                           render_video_width,
                           render_video_height,
                           render_concurent_nums,
                           render_tmp_img_filetype,
                           frames, 
                           vertices_all,
                           faces,
                           output_dir):

    subproc_frame_ids, subproc_vertices = distribute_frames_no_gt(frames, render_video_fps, render_concurent_nums, vertices_all)

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    #sub_process_process_frame_no_gt(0,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[0],  subproc_vertices[0], faces, output_dir) 
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame_no_gt, 
            [
                (subprocess_index,  render_video_width, render_video_height, render_tmp_img_filetype, subproc_frame_ids[subprocess_index],  subproc_vertices[subprocess_index], faces, output_dir) 
                    for subprocess_index in range(render_concurent_nums)
            ]
        )

    output_file = os.path.join(output_dir, "silence_video.mp4")
    utils.media.convert_img_to_mp4(os.path.join(output_dir, f"frame_%d.{render_tmp_img_filetype}"), output_file, render_video_fps)
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        os.remove(filename)

    return output_file'''

def generate_silent_videos_no_gt(
    render_video_fps,
    render_video_width,
    render_video_height,
    render_concurent_nums,
    render_tmp_img_filetype,
    frames,
    vertices_all,
    faces,
    output_dir
):
    import moviepy.config as mp_config
    import shutil

    # 验证 FFmpeg
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.error("FFmpeg not found in PATH")
        raise FileNotFoundError("FFmpeg not found")
    mp_config.FFMPEG_BINARY = ffmpeg_path
    logger.debug(f"Using FFmpeg: {ffmpeg_path}")

    # 验证输入
    logger.debug(f"Input params: fps={render_video_fps}, width={render_video_width}, height={render_video_height}, concurrent={render_concurent_nums}, filetype={render_tmp_img_filetype}, frames={frames}, output_dir={output_dir}")
    logger.debug(f"Vertices shape: {vertices_all.shape}, Faces shape: {faces.shape}")

    # 分帧
    subproc_frame_ids, subproc_vertices = distribute_frames_no_gt(
        frames, render_video_fps, render_concurent_nums, vertices_all
    )
    logger.debug(f"Subproc frame IDs: {subproc_frame_ids}")
    logger.debug(f"Subproc vertices types: {[type(v) for v in subproc_vertices]}")
    logger.debug(f"Subproc vertices shapes: {[v.shape if hasattr(v, 'shape') else 'no shape' for v in subproc_vertices]}")

    print(f"generate_silent_videos concurrentNum={render_concurent_nums} time={time.time()}")
    # 多进程渲染
    with multiprocessing.Pool(render_concurent_nums) as pool:
        pool.starmap(
            sub_process_process_frame_no_gt,
            [
                (
                    subprocess_index,
                    render_video_width,
                    render_video_height,
                    render_tmp_img_filetype,
                    subproc_frame_ids[subprocess_index],
                    subproc_vertices[subprocess_index],
                    faces,
                    output_dir
                )
                for subprocess_index in range(render_concurent_nums)
            ]
        )

    # 验证图像文件
    img_pattern = os.path.join(output_dir, f"frame_*.{render_tmp_img_filetype}")
    img_files = glob.glob(img_pattern)
    if not img_files:
        logger.error(f"No image files found at {img_pattern}")
        raise FileNotFoundError("No rendered images found")
    logger.debug(f"Found {len(img_files)} image files")

    # 转换为 MP4
    output_file = os.path.join(output_dir, "silence_video.mp4")
    try:
        utils.media.convert_img_to_mp4(img_pattern, output_file, render_video_fps)
        logger.info(f"Generated video: {output_file}")
    except Exception as e:
        logger.error(f"Failed to convert images to MP4: {str(e)}")
        raise

    # 清理临时文件
    filenames = glob.glob(os.path.join(output_dir, f"*.{render_tmp_img_filetype}"))
    for filename in filenames:
        try:
            os.remove(filename)
        except Exception as e:
            logger.warning(f"Failed to remove {filename}: {str(e)}")

    return output_file