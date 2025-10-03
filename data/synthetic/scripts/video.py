import os
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed


class Generator:

    def frames_to_video_ffmpeg(self, frame_dir, output_path, fps=25):
        """Convert image frames to video using FFmpeg."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        frame_pattern = os.path.join(frame_dir, "frame_%04d.png")

        if not os.path.exists(frame_dir):
            print(f"❌ Directory not found: {frame_dir}")
            return False
        import glob
        if not glob.glob(os.path.join(frame_dir, '*.png')):
            print(f"❌ No .png files found in {frame_dir}")
            return False

        # FFmpeg command to convert frames to video
        cmd = [
            "ffmpeg",
            "-y", 
            "-framerate", str(fps),
            "-start_number", "0",  
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            print(f"🎬 Converting: {frame_dir} -> {output_path}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Success: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"💥 Failed: {frame_dir}")
            print(f"   Reason: {e.stderr.strip()}")
            return False


    def parse_args(self):
        """Parse command line arguments for video generation."""
        parser = argparse.ArgumentParser(description='Convert frames to videos')

        parser.add_argument('--input_dir', type=str, default='./evaluation/data/double/basic/frames',
                            help='Input directory containing frames')
        parser.add_argument('--output_dir', type=str, default='./evaluation/data/double/basic/frames',
                            help='Output directory for videos')
        parser.add_argument('--scene', type=str, default='overdetermination',
                            help='Scene name')
        parser.add_argument('--workers', type=int, default=5,
                            help='Number of parallel processes')
        parser.add_argument('--fps', type=int, default=25,
                            help='Frames per second')
        parser.add_argument('--min_index', type=int, default=0,
                            help='Starting index')
        parser.add_argument('--max_index', type=int, default=100,
                            help='Ending index')

        return parser.parse_args()


def process_scene(scene, index, args, generator):
    """Process a single scene to convert frames to video."""
    frame_dir = os.path.join(args.input_dir, scene, f'{index:02d}')
    output_path = os.path.join(args.output_dir, scene, f'{index:02d}.mp4')
    return generator.frames_to_video_ffmpeg(frame_dir, output_path, fps=args.fps)


if __name__ == "__main__":
    a = Generator()
    args = a.parse_args()

    scene_list = [args.scene] 
    # scene_list = ["bogus", "double", "early", "late", "overdetermination", "switch"]

    # Process scenes in parallel using ProcessPoolExecutor
    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for scene in scene_list:
            for index in range(args.min_index, args.max_index):
                tasks.append(executor.submit(process_scene, scene, index, args, a))

        for future in as_completed(tasks):
            future.result()
