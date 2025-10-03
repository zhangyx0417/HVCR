# Physics Simulation Video Generation Pipeline

A comprehensive pipeline for generating high-quality Human-like Causal Reasoning videos using PyBullet for simulation and Blender for rendering.

## Project Overview

This project consists of three main components:
1. **simulation.py** - Generates physics simulation data using PyBullet
2. **render.py** - Renders high-quality animation frames using Blender
3. **gene_video.py** - Converts rendered frames into video files

## Supported Scene Types

- **Symthetic Overdetermination** - Multiple processes, all producing the same outcome, terminate at the same time.
- **Switch** - An event triggers one of two processes, both of which have the same outcome, making the event immaterial for the final result.
- **Early Preemption** -Two causal processes could produce the same outcome, but one terminates before the other even starts.
- **Late Preemption** - Two causal processes run in parallel; both would produce the same outcome, but one terminates before the other, rendering the latter irrelevant. 
- **Double Preemption** - A process that would have prevented another process is itself prevented by a different process.
- **Bogus Preemption** - An action is taken to interrupt a process that was never active.
## System Requirements

### Required Software
- **Python 3.7+**
- **Blender 4.2.3** - Download from [official website](https://www.blender.org/download/)
- **FFmpeg** - For video encoding
- **CUDA-compatible GPU** (recommended for accelerated rendering)

### Python Dependencies
See `requirements.txt` for complete list

## Installation Guide

1. **Clone the repository**
```bash
git clone <repository-url>
cd physics-simulation-pipeline
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Blender 4.2.3**
- Download and install Blender 4.2.3 from the official website
- Ensure Blender executable is in your system PATH or note the installation path

## Usage

### Step 1: Generate Simulation Data

```bash
python simulation.py --scene overdetermination --nsave 100 --output_dir simulation_output
```

**Parameters:**
- `--scene`: Scene type (overdetermination, switch, early, late, double, bogus, all)
- `--nsave`: Number of scenes to generate
- `--output_dir`: Output directory for simulation data
- `--extra_static_objects`: Number of additional static objects
- `--extra_moving_objects`: Number of additional moving objects
- `--seed`: Random seed for reproducibility

**Example commands for different scenarios:**
```bash
# Generate 50 overdetermination scenes
python simulation.py --scene overdetermination --nsave 50 --output_dir ./data/simulation

# Generate all scene types with extra objects
python simulation.py --scene all --nsave 20 --extra_static_objects 2 --extra_moving_objects 1

# Generate with specific seed for reproducibility
python simulation.py --scene switch --nsave 10 --seed 42
```

### Step 2: Render Animation Frames

```bash
blender --background --python render.py -- --scene overdetermination --input_dir simulation_output --output_dir render_output --begin 0 --end 99
```

**Parameters:**
- `--scene`: Scene type to render
- `--input_dir`: Input directory containing simulation data
- `--output_dir`: Output directory for rendered frames
- `--begin/--end`: Range of scene indices to render
- `--camera_distance_factor`: Camera distance multiplier (default: 1.5)
- `--camera_height_factor`: Camera height multiplier (default: 0.8)
- `--camera_angle`: Camera overhead angle in degrees (default: 30.0)

**Example rendering commands:**
```bash
# Render first 10 scenes with custom camera settings
blender --background --python render.py -- --scene overdetermination --input_dir ./data/simulation --output_dir ./data/render --begin 0 --end 9 --camera_distance_factor 2.0

# Render all scenes in a range
blender --background --python render.py -- --scene switch --input_dir ./data/simulation --output_dir ./data/render --begin 0 --end 49

# Render with closer camera view
blender --background --python render.py -- --scene early --input_dir ./data/simulation --output_dir ./data/render --begin 0 --end 19 --camera_distance_factor 1.2 --camera_angle 45.0
```

### Step 3: Generate Videos

```bash
python gene_video.py --input_dir render_output --output_dir videos --scene overdetermination --min_index 0 --max_index 99 --fps 25
```

**Parameters:**
- `--input_dir`: Input directory containing rendered frames
- `--output_dir`: Output directory for videos
- `--scene`: Scene type to process
- `--min_index/--max_index`: Range of scene indices to process
- `--fps`: Video frame rate
- `--workers`: Number of parallel processing workers

**Example video generation commands:**
```bash
# Generate videos with high frame rate
python gene_video.py --input_dir ./data/render --output_dir ./data/videos --scene overdetermination --min_index 0 --max_index 49 --fps 30

# Generate with parallel processing
python gene_video.py --input_dir ./data/render --output_dir ./data/videos --scene switch --min_index 0 --max_index 19 --workers 4

# Generate single video
python gene_video.py --input_dir ./data/render --output_dir ./data/videos --scene early --min_index 5 --max_index 5 --fps 25
```

## Complete Workflow Example

Here's a complete example to generate 10 overdetermination scenes from start to finish:

```bash
# Step 1: Generate simulation data
python simulation.py --scene overdetermination --nsave 10 --output_dir ./data/simulation

# Step 2: Render frames (make sure Blender is in PATH)
blender --background --python render.py -- --scene overdetermination --input_dir ./data/simulation --output_dir ./data/render --begin 0 --end 9

# Step 3: Generate videos
python gene_video.py --input_dir ./data/render --output_dir ./data/videos --scene overdetermination --min_index 0 --max_index 9 --fps 25
```

## Output File Structure

```
project/
├── simulation_output/          # Simulation data
│   └── overdetermination/
│       ├── annotation_00000.json
│       ├── annotation_00001.json
│       └── ...
├── render_output/              # Rendered frames
│   └── overdetermination/
│       ├── 00000/
│       │   ├── frame_0000.png
│       │   ├── frame_0001.png
│       │   └── ...
│       └── 00001/
└── videos/                     # Final videos
    └── overdetermination/
        ├── 00000.mp4
        ├── 00001.mp4
        └── ...
```

## Configuration Details

### Physics Simulation Parameters
- **Frame Rate**: 25 FPS
- **Duration**: 5 seconds (125 frames)
- **Gravity**: -9.81 m/s²
- **Materials**: 
  - Metal: High restitution (0.95), Low friction (0.3)
  - Rubber: High restitution (0.95), High friction (0.95)
- **Shapes**: Cube, Sphere, Cylinder
- **Colors**: Gray, Red, Blue, Green, Brown, Cyan, Purple, Yellow
- **Object Scale**: 0.167 units for all dimensions

### Rendering Settings
- **Resolution**: Configurable (default optimized for quality)
- **Render Engine**: Cycles with GPU acceleration
- **Samples**: 128 (configurable)
- **Denoising**: OPTIX denoiser when available
- **Tile Size**: 256px for optimal GPU utilization
- **Lighting**: HDRI environment lighting with additional area lights

### Material Properties
- **Metal Objects**: 
  - Metallic: 0.9
  - Roughness: 0.2
  - Base Color: Varies by object
- **Rubber Objects**:
  - Metallic: 0.0
  - Roughness: 0.7
  - Base Color: Varies by object

## Troubleshooting

### Common Issues

1. **Blender Not Found**
   ```bash
   # Check if Blender is in PATH
   blender --version
   
   # If not found, use full path
   /path/to/blender --background --python render.py -- [arguments]
   ```

2. **GPU Rendering Fails**
   - Open Blender GUI → Edit → Preferences → System → Cycles Render Devices
   - Enable CUDA or OpenCL devices
   - Verify GPU drivers are up to date

3. **Memory Issues**
   ```bash
   # Reduce parallel workers
   python gene_video.py --workers 1
   
   # Monitor memory usage
   python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')"
   ```

4. **FFmpeg Errors**
   ```bash
   # Test FFmpeg installation
   ffmpeg -version
   
   # Check frame files exist
   ls render_output/overdetermination/00000/
   
   # Manual video creation test
   ffmpeg -r 25 -i render_output/overdetermination/00000/frame_%04d.png -c:v libx264 test.mp4
   ```

5. **PyBullet Simulation Issues**
   - Check console output for collision warnings
   - Verify object positions don't overlap initially
   - Adjust physics solver parameters if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **PyBullet** - Excellent physics simulation library
- **Blender Foundation** - Outstanding open-source 3D creation suite
- **CLEVRER Dataset** - Inspiration for physics reasoning scenarios
- **OpenCV** - Computer vision and video processing capabilities
- **FFmpeg** - Video encoding and processing

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{HVCR,
  title={HVCR: Causal Evaluation of Large Multimodal Models in Human-like Video Reasoning},
  author={Yanxi Zhang, Junjie Li, Dongyan Zhao, Chaochao Lu},
  year={2025},
  url={https://github.com/zhangyx0417/VideoAC}
}
```

## Contact and Support

- **Issues**: Please report bugs and feature requests through GitHub Issues
- **Discussions**: Use GitHub Discussions for questions and community support
- **Email**: zhangyx0417@gmail.com

For more detailed information about specific components, please refer to the inline documentation in each Python file or contact the development team.