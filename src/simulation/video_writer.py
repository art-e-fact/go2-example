from pathlib import Path
import av
import numpy as np
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

class VideoWriter:
    """Simple class to write RGB frames from Isaac camera to a video file."""

    def __init__(self, output_path: str | Path, width=270, height=180, framerate=20,
                 camera_path="/World/Go2/Head_upper/camera", codec='libx264'):
        """Initialize video writer with Isaac camera.
        
        Args:
            output_path: Path where to save the video file
            width: Frame width in pixels
            height: Frame height in pixels
            framerate: Frames per second
            camera_path: Path to the camera prim in Isaac
            codec: Video codec to use
        """
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.framerate = framerate

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        av.logging.set_level(av.logging.ERROR)
        
        # Create container and stream
        self.container = av.open(output_path, 'w')
        self.stream = self.container.add_stream(codec, rate=framerate)
        self.stream.width = width
        self.stream.height = height
        self.stream.max_b_frames = 0  # Disable B-frames for lower latency
        
        # Initialize Isaac camera
        self.camera = Camera(
            prim_path=camera_path,
            translation=np.array([0.04, 0.0, 0.021]),
            frequency=framerate,
            resolution=(width, height),
            orientation=rot_utils.euler_angles_to_quats(
                np.array([0, 0, 0]), degrees=True
            ),
        )
        
        # Configure camera properties
        self.camera.set_focal_length(3.0)
        self.camera.set_clipping_range(0.01, 1000000000.0)
        
        # Frame counter
        self.frame_count = 0
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the camera."""
        self.camera.initialize()
        self.is_initialized = True
        print(f"Camera initialized at resolution {self.width}x{self.height}")
        
    def get_camera_rgb(self):
        """Get RGB frame from the camera."""
        if not self.is_initialized:
            self.initialize()
            
        rgba = self.camera.get_rgba()
        if rgba is None:
            print("Warning: No image received from camera")
            return None
        if len(rgba.shape) != 3:
            print(f"Warning: Image has unexpected shape {rgba.shape}, expected (H,W,3)")
            return None

        rgb = rgba[:, :, :3]
        return rgb
    
    def capture_frame(self):
        """Capture a frame from the camera and add it to the video."""
        rgb_frame = self.get_camera_rgb()
        if rgb_frame is not None:
            self.add_frame(rgb_frame)
            return True
        return False
    
    def add_frame(self, rgb_frame):
        """Add a single RGB frame to the video.
        
        Args:
            rgb_frame: RGB numpy array with shape (height, width, 3)
        """
        if rgb_frame is None:
            return
            
        # Ensure frame has correct dimensions
        if rgb_frame.shape != (self.height, self.width, 3):
            print(f"Warning: Frame has shape {rgb_frame.shape}, expected ({self.height}, {self.width}, 3)")
            # You could resize here if needed
            
        # Convert numpy array to VideoFrame
        frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
        
        # Encode frame
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
            
        self.frame_count += 1
    
    def close(self):
        """Finish encoding and close the video file."""
        # Flush any remaining frames
        for packet in self.stream.encode():
            self.container.mux(packet)
            
        # Close the container
        self.container.close()
        print(f"Video saved to {self.output_path} ({self.frame_count} frames)")