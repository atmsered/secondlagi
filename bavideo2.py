import cv2
import numpy as np
import mediapipe as mp
import os
import time

class VideoDepthSegmenter:
    def __init__(self, num_layers=3, output_resolution=(800, 600)):
        """Initialize the video depth segmenter.
        
        Args:
            num_layers: Number of depth layers to generate
            output_resolution: Resolution of output frames (width, height)
        """
        self.num_layers = num_layers
        self.output_resolution = output_resolution
        
        # Initialize MediaPipe solutions
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.mp_pose = mp.solutions.pose
        
        # Create segmentation and pose detection models
        self.selfie_segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.pose_detection = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        # Initialize depth estimation model
        try:
            # Try several possible paths for the MiDaS model
            model_paths = [
                'MiDaS_small.onnx',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MiDaS_small.onnx'),
                os.path.expanduser('~/Downloads/MiDaS_small.onnx')
            ]
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    print(f"Loading MiDaS model from: {path}")
                    self.midas = cv2.dnn.readNetFromONNX(path)
                    model_loaded = True
                    break
            
            if not model_loaded:
                print("MiDaS model not found. Will use alternative depth estimation.")
                self.midas = None
        except Exception as e:
            print(f"Failed to load MiDaS model: {e}")
            self.midas = None
        
        print("Video Depth Segmenter initialized with", num_layers, "layers")
        
    def estimate_depth_map(self, frame):
        """Estimate depth map using MiDaS model or alternative methods."""
        if self.midas is not None:
            try:
                # Preprocess image for depth estimation
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (256, 256))
                img = img.astype(np.float32) / 255.0
                
                # Normalize image
                img = img.transpose((2, 0, 1))
                img = np.expand_dims(img, axis=0)
                
                # Get depth prediction
                self.midas.setInput(img)
                depth_map = self.midas.forward()
                
                # Post-process depth map
                depth_map = cv2.resize(depth_map[0, 0], (frame.shape[1], frame.shape[0]))
                depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
                
                return depth_map
            
            except Exception as e:
                print(f"Error in MiDaS depth estimation: {e}")
                # Fall back to alternative depth estimation
        
        # Alternative depth estimation using person distance and image features
        # This is a simplified approach when MiDaS is not available
        print("Using alternative depth estimation method")
        
        # Use grayscale gradient as a simple depth cue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Blur to smooth out the gradient
        gradient_magnitude = cv2.GaussianBlur(gradient_magnitude, (15, 15), 0)
        
        # Normalize to 0-1 range
        depth_map = cv2.normalize(gradient_magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # Invert so that closer objects have lower depth values
        depth_map = 1 - depth_map
        
        return depth_map
    
    def detect_person(self, frame):
        """Detect person using MediaPipe Selfie Segmentation."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        results = self.selfie_segmentation.process(rgb_frame)
        
        # Get segmentation mask (1 for person, 0 for background)
        person_mask = results.segmentation_mask
        if person_mask is None:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        # Convert mask to appropriate format
        person_mask = (person_mask > 0.5).astype(np.uint8) * 255
        return person_mask
    
    def segment_into_layers(self, frame):
        """Segment frame into multiple depth layers."""
        # Initialize layers
        layers = [np.zeros((*self.output_resolution[::-1], 4), dtype=np.uint8) for _ in range(self.num_layers)]
        
        # Resize frame to output resolution
        frame = cv2.resize(frame, self.output_resolution)
        
        # Get person mask
        person_mask = self.detect_person(frame)
        person_mask = cv2.resize(person_mask, self.output_resolution)
        
        # Get depth map
        try:
            depth_map = self.estimate_depth_map(frame)
            depth_map = cv2.resize(depth_map, self.output_resolution)
        except Exception as e:
            print(f"Error estimating depth: {e}")
            # If depth estimation fails, use basic segmentation
            depth_map = np.zeros(self.output_resolution[::-1], dtype=np.float32)
            depth_map[person_mask > 128] = 0  # Foreground (people)
            depth_map[person_mask <= 128] = 0.7  # Background
        
        # Create person layer (foreground)
        person_layer = np.zeros((*self.output_resolution[::-1], 4), dtype=np.uint8)
        person_layer[..., :3] = frame
        person_layer[..., 3] = person_mask
        
        # Separate background into multiple layers based on depth
        background_mask = 255 - person_mask
        background = frame.copy()
        
        for i in range(self.num_layers):
            if i == 0:  # Foreground layer (people)
                layers[i] = person_layer
            else:
                # Determine depth range for this layer
                depth_min = (i - 1) / (self.num_layers - 1)
                depth_max = i / (self.num_layers - 1)
                
                # Create mask for this depth range
                layer_mask = np.zeros_like(background_mask)
                condition = (depth_map >= depth_min) & (depth_map < depth_max) & (background_mask > 0)
                layer_mask[condition] = 255
                
                # Create layer with alpha channel
                layers[i][..., :3] = background
                layers[i][..., 3] = layer_mask
        
        return layers
    
    def process_video(self, input_path, output_dir='output_layers', fps=30):
        """Process video and save each layer as a separate video."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {width}x{height}, {original_fps} FPS, {total_frames} frames")
        
        # Create video writers for each layer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writers = []
        for i in range(self.num_layers):
            out_path = os.path.join(output_dir, f"layer_{i}.avi")
            writer = cv2.VideoWriter(out_path, fourcc, fps, self.output_resolution)
            writers.append(writer)
        
        # Process video frame by frame
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame and get layers
            layers = self.segment_into_layers(frame)
            
            # Write layers to output videos
            for i, layer in enumerate(layers):
                # Convert RGBA to BGR for video writing
                bgr_frame = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGR)
                writers[i].write(bgr_frame)
            
            # Progress update
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames ({fps:.2f} FPS)")
        
        # Release resources
        cap.release()
        for writer in writers:
            writer.release()
        
        print(f"Processed {frame_count} frames. Output saved to {output_dir}")
        return True
    
    def process_for_sliceview(self, input_path, output_path, mirror=True, keystone_correction=False):
        """Process video specifically for SliceView display.
        
        This follows the SliceView paper's recommendations for preparing content.
        """
        # Process video into layers
        self.process_video(input_path, output_dir='temp_layers')
        
        # Create a combined video with all layers side by side
        self.combine_layers_for_sliceview('temp_layers', output_path, mirror, keystone_correction)
        
        print(f"SliceView compatible video saved to {output_path}")
        return True
        
    def combine_layers_for_sliceview(self, layers_dir, output_path, mirror=True, keystone_correction=False):
        """Combine layer videos into a single video for SliceView.
        
        Based on the SliceView paper, this will:
        1. Mirror layers vertically if needed
        2. Apply keystone correction if needed
        3. Scale layers proportionally to their distance
        """
        # Get all layer videos
        layer_paths = [os.path.join(layers_dir, f"layer_{i}.avi") for i in range(self.num_layers)]
        
        # Open all layer videos
        caps = [cv2.VideoCapture(path) for path in layer_paths]
        if not all(cap.isOpened() for cap in caps):
            print("Error: Could not open all layer videos")
            return False
        
        # Get video properties
        width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = caps[0].get(cv2.CAP_PROP_FPS)
        
        # Calculate combined width based on number of layers
        combined_width = width * self.num_layers
        
        # Create video writer for combined output
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, height))
        
        while all(cap.isOpened() for cap in caps):
            frames = []
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if len(frames) != self.num_layers:
                break
                
            # Apply transformations to each layer
            for i, frame in enumerate(frames):
                # 1. Vertical mirroring if needed
                if mirror:
                    frame = cv2.flip(frame, 0)  # 0 for vertical flip
                
                # 2. Keystone correction if needed
                if keystone_correction:
                    # Simple keystone correction (more sophisticated methods exist)
                    src_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
                    dst_points = np.float32([[0, 0], [width, 0], [20, height], [width-20, height]])
                    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                    frame = cv2.warpPerspective(frame, matrix, (width, height))
                
                # 3. Scale proportionally to distance
                # The further the layer, the larger it should appear
                scale_factor = 1.0 + (i * 0.05)  # Increase size by 5% for each deeper layer
                if scale_factor != 1.0:
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    scaled = cv2.resize(frame, (new_width, new_height))
                    
                    # Crop to original size from center
                    start_x = (new_width - width) // 2
                    start_y = (new_height - height) // 2
                    frame = scaled[start_y:start_y+height, start_x:start_x+width]
                
                frames[i] = frame
            
            # Combine all frames horizontally
            combined_frame = np.hstack(frames)
            writer.write(combined_frame)
        
        # Release resources
        for cap in caps:
            cap.release()
        writer.release()
        
        return True

# Example usage
if __name__ == "__main__":
    # Create the segmenter
    segmenter = VideoDepthSegmenter(num_layers=3, output_resolution=(640, 480))
    
    # Process a video
    input_file = "input_video.mp4"
    
    # Check if file exists, otherwise try webcam
    if not os.path.exists(input_file):
        print(f"File {input_file} not found. Trying to use webcam (device 0)")
        input_file = 0  # Use webcam
        
        # For webcam, process frame by frame in real-time
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
        else:
            # Create windows for displaying layers
            cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
            for i in range(segmenter.num_layers):
                cv2.namedWindow(f"Layer {i}", cv2.WINDOW_NORMAL)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                layers = segmenter.segment_into_layers(frame)
                
                # Display original
                cv2.imshow("Original", frame)
                
                # Display layers
                for i, layer in enumerate(layers):
                    # Convert RGBA to BGR for display
                    bgr = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGR)
                    cv2.imshow(f"Layer {i}", bgr)
                
                # Break on ESC key
                if cv2.waitKey(1) == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
    else:
        # Process a video file
        output_path = "sliceview_output.avi"
        print(f"Processing video file: {input_file}")
        print(f"Output will be saved to: {output_path}")
        
        segmenter.process_for_sliceview(
            input_path=input_file,
            output_path=output_path,
            mirror=True,
            keystone_correction=True
        )