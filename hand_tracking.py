"""
HAND TRACKING PROTOTYPE - Single File Solution
Real-time hand tracking with virtual object interaction
No import errors - Everything in one file
"""

import cv2
import numpy as np
import time
from collections import deque

class VirtualObject:
    """Virtual object for collision detection"""
    def __init__(self, x=220, y=140, width=200, height=200, boundary=30):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.boundary = boundary
        self.color = (0, 255, 255)  # Yellow
        self.boundary_color = (0, 165, 255)  # Orange
        
        # Store original for reset
        self.original = (x, y, width, height)
    
    def draw(self, frame):
        """Draw virtual object on frame"""
        # Draw warning boundary
        cv2.rectangle(
            frame,
            (self.x - self.boundary, self.y - self.boundary),
            (self.x + self.width + self.boundary, 
             self.y + self.height + self.boundary),
            self.boundary_color,
            1
        )
        
        # Draw main object
        cv2.rectangle(
            frame,
            (self.x, self.y),
            (self.x + self.width, self.y + self.height),
            self.color,
            2
        )
        
        return frame
    
    def calculate_distance(self, point):
        """Calculate distance from hand to object"""
        if point is None:
            return float('inf')
        
        x, y = point
        ox, oy, ow, oh = self.x, self.y, self.width, self.height
        b = self.boundary
        
        # If inside object
        if ox <= x <= ox + ow and oy <= y <= oy + oh:
            return -1  # Inside object
        
        # If inside warning zone
        if (ox - b <= x <= ox + ow + b) and (oy - b <= y <= oy + oh + b):
            # Calculate how far inside warning zone
            dist_to_left = max(0, ox - x)
            dist_to_right = max(0, x - (ox + ow))
            dist_to_top = max(0, oy - y)
            dist_to_bottom = max(0, y - (oy + oh))
            
            # Return negative distance (inside warning zone)
            return -min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
        
        # Outside warning zone - calculate actual distance
        # Find closest point on rectangle
        closest_x = max(ox - b, min(x, ox + ow + b))
        closest_y = max(oy - b, min(y, oy + oh + b))
        
        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        return distance
    
    def reset(self):
        """Reset to original position"""
        self.x, self.y, self.width, self.height = self.original

class HandTracker:
    """Main hand tracking class"""
    def __init__(self, camera_id=0):
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Virtual object (center of screen)
        self.virtual_obj = VirtualObject(
            x=self.width//2 - 100,
            y=self.height//2 - 100,
            width=200,
            height=200,
            boundary=30
        )
        
        # State variables
        self.state = "SAFE"
        self.hand_position = None
        self.distance = float('inf')
        
        # Skin detection parameters
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Smoothing
        self.position_history = deque(maxlen=5)
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Debug mode
        self.debug = False
        
        print(f"Camera initialized: {self.width}x{self.height}")
        print("Virtual object at center of screen")
    
    def detect_hand(self, frame):
        """Detect hand using skin color segmentation"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Filter small contours (noise)
            if area > 1000:
                # Calculate centroid
                M = cv2.moments(largest)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy), mask, largest
        
        return None, mask, None
    
    def update_state(self, distance):
        """Update state based on distance"""
        if distance <= -1:
            return "DANGER"
        elif distance <= 0:
            return "WARNING"
        elif distance <= 50:
            return "WARNING"
        else:
            return "SAFE"
    
    def draw_state_panel(self, frame):
        """Draw state information"""
        # State colors
        colors = {
            "SAFE": (0, 255, 0),      # Green
            "WARNING": (0, 165, 255), # Orange
            "DANGER": (0, 0, 255)     # Red
        }
        
        color = colors.get(self.state, (255, 255, 255))
        
        # Draw panel
        cv2.rectangle(frame, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 100), color, 2)
        
        # State text
        cv2.putText(
            frame,
            f"STATE: {self.state}",
            (20, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            color,
            2
        )
        
        # Distance info
        if self.distance < float('inf'):
            if self.distance < 0:
                dist_text = f"INSIDE ({abs(self.distance):.0f}px)"
            else:
                dist_text = f"Distance: {self.distance:.0f}px"
            
            cv2.putText(
                frame,
                dist_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1
            )
        
        # Danger warning
        if self.state == "DANGER":
            cv2.putText(
                frame,
                "DANGER DANGER",
                (20, 95),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 0, 255),
                2
            )
            
            # Flash effect
            if int(time.time() * 3) % 2 == 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (self.width, self.height), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        
        return frame
    
    def draw_hand(self, frame, position, contour):
        """Draw hand visualization"""
        if position:
            x, y = position
            
            # Draw contour
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            
            # Draw center point
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
            
            # Draw coordinates
            cv2.putText(
                frame,
                f"({x}, {y})",
                (x + 20, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
    
    def draw_fps(self, frame):
        """Draw FPS counter"""
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (self.width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        return frame
    
    def draw_instructions(self, frame):
        """Draw instructions"""
        instructions = [
            "Move hand towards yellow box",
            "Controls: Q=Quit, R=Reset, D=Debug, S=Screenshot"
        ]
        
        y_start = self.height - 60
        for i, text in enumerate(instructions):
            y = y_start + (i * 25)
            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("\n" + "="*50)
        print("HAND TRACKING PROTOTYPE")
        print("="*50)
        print("Real-time hand tracking with virtual object interaction")
        print("\nCONTROLS:")
        print("  Q - Quit application")
        print("  R - Reset virtual object position")
        print("  D - Toggle debug mode (show skin mask)")
        print("  S - Save screenshot")
        print("\nMove your hand towards the yellow box to trigger warnings")
        print("="*50 + "\n")
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect hand
            hand_pos, mask, contour = self.detect_hand(frame)
            
            # Smooth hand position
            if hand_pos:
                self.position_history.append(hand_pos)
                if self.position_history:
                    avg_x = int(np.mean([p[0] for p in self.position_history]))
                    avg_y = int(np.mean([p[1] for p in self.position_history]))
                    hand_pos = (avg_x, avg_y)
                
                self.hand_position = hand_pos
                
                # Calculate distance to virtual object
                self.distance = self.virtual_obj.calculate_distance(hand_pos)
                
                # Update state
                self.state = self.update_state(self.distance)
            else:
                self.hand_position = None
                self.distance = float('inf')
                self.state = "SAFE"
            
            # Draw everything
            frame = self.virtual_obj.draw(frame)
            frame = self.draw_hand(frame, self.hand_position, contour)
            frame = self.draw_state_panel(frame)
            frame = self.draw_fps(frame)
            frame = self.draw_instructions(frame)
            
            # Update FPS
            self.update_fps()
            
            # Display main window
            cv2.imshow("Hand Tracking Prototype", frame)
            
            # Show debug window if enabled
            if self.debug and mask is not None:
                cv2.imshow("Debug - Skin Mask", mask)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("\nApplication stopped by user")
                break
            elif key == ord('r'):
                self.virtual_obj.reset()
                print("Virtual object reset to center")
            elif key == ord('d'):
                self.debug = not self.debug
                print(f"Debug mode: {'ON' if self.debug else 'OFF'}")
                if not self.debug:
                    cv2.destroyWindow("Debug - Skin Mask")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function"""
    print("Starting Hand Tracking Prototype...")
    
    try:
        # Create and run tracker
        tracker = HandTracker(camera_id=0)
        tracker.run()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure webcam is connected")
        print("2. Ensure good lighting conditions")
        print("3. Try camera_id=1 if you have multiple cameras")
        print("4. Make sure OpenCV is installed: pip install opencv-python")
    
    print("\nApplication closed")

if __name__ == "__main__":
    main()