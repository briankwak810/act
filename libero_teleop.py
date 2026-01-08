"""
Teleoperation module for LIBERO robot control via keyboard input.
Allows human intervention during on-screen rendering.
"""

import numpy as np
import threading
import time
from collections import deque
import sys

try:
    import termios
    import tty
    import select
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False
    print("Warning: termios not available. Keyboard input may not work on this system.")


class LiberoTeleop:
    """
    Keyboard teleoperation controller for LIBERO robot.
    
    Controls:
    - WASD: X/Y translation (W=forward, S=backward, A=left, D=right)
    - QE: Z translation (Q=up, E=down)
    - Arrow keys: Rotation (Up/Down for pitch, Left/Right for yaw)
    - IJ: Roll rotation (I=counter-clockwise, J=clockwise)
    - Space: Toggle gripper open/close
    - R: Reset to current position (stop movement)
    - ESC: Exit teleoperation mode
    """
    
    def __init__(self, 
                 pos_step=0.01,      # Position step size in meters
                 rot_step=0.1,      # Rotation step size in radians
                 gripper_toggle=True, # Whether to toggle gripper or hold
                 use_threading=True): # Use threading for non-blocking input
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.gripper_toggle = gripper_toggle
        self.use_threading = use_threading
        
        # Current action delta (relative to current position)
        self.action_delta = np.zeros(7)  # [dx, dy, dz, droll, dpitch, dyaw, dgripper]
        
        # State tracking
        self.gripper_state = -1.0  # -1 = closed, 1 = open
        self.active = False
        self.exit_requested = False
        
        # For non-blocking keyboard input
        self.key_queue = deque()
        self.input_thread = None
        
        if self.use_threading and HAS_TERMIOS:
            self._setup_non_blocking_input()
    
    def _setup_non_blocking_input(self):
        """Setup non-blocking keyboard input using termios."""
        if not HAS_TERMIOS:
            return
        
        # Save terminal settings
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        
        # Start input thread
        self.input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self.input_thread.start()
    
    def _input_loop(self):
        """Background thread for reading keyboard input."""
        while not self.exit_requested:
            if select.select([sys.stdin], [], [], 0.01)[0]:
                try:
                    key = sys.stdin.read(1)
                    self.key_queue.append(key)
                except:
                    break
    
    def _get_key(self):
        """Get a key from the queue (non-blocking)."""
        if self.key_queue:
            return self.key_queue.popleft()
        return None
    
    def _process_key(self, key):
        """Process a single keypress and update action delta."""
        if key is None:
            # Reset action delta when no key is pressed
            self.action_delta = np.zeros(7)
            return
        
        key = key.lower()
        
        # Reset action delta for new keypress
        self.action_delta = np.zeros(7)
        
        # Position controls (WASD + QE)
        if key == 'w':  # Forward (Y+)
            self.action_delta[1] = self.pos_step
        elif key == 's':  # Backward (Y-)
            self.action_delta[1] = -self.pos_step
        elif key == 'a':  # Left (X-)
            self.action_delta[0] = -self.pos_step
        elif key == 'd':  # Right (X+)
            self.action_delta[0] = self.pos_step
        elif key == 'q':  # Up (Z+)
            self.action_delta[2] = self.pos_step
        elif key == 'e':  # Down (Z-)
            self.action_delta[2] = -self.pos_step
        
        # Rotation controls
        elif key == '\x1b':  # ESC - check for arrow keys
            # Try to read more characters for arrow keys
            if select.select([sys.stdin], [], [], 0.01)[0]:
                seq = sys.stdin.read(2)
                if seq == '[A':  # Up arrow - pitch up
                    self.action_delta[4] = self.rot_step
                elif seq == '[B':  # Down arrow - pitch down
                    self.action_delta[4] = -self.rot_step
                elif seq == '[C':  # Right arrow - yaw right
                    self.action_delta[5] = self.rot_step
                elif seq == '[D':  # Left arrow - yaw left
                    self.action_delta[5] = -self.rot_step
                else:
                    # ESC key pressed - exit
                    self.exit_requested = True
        elif key == 'i':  # Roll counter-clockwise
            self.action_delta[3] = self.rot_step
        elif key == 'j':  # Roll clockwise
            self.action_delta[3] = -self.rot_step
        
        # Gripper control
        elif key == ' ':  # Space - toggle gripper
            if self.gripper_toggle:
                self.gripper_state = -self.gripper_state
            else:
                self.gripper_state = 1.0
            self.action_delta[6] = self.gripper_state
        elif key == 'r':  # Reset - no movement
            self.action_delta = np.zeros(7)
    
    def get_action(self, current_state):
        """
        Get teleoperation action based on current state and keyboard input.
        
        Args:
            current_state: Current robot state [x, y, z, roll, pitch, yaw, gripper]
            
        Returns:
            action: Target action [x, y, z, roll, pitch, yaw, gripper]
        """
        # Reset action delta at start of each call
        self.action_delta = np.zeros(7)
        
        # Process any pending keypresses
        key_processed = False
        while True:
            key = self._get_key()
            if key is None:
                break
            self._process_key(key)
            key_processed = True
        
        # If no key was processed, maintain current gripper state but no movement
        if not key_processed:
            self.action_delta[6] = self.gripper_state
        
        # Compute target action
        action = current_state.copy() + self.action_delta
        
        # Update gripper based on current state
        if self.action_delta[6] != 0:
            action[6] = self.gripper_state
        else:
            # Maintain current gripper state if no gripper key was pressed
            action[6] = current_state[6] if len(current_state) > 6 else self.gripper_state
        
        return action
    
    def is_active(self):
        """Check if teleoperation is active."""
        return self.active and not self.exit_requested
    
    def start(self):
        """Start teleoperation mode."""
        self.active = True
        self.exit_requested = False
        self.action_delta = np.zeros(7)
        print("\n" + "="*60)
        print("TELEOPERATION MODE ACTIVE")
        print("="*60)
        print("Controls:")
        print("  WASD: Move end-effector (W=forward, S=backward, A=left, D=right)")
        print("  QE: Move up/down (Q=up, E=down)")
        print("  Arrow keys: Rotate (Up/Down=pitch, Left/Right=yaw)")
        print("  IJ: Roll rotation (I=CCW, J=CW)")
        print("  Space: Toggle gripper")
        print("  R: Reset (stop movement)")
        print("  ESC: Exit teleoperation mode")
        print("="*60 + "\n")
    
    def stop(self):
        """Stop teleoperation mode."""
        self.active = False
        self.action_delta = np.zeros(7)
        print("\nTeleoperation mode deactivated.\n")
    
    def cleanup(self):
        """Clean up resources."""
        self.exit_requested = True
        if hasattr(self, 'old_settings') and HAS_TERMIOS:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


class SimpleLiberoTeleop:
    """
    Simplified teleoperation using blocking input (for systems without termios).
    Uses a simple polling approach.
    """
    
    def __init__(self, pos_step=0.01, rot_step=0.1):
        self.pos_step = pos_step
        self.rot_step = rot_step
        self.action_delta = np.zeros(7)
        self.gripper_state = -1.0
        self.active = False
        self.exit_requested = False
    
    def get_action(self, current_state):
        """Get action - for simple version, returns current state (no input)."""
        # Simple version doesn't support real-time input
        # This is a placeholder that can be extended
        return current_state
    
    def is_active(self):
        return self.active and not self.exit_requested
    
    def start(self):
        self.active = True
        print("\nSimple teleoperation mode (input not fully supported on this system)")
    
    def stop(self):
        self.active = False
    
    def cleanup(self):
        self.exit_requested = True


def create_teleop(use_threading=True):
    """Factory function to create appropriate teleop instance."""
    if HAS_TERMIOS and use_threading:
        return LiberoTeleop(use_threading=True)
    else:
        print("Warning: Using simplified teleoperation (limited input support)")
        return SimpleLiberoTeleop()

