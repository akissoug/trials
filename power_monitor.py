#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import BatteryStatus, VehicleGlobalPosition, VehicleStatus, VehicleCommand, VehicleCommandAck, DistanceSensor
import math
from threading import Lock
from collections import deque
import time

class PowerMonitor(Node):
    def __init__(self):
        super().__init__('power_monitor')

        # WIG-specific parameters
        self.declare_parameter('wingspan', 5.0)  # meters
        self.declare_parameter('optimal_height_ratio', 0.15)  # 15% of wingspan for best efficiency
        self.declare_parameter('min_wave_clearance', 1.0)  # minimum meters above waves
        self.declare_parameter('wave_height_sitl', 1.0)  # simulated wave height for SITL
        self.declare_parameter('wave_period_sitl', 8.0)  # simulated wave period for SITL
        self.declare_parameter('wave_direction_sitl', 0.0)  # simulated wave direction for SITL

        # Standard parameters
        self.declare_parameter('safety_margin', 0.3)
        self.declare_parameter('average_return_speed', 30.0)  # m/s - WIG cruise speed
        self.declare_parameter('battery_check_interval', 3.0)
        self.declare_parameter('min_battery_voltage', 14.0)
        self.declare_parameter('rtl_triggered_threshold', 0.25)
        self.declare_parameter('battery_capacity_mah', 5000.0)
        self.declare_parameter('min_current_threshold', 0.1)
        self.declare_parameter('current_averaging_window', 10)
        self.declare_parameter('home_position_timeout', 30.0)
        self.declare_parameter('sitl_mode', True)

        # Get parameters
        self.wingspan = self.get_parameter('wingspan').value
        self.optimal_height_ratio = self.get_parameter('optimal_height_ratio').value
        self.min_wave_clearance = self.get_parameter('min_wave_clearance').value
        self.wave_height_sitl = self.get_parameter('wave_height_sitl').value
        self.wave_period_sitl = self.get_parameter('wave_period_sitl').value
        self.wave_direction_sitl = self.get_parameter('wave_direction_sitl').value
        
        self.safety_margin = self.get_parameter('safety_margin').value
        self.return_speed = self.get_parameter('average_return_speed').value
        self.check_interval = self.get_parameter('battery_check_interval').value
        self.rtl_threshold = self.get_parameter('rtl_triggered_threshold').value
        self.battery_capacity = self.get_parameter('battery_capacity_mah').value
        self.min_current = self.get_parameter('min_current_threshold').value
        self.current_window_size = self.get_parameter('current_averaging_window').value
        self.home_timeout = self.get_parameter('home_position_timeout').value
        self.sitl_mode = self.get_parameter('sitl_mode').value

        # State
        self.battery_status = None
        self.global_position = None
        self.home_position = None
        self.vehicle_status = None
        self.distance_sensor = None  # For LIDAR/Radar altitude
        self.rtl_triggered = False
        self.lock = Lock()
        self.start_time = time.time()
        self.last_command_result = None
        self.command_retry_count = 0
        self.max_command_retries = 3
        self.last_command_time = 0
        self.command_cooldown = 5.0
        self.failsafe_notified = False
        self.last_altitude_warning_time = 0
        self.altitude_warning_cooldown = 10.0  # seconds between warnings
        
        # Current averaging
        self.current_history = deque(maxlen=self.current_window_size)
        self.sitl_default_current = 10.0

        # Define QoS profile
        px4_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos_profile
        )

        # Subscribers
        self.create_subscription(BatteryStatus, '/fmu/out/battery_status', self.battery_callback, qos_profile=px4_qos_profile)
        self.create_subscription(VehicleGlobalPosition, '/fmu/out/vehicle_global_position', self.gps_callback, qos_profile=px4_qos_profile)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile=px4_qos_profile)
        self.create_subscription(VehicleCommandAck, '/fmu/out/vehicle_command_ack', self.command_ack_callback, qos_profile=px4_qos_profile)
        self.create_subscription(DistanceSensor, '/fmu/out/distance_sensor', self.distance_sensor_callback, qos_profile=px4_qos_profile)

        # Periodic check
        self.create_timer(self.check_interval, self.check_battery_status)
        
        # Ground effect calculation timer (more frequent)
        self.create_timer(1.0, self.calculate_and_log_ground_effect)

        self.get_logger().info('✅ WIG PowerMonitor node initialized')
        self.get_logger().info(f'📐 Wingspan: {self.wingspan}m, Optimal height ratio: {self.optimal_height_ratio}')
        if self.sitl_mode:
            self.get_logger().info(f'🎮 SITL mode: Wave height={self.wave_height_sitl}m, Period={self.wave_period_sitl}s')

    def distance_sensor_callback(self, msg):
        """Receive LIDAR/Radar altitude data"""
        with self.lock:
            self.distance_sensor = msg

    def get_altitude_agl(self):
        """Get altitude above ground/water level"""
        if self.distance_sensor is not None:
            return self.distance_sensor.current_distance
        elif self.global_position is not None:
            # Fallback to GPS altitude if no distance sensor
            return self.global_position.alt - self.home_position[2] if self.home_position else self.global_position.alt
        return None

    def get_wave_conditions(self):
        """Get wave height, period, and direction"""
        if self.sitl_mode:
            # In SITL, use parameters
            return {
                'height': self.wave_height_sitl,
                'period': self.wave_period_sitl,
                'direction': self.wave_direction_sitl
            }
        else:
            # Real hardware: This would process radar data
            # For now, return defaults - implement radar processing when available
            return {
                'height': 1.0,  # Default 1m waves
                'period': 8.0,  # Default 8s period
                'direction': 0.0  # Default north
            }

    def calculate_ground_effect_efficiency(self):
        """Sophisticated ground effect calculation"""
        altitude_agl = self.get_altitude_agl()
        if altitude_agl is None:
            return 0.0, "No altitude data"
        
        # Height to wingspan ratio
        h_b_ratio = altitude_agl / self.wingspan
        
        # Ground effect model based on research
        # Maximum effect at h/b = 0.1, diminishes exponentially
        if h_b_ratio < 0.05:
            # Very close to surface - maximum benefit but risky
            efficiency = 0.35
            status = "Maximum ground effect (CAUTION: Very low!)"
        elif h_b_ratio < 0.1:
            # Optimal range
            efficiency = 0.30
            status = "Optimal ground effect"
        elif h_b_ratio < 0.2:
            # Good efficiency
            efficiency = 0.25 - (h_b_ratio - 0.1) * 0.5
            status = "Strong ground effect"
        elif h_b_ratio < 0.3:
            # Moderate efficiency
            efficiency = 0.20 - (h_b_ratio - 0.2) * 1.0
            status = "Moderate ground effect"
        elif h_b_ratio < 0.5:
            # Weak effect
            efficiency = 0.10 - (h_b_ratio - 0.3) * 0.25
            status = "Weak ground effect"
        elif h_b_ratio < 1.0:
            # Minimal effect
            efficiency = 0.05 * (1.0 - h_b_ratio)
            status = "Minimal ground effect"
        else:
            # No ground effect
            efficiency = 0.0
            status = "No ground effect"
        
        return efficiency, status

    def calculate_optimal_altitude(self):
        """Calculate optimal altitude considering waves and efficiency"""
        wave_conditions = self.get_wave_conditions()
        wave_height = wave_conditions['height']
        wave_period = wave_conditions['period']
        
        # Minimum safe altitude above waves
        min_safe_altitude = wave_height + self.min_wave_clearance
        
        # Optimal altitude for ground effect
        optimal_wig_altitude = self.wingspan * self.optimal_height_ratio
        
        # Account for wave dynamics (longer period = need more clearance)
        wave_factor = 1.0 + (wave_period / 10.0) * 0.2  # 20% more clearance for 10s waves
        dynamic_clearance = min_safe_altitude * wave_factor
        
        # Choose higher of the two
        recommended_altitude = max(dynamic_clearance, optimal_wig_altitude)
        
        return recommended_altitude, min_safe_altitude, optimal_wig_altitude

    def calculate_and_log_ground_effect(self):
        """Periodically calculate and log ground effect status"""
        with self.lock:
            altitude_agl = self.get_altitude_agl()
            if altitude_agl is None:
                return
            
            efficiency, status = self.calculate_ground_effect_efficiency()
            
            # Calculate altitude optimization
            recommended_alt, min_safe_alt, optimal_wig_alt = self.calculate_optimal_altitude()
            
            # Log ground effect status
            self.get_logger().info(
                f'🌊 WIG Status: Alt={altitude_agl:.1f}m, '
                f'Efficiency={efficiency*100:.0f}%, '
                f'Status: {status}'
            )
            
            # Check if altitude warning needed
            current_time = time.time()
            if current_time - self.last_altitude_warning_time > self.altitude_warning_cooldown:
                if altitude_agl < min_safe_alt:
                    self.get_logger().warn(
                        f'⚠️ ALTITUDE WARNING: Current {altitude_agl:.1f}m < '
                        f'Min safe {min_safe_alt:.1f}m above {self.get_wave_conditions()["height"]:.1f}m waves!'
                    )
                    self.last_altitude_warning_time = current_time
                elif abs(altitude_agl - recommended_alt) > 2.0:
                    self.get_logger().info(
                        f'💡 ALTITUDE TIP: Optimal altitude is {recommended_alt:.1f}m '
                        f'(currently at {altitude_agl:.1f}m)'
                    )
                    self.last_altitude_warning_time = current_time

    def get_efficiency_adjusted_current(self):
        """Get current draw adjusted for ground effect efficiency"""
        base_current = self.get_average_current()
        if base_current is None:
            return None
            
        efficiency, _ = self.calculate_ground_effect_efficiency()
        
        # In ground effect, we use less power
        adjusted_current = base_current * (1.0 - efficiency)
        
        return adjusted_current

    def battery_callback(self, msg):
        with self.lock:
            self.battery_status = msg
            current = msg.current_a if msg.current_a > 0 else self.sitl_default_current
            self.current_history.append(current)

    def gps_callback(self, msg):
        with self.lock:
            self.global_position = msg
            if (self.home_position is None and 
                not msg.dead_reckoning and 
                abs(msg.lat) > 0.001 and abs(msg.lon) > 0.001 and
                abs(msg.lat) <= 90.0 and abs(msg.lon) <= 180.0):
                
                self.home_position = (msg.lat, msg.lon, msg.alt)
                self.get_logger().info(f'🏠 Home position set: lat={msg.lat:.6f}, lon={msg.lon:.6f}, alt={msg.alt:.2f} m')

    def status_callback(self, msg):
        with self.lock:
            self.vehicle_status = msg
            
            # Check and notify about failsafe status changes
            if msg.failsafe and not self.failsafe_notified:
                self.get_logger().info('PX4 failsafe active - letting it handle the situation')
                self.failsafe_notified = True
            elif not msg.failsafe and self.failsafe_notified:
                self.get_logger().info('PX4 failsafe cleared')
                self.failsafe_notified = False

    def command_ack_callback(self, msg):
        with self.lock:
            self.last_command_result = msg
            if msg.command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
                if msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
                    self.get_logger().info('✅ RTL mode change accepted by PX4')
                    self.command_retry_count = 0
                    self.rtl_triggered = True
                elif msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_TEMPORARILY_REJECTED:
                    self.get_logger().warn('⏳ RTL command temporarily rejected - will retry')
                else:
                    self.get_logger().error(f'❌ RTL mode change failed: result={msg.result}')
                    self.command_retry_count += 1
                    if self.command_retry_count >= self.max_command_retries:
                        self.get_logger().error('❌ Max retries reached - giving up on RTL command')
                        self.rtl_triggered = True

    def get_current_mode(self):
        if not self.vehicle_status:
            return "UNKNOWN"
        
        mode_map = {
            0: "MANUAL", 1: "ALTITUDE", 2: "POSITION", 3: "AUTO.LOITER",
            4: "AUTO.MISSION", 5: "AUTO.RTL", 6: "AUTO.LAND", 7: "AUTO.TAKEOFF",
            8: "OFFBOARD", 9: "STABILIZED", 10: "ACRO", 11: "AUTO.LAND_ENGAGED",
            12: "AUTO.PRECLAND", 13: "ORBIT", 14: "AUTO.VTOL_TAKEOFF"
        }
        
        return mode_map.get(self.vehicle_status.nav_state, f"UNKNOWN({self.vehicle_status.nav_state})")

    def is_failsafe_active(self):
        """Check if PX4 failsafe is already active"""
        if self.vehicle_status:
            return self.vehicle_status.failsafe
        return False

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_command_time < self.command_cooldown:
            self.get_logger().debug('Command cooldown active - skipping')
            return
            
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = param1
        msg.param2 = param2
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        
        self.vehicle_command_pub.publish(msg)
        self.last_command_time = current_time

    def get_battery_percentage(self):
        """Get battery percentage from real PX4 data only"""
        if self.battery_status is None:
            return None
            
        # Use real battery data from PX4
        if hasattr(self.battery_status, 'remaining'):
            if self.battery_status.remaining <= 1.0:
                return self.battery_status.remaining * 100
            else:
                return self.battery_status.remaining
        return None

    def calculate_remaining_time(self):
        battery_percentage = self.get_battery_percentage()
        if battery_percentage is None:
            return None

        # Use efficiency-adjusted current for WIG
        avg_current = self.get_efficiency_adjusted_current()
        if avg_current is None or avg_current < self.min_current:
            return None

        capacity = self.battery_capacity
        if hasattr(self.battery_status, 'capacity') and self.battery_status.capacity > 0:
            capacity = self.battery_status.capacity

        remaining_mah = (battery_percentage / 100.0) * capacity
        current_draw_ma = avg_current * 1000.0
        time_hours = remaining_mah / current_draw_ma
        return time_hours * 3600.0

    def get_average_current(self):
        if not self.current_history:
            return self.sitl_default_current if self.sitl_mode else None
        return sum(self.current_history) / len(self.current_history)

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = math.sin(d_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_distance_to_home(self):
        if self.home_position is None or self.global_position is None:
            return None
        
        lat1, lon1, alt1 = self.home_position
        lat2, lon2, alt2 = self.global_position.lat, self.global_position.lon, self.global_position.alt
        horizontal = self.haversine_distance(lat1, lon1, lat2, lon2)
        # WIG craft doesn't need vertical component for return
        return horizontal

    def check_battery_status(self):
        with self.lock:
            if self.rtl_triggered:
                return

            # Skip if PX4 failsafe is already active
            if self.is_failsafe_active():
                return

            if self.home_position is None:
                elapsed = time.time() - self.start_time
                if elapsed > self.home_timeout:
                    self.get_logger().warn(f'⚠️ No home position after {self.home_timeout}s')
                return

            current_mode = self.get_current_mode()
            if current_mode in ['AUTO.RTL', 'AUTO.LAND', 'MANUAL', 'AUTO.LAND_ENGAGED', 'AUTO.PRECLAND']:
                return

            if not all([self.battery_status, self.global_position]):
                return

            battery_percentage = self.get_battery_percentage()
            if battery_percentage is None:
                return

            remaining_time = self.calculate_remaining_time()
            distance = self.calculate_distance_to_home()

            if distance is None:
                return

            # Calculate return time for WIG (no climb needed)
            base_return_time = distance / self.return_speed
            safety_return_time = base_return_time * (1 + self.safety_margin)
            
            # Add time for landing approach (WIG specific)
            landing_approach_time = 60.0  # 1 minute for approach and water landing
            total_time_needed = safety_return_time + landing_approach_time

            # Get current efficiency
            efficiency, status = self.calculate_ground_effect_efficiency()

            # Build log message
            log_msg = f'🔋 Battery: {battery_percentage:.1f}%, '
            if remaining_time is not None:
                log_msg += f'⏳ Remaining: {remaining_time:.1f}s, '
            else:
                log_msg += '⏳ Remaining: N/A, '
            log_msg += f'🏠 Distance: {distance:.1f}m, '
            log_msg += f'🔁 Needed: {total_time_needed:.1f}s, '
            log_msg += f'⚡ Efficiency: {efficiency*100:.0f}%, '
            log_msg += f'📍 Mode: {current_mode}'
            
            self.get_logger().info(log_msg)

            # Check RTL conditions
            should_rtl = False
            reason = ""
            
            if battery_percentage < self.rtl_threshold * 100:
                should_rtl = True
                reason = f"low battery ({battery_percentage:.1f}%)"
            elif remaining_time is not None and remaining_time < total_time_needed:
                should_rtl = True
                reason = f"insufficient time for return and landing"
            
            if should_rtl and self.command_retry_count < self.max_command_retries:
                self.trigger_rtl(reason)

    def trigger_rtl(self, reason="battery low"):
        if self.rtl_triggered:
            return
        
        self.get_logger().warn(f'⚠️ Triggering RTL: {reason}')
        self.get_logger().info('🌊 Note: WIG-specific RTL would maintain low altitude for efficiency')
        
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=5.0
        )
        
        self.get_logger().info('📡 RTL command sent directly to PX4')

def main(args=None):
    rclpy.init(args=args)
    node = PowerMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
