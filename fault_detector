#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import EstimatorStatusFlags, SensorGps, VehicleStatus, VehicleCommand, VehicleCommandAck, DistanceSensor
import time
from threading import Lock
import math

class FaultDetector(Node):
    def __init__(self):
        super().__init__('fault_detector')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('gps_timeout', 10.0),
                ('min_satellites', 6),
                ('min_fix_type', 3),
                ('check_interval', 2.0),
                ('estimator_timeout', 10.0),
                ('startup_grace_period', 15.0),
                ('sitl_mode', True),
                # WIG-specific parameters
                ('altitude_sensor_max_diff', 2.0),  # Max difference between LIDAR/Radar in meters
                ('altitude_sensor_timeout', 5.0),   # Timeout for altitude sensors
            ]
        )

        # Get parameters
        self.gps_timeout = self.get_parameter('gps_timeout').value
        self.min_satellites = self.get_parameter('min_satellites').value
        self.min_fix_type = self.get_parameter('min_fix_type').value
        self.check_interval = self.get_parameter('check_interval').value
        self.estimator_timeout = self.get_parameter('estimator_timeout').value
        self.startup_grace = self.get_parameter('startup_grace_period').value
        self.sitl_mode = self.get_parameter('sitl_mode').value
        self.altitude_sensor_max_diff = self.get_parameter('altitude_sensor_max_diff').value
        self.altitude_sensor_timeout = self.get_parameter('altitude_sensor_timeout').value

        # Log parameters
        self.get_logger().info(f'Parameters: gps_timeout={self.gps_timeout}s, '
                               f'min_satellites={self.min_satellites}, '
                               f'min_fix_type={self.min_fix_type}, '
                               f'check_interval={self.check_interval}s, '
                               f'estimator_timeout={self.estimator_timeout}s, '
                               f'altitude_sensor_max_diff={self.altitude_sensor_max_diff}m')

        # State variables
        self.gps_status = None
        self.estimator_status = None
        self.vehicle_status = None
        self.last_gps_time = None
        self.last_estimator_time = None
        self.emergency_triggered = False
        self.data_lock = Lock()
        self.last_command_result = None
        self.node_start_time = time.time()
        self.estimator_ever_received = False
        
        # WIG-specific: altitude sensor data
        self.lidar_data = None
        self.radar_data = None
        self.last_lidar_time = None
        self.last_radar_time = None
        self.altitude_mismatch_count = 0
        self.altitude_sensor_warning_time = 0
        self.altitude_warning_cooldown = 10.0

        # Define QoS profile for PX4 topics
        px4_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publisher for direct PX4 control
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            px4_qos_profile
        )

        # Subscribers
        self.gps_sub = self.create_subscription(
            SensorGps,
            '/fmu/out/vehicle_gps_position',
            self.gps_callback,
            qos_profile=px4_qos_profile
        )

        self.estimator_sub = self.create_subscription(
            EstimatorStatusFlags,
            '/fmu/out/estimator_status_flags',
            self.estimator_callback,
            qos_profile=px4_qos_profile
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            qos_profile=px4_qos_profile
        )

        self.command_ack_sub = self.create_subscription(
            VehicleCommandAck,
            '/fmu/out/vehicle_command_ack',
            self.command_ack_callback,
            qos_profile=px4_qos_profile
        )
        
        # WIG-specific: Subscribe to multiple distance sensors
        # In SITL, we might only have one, but in real hardware we'd have LIDAR and Radar
        self.distance_sensor_sub = self.create_subscription(
            DistanceSensor,
            '/fmu/out/distance_sensor',
            self.distance_sensor_callback,
            qos_profile=px4_qos_profile
        )

        # Timer for periodic health checks
        self.timer = self.create_timer(self.check_interval, self.check_system_health)
        
        # Timer for altitude sensor validation (more frequent)
        self.altitude_timer = self.create_timer(0.5, self.validate_altitude_sensors)

        self.get_logger().info('WIG FaultDetector node initialized with altitude sensor validation')
        if self.sitl_mode:
            self.get_logger().info(f'🎮 SITL mode active with {self.startup_grace}s startup grace period')

    def distance_sensor_callback(self, msg):
        """Handle distance sensor data (LIDAR/Radar)"""
        with self.data_lock:
            # In SITL, we'll treat the single distance sensor as both LIDAR and Radar
            # In real hardware, you'd differentiate based on msg.type or topic
            current_time = time.time()
            
            if self.sitl_mode:
                # In SITL, simulate both sensors from one source
                self.lidar_data = msg
                self.radar_data = msg
                self.last_lidar_time = current_time
                self.last_radar_time = current_time
            else:
                # Real hardware: differentiate by sensor type
                # This would need adjustment based on your actual sensor setup
                if msg.type == 0:  # Assuming 0 is LIDAR
                    self.lidar_data = msg
                    self.last_lidar_time = current_time
                elif msg.type == 1:  # Assuming 1 is Radar
                    self.radar_data = msg
                    self.last_radar_time = current_time

    def gps_callback(self, msg):
        with self.data_lock:
            self.gps_status = msg
            if msg.fix_type >= self.min_fix_type:
                self.last_gps_time = time.time()

    def estimator_callback(self, msg):
        with self.data_lock:
            self.estimator_status = msg
            self.last_estimator_time = time.time()
            self.estimator_ever_received = True

    def status_callback(self, msg):
        with self.data_lock:
            self.vehicle_status = msg

    def command_ack_callback(self, msg):
        with self.data_lock:
            self.last_command_result = msg
            if msg.command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
                if msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED:
                    self.get_logger().info('✅ Emergency landing mode accepted by PX4')
                else:
                    self.get_logger().error(f'❌ Emergency landing mode failed: result={msg.result}')
                    self.emergency_triggered = False

    def is_in_startup_grace_period(self):
        """Check if still in startup grace period"""
        return (time.time() - self.node_start_time) < self.startup_grace

    def get_current_mode(self):
        if not self.vehicle_status:
            return "UNKNOWN"
        
        # Extended mapping
        mode_map = {
            0: "MANUAL", 1: "ALTITUDE", 2: "POSITION", 3: "AUTO.LOITER",
            4: "AUTO.MISSION", 5: "AUTO.RTL", 6: "AUTO.LAND", 7: "AUTO.TAKEOFF",
            8: "OFFBOARD", 9: "STABILIZED", 10: "ACRO", 11: "AUTO.LAND_ENGAGED",
            12: "AUTO.PRECLAND", 13: "ORBIT", 14: "AUTO.VTOL_TAKEOFF",
            15: "EXTERNAL1", 16: "EXTERNAL2", 17: "EXTERNAL3",
            18: "EXTERNAL4", 19: "EXTERNAL5", 20: "EXTERNAL6",
            21: "EXTERNAL7", 22: "EXTERNAL8"
        }
        
        return mode_map.get(self.vehicle_status.nav_state, f"UNKNOWN({self.vehicle_status.nav_state})")

    def is_failsafe_active(self):
        """Check if PX4 failsafe is already active"""
        if self.vehicle_status:
            return self.vehicle_status.failsafe
        return False
    
    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
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

    def validate_altitude_sensors(self):
        """WIG-specific: Cross-check LIDAR and Radar altitude readings"""
        if self.is_in_startup_grace_period():
            return
            
        with self.data_lock:
            current_time = time.time()
            
            # Check if we have recent data from both sensors
            lidar_valid = (self.lidar_data is not None and 
                          self.last_lidar_time is not None and 
                          (current_time - self.last_lidar_time) < self.altitude_sensor_timeout)
            
            radar_valid = (self.radar_data is not None and 
                          self.last_radar_time is not None and 
                          (current_time - self.last_radar_time) < self.altitude_sensor_timeout)
            
            # Log sensor status
            if not lidar_valid or not radar_valid:
                if current_time - self.altitude_sensor_warning_time > self.altitude_warning_cooldown:
                    sensor_status = []
                    if not lidar_valid:
                        sensor_status.append("LIDAR")
                    if not radar_valid:
                        sensor_status.append("Radar")
                    
                    self.get_logger().warn(f'⚠️ Altitude sensor timeout: {", ".join(sensor_status)}')
                    self.altitude_sensor_warning_time = current_time
                return
            
            # Cross-check altitude readings
            lidar_altitude = self.lidar_data.current_distance
            radar_altitude = self.radar_data.current_distance
            
            # Check if readings are valid (not max range)
            if lidar_altitude >= self.lidar_data.max_distance or radar_altitude >= self.radar_data.max_distance:
                self.get_logger().debug('Altitude sensors at max range - likely too high')
                return
            
            # Calculate difference
            altitude_diff = abs(lidar_altitude - radar_altitude)
            
            # Log current readings periodically
            if int(current_time) % 10 == 0:  # Every 10 seconds
                self.get_logger().info(
                    f'📏 Altitude sensors - LIDAR: {lidar_altitude:.2f}m, '
                    f'Radar: {radar_altitude:.2f}m, Diff: {altitude_diff:.2f}m'
                )
            
            # Check for mismatch
            if altitude_diff > self.altitude_sensor_max_diff:
                self.altitude_mismatch_count += 1
                
                if self.altitude_mismatch_count >= 3:  # 3 consecutive mismatches
                    self.get_logger().error(
                        f'❌ ALTITUDE SENSOR MISMATCH: LIDAR={lidar_altitude:.2f}m, '
                        f'Radar={radar_altitude:.2f}m, Difference={altitude_diff:.2f}m'
                    )
                    
                    # Determine which sensor to trust
                    # Generally, radar is more reliable over water
                    if self.sitl_mode:
                        self.get_logger().info('📡 In SITL mode - using average of both sensors')
                        trusted_altitude = (lidar_altitude + radar_altitude) / 2
                    else:
                        self.get_logger().info('📡 Using Radar as primary (more reliable over water)')
                        trusted_altitude = radar_altitude
                    
                    self.get_logger().info(f'✅ Trusted altitude: {trusted_altitude:.2f}m')
                    
                    # Reset counter after logging
                    self.altitude_mismatch_count = 0
            else:
                # Reset mismatch counter if readings agree
                self.altitude_mismatch_count = 0

    def check_system_health(self):
        """Main system health monitoring logic"""
        if self.emergency_triggered:
            return

        # Skip checks during startup grace period
        if self.is_in_startup_grace_period():
            self.get_logger().debug(f'In startup grace period ({time.time() - self.node_start_time:.1f}s)')
            return
        
        if self.is_failsafe_active():
            self.get_logger().info('PX4 failsafe active - letting it handle the situation')
            return

        with self.data_lock:
            current_time = time.time()

            # Check GPS health
            gps_healthy = self.check_gps_health(current_time)

            # Check estimator health (more lenient in SITL)
            estimator_healthy = self.check_estimator_health(current_time)
            
            # Check altitude sensor health (WIG-specific)
            altitude_sensors_healthy = self.check_altitude_sensor_health(current_time)

            # Log system status
            gps_status_str = "HEALTHY" if gps_healthy else "DEGRADED"
            est_status_str = "HEALTHY" if estimator_healthy else "DEGRADED"
            alt_status_str = "HEALTHY" if altitude_sensors_healthy else "DEGRADED"
            mode = self.get_current_mode()

            self.get_logger().info(
                f'System Health - GPS: {gps_status_str}, Estimator: {est_status_str}, '
                f'Altitude Sensors: {alt_status_str}, Mode: {mode}'
            )

            # Only trigger emergency for GPS failures (not estimator in SITL)
            if not gps_healthy:
                if self.gps_status is None:
                    self.get_logger().warn("No GPS data received yet")
                elif self.gps_status.fix_type < 2:
                    self.trigger_emergency_landing(f"GPS fix lost (fix_type={self.gps_status.fix_type})")
                elif self.gps_status.satellites_used < self.min_satellites:
                    self.trigger_emergency_landing(f"Low satellite count ({self.gps_status.satellites_used})")
                elif self.last_gps_time is None or (current_time - self.last_gps_time > self.gps_timeout):
                    self.trigger_emergency_landing("GPS data timeout")

            # For estimator, only warn in SITL mode
            if not estimator_healthy and self.sitl_mode:
                if not self.estimator_ever_received:
                    self.get_logger().debug("Estimator status not yet received (normal in SITL)")
                else:
                    self.get_logger().warn("Estimator degraded but not triggering emergency in SITL mode")
            
            # For altitude sensors, warn but don't trigger emergency
            if not altitude_sensors_healthy:
                self.get_logger().warn("⚠️ Altitude sensor issue - monitor altitude carefully")

    def check_gps_health(self, current_time):
        """Check GPS health status"""
        if self.gps_status is None:
            return False

        if self.gps_status.fix_type < self.min_fix_type:
            return False

        if self.gps_status.satellites_used < self.min_satellites:
            return False

        if self.last_gps_time is None or (current_time - self.last_gps_time > self.gps_timeout):
            return False

        if hasattr(self.gps_status, 'jamming_state') and self.gps_status.jamming_state == 3:
            return False
        
        if hasattr(self.gps_status, 'spoofing_state') and self.gps_status.spoofing_state >= 2:
            return False

        return True

    def check_estimator_health(self, current_time):
        """Check estimator health status - more lenient in SITL"""
        # In SITL, estimator might not always be available
        if self.sitl_mode and not self.estimator_ever_received:
            return True  # Assume healthy if never received in SITL
            
        if self.estimator_status is None:
            return False

        # More lenient timeout in SITL
        timeout = self.estimator_timeout * 2 if self.sitl_mode else self.estimator_timeout
        if self.last_estimator_time is None or (current_time - self.last_estimator_time > timeout):
            return False

        # Check GPS-related flags
        if hasattr(self.estimator_status, 'gps_check_fail_flags'):
            gps_flags = self.estimator_status.gps_check_fail_flags
            # In SITL, only check critical flags
            if gps_flags & (1 << 0):  # GPS_CHECK_FAIL_GPS_FIX
                return False

        return True
    
    def check_altitude_sensor_health(self, current_time):
        """Check if altitude sensors are providing valid data"""
        # Check LIDAR
        lidar_healthy = (self.lidar_data is not None and 
                        self.last_lidar_time is not None and 
                        (current_time - self.last_lidar_time) < self.altitude_sensor_timeout)
        
        # Check Radar
        radar_healthy = (self.radar_data is not None and 
                        self.last_radar_time is not None and 
                        (current_time - self.last_radar_time) < self.altitude_sensor_timeout)
        
        # In SITL, we might only have one sensor
        if self.sitl_mode:
            return lidar_healthy or radar_healthy
        else:
            # In real hardware, we want both sensors working
            return lidar_healthy and radar_healthy

    def trigger_emergency_landing(self, reason):
        """Trigger emergency landing"""
        if self.emergency_triggered:
            return

        # Don't trigger during startup
        if self.is_in_startup_grace_period():
            self.get_logger().warn(f'Emergency condition detected during startup: {reason} - ignoring')
            return

        self.emergency_triggered = True
        self.get_logger().error(f'EMERGENCY LANDING TRIGGERED: {reason}')
        self.get_logger().info('🌊 Note: WIG emergency landing would maintain forward speed for water landing')

        # Send LAND mode command
        self.send_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0,
            param2=6.0
        )
        
        self.get_logger().info('📡 Emergency landing command sent directly to PX4')

def main(args=None):
    rclpy.init(args=args)
    node = FaultDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down FaultDetector...')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
