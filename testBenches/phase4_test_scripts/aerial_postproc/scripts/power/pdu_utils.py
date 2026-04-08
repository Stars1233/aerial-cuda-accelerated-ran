#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Raritan PDU utility library for retrieving outlet power data.
"""

# Add raritan rpc module to path if it exists
try:
    from raritan import rpc
    from raritan.rpc import pdumodel
except ImportError:
    print("Warning: Raritan RPC module not found. Please install it first.")
    print("You may need to download and install the Raritan Python client bindings.")
    rpc = None
    pdumodel = None


class RaritanPDU:
    """
    Raritan PDU interface class for retrieving outlet power/current/voltage data.
    
    Provides the same interface as the TripLite PDU implementation in collect_power.py
    """
    
    def __init__(self, hostname, outlet_list, username, password, 
                 disable_ssl_verification=True, timeout=5):
        """
        Initialize Raritan PDU connection.
        
        Args:
            hostname (str): PDU hostname or IP address
            outlet_list (list): List of outlet numbers to monitor
            username (str): Username for authentication
            password (str): Password for authentication
            disable_ssl_verification (bool): Whether to disable SSL certificate verification
            timeout (int): Connection timeout in seconds
        """
        if rpc is None or pdumodel is None:
            raise ImportError("Raritan RPC modules not available. Please install the Raritan Python client bindings.")
            
        self.hostname = hostname
        self.outlet_list = outlet_list
        self.username = username
        self.password = password
        self.timeout = timeout
        self.disable_ssl_verification = disable_ssl_verification
        
        # Initialize the RPC agent
        self.agent = None
        self.pdu = None
        self.outlets = None
        self.outlet_sensors = {}  # Cache for sensor objects
        self.bulk_helper = None  # Cache for bulk request helper
        self.request_map = {}  # Cache for request mapping
        self._connect()
    
    def _connect(self):
        """Establish connection to the PDU."""
        try:
            # Create RPC agent with optimized settings
            if self.disable_ssl_verification:
                self.agent = rpc.Agent(
                    "https", 
                    self.hostname, 
                    self.username, 
                    self.password,
                    disable_certificate_verification=True,
                    timeout=self.timeout
                )
            else:
                self.agent = rpc.Agent(
                    "https", 
                    self.hostname, 
                    self.username, 
                    self.password,
                    timeout=self.timeout
                )
            
            # Get the main PDU object
            self.pdu = pdumodel.Pdu("/model/pdu/0", self.agent)
            
            # Test connection by getting metadata
            metadata = self.pdu.getMetaData()
            print(f"Connected to Raritan PDU: {metadata.nameplate.model} "
                  f"S/N: {metadata.nameplate.serialNumber}")
            
            # Cache outlets and sensors for performance
            self.outlets = self.pdu.getOutlets()
            for outlet_num in self.outlet_list:
                outlet_idx = outlet_num - 1
                if outlet_idx < len(self.outlets):
                    self.outlet_sensors[outlet_num] = self.outlets[outlet_idx].getSensors()
            
            # Create and cache bulk request helper for performance
            self.bulk_helper = rpc.BulkRequestHelper(self.agent)
            
            # Pre-build the bulk request queue for maximum efficiency
            self._build_request_queue()
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Raritan PDU at {self.hostname}: {str(e)}")
    
    def _build_request_queue(self):
        """Build the bulk request queue once for reuse."""
        request_idx = 0
        
        for outlet_num in self.outlet_list:
            if outlet_num not in self.outlet_sensors:
                print(f"Warning: Outlet {outlet_num} not available during setup")
                continue
            
            sensors = self.outlet_sensors[outlet_num]
            
            # Queue current reading request
            self.bulk_helper.add_request(sensors.current.getReading)
            self.request_map[request_idx] = (outlet_num, 'current')
            request_idx += 1
            
            # Queue power reading request
            self.bulk_helper.add_request(sensors.activePower.getReading)
            self.request_map[request_idx] = (outlet_num, 'power')
            request_idx += 1
            
            # Queue voltage reading request
            self.bulk_helper.add_request(sensors.voltage.getReading)
            self.request_map[request_idx] = (outlet_num, 'voltage')
            request_idx += 1
    
    def get_pdu_data(self):
        """
        Retrieve power data from specified outlets using bulk requests for optimal performance.
        
        Returns the same dictionary format as get_pdu_data_tripplite():
        {
            'outlet_N_current': float,  # Current in Amps
            'outlet_N_power': float,    # Power in Watts
            'outlet_N_voltage': float,  # Voltage in Volts (per outlet)
            'total_current': float,     # Sum of all outlet currents
            'total_power': float        # Sum of all outlet powers
        }
        """
        if self.pdu is None:
            raise RuntimeError("PDU not connected. Call _connect() first.")
        
        out_dict = {}
        total_current = 0.0
        total_power = 0.0
        
        try:
            # Execute the pre-built bulk request queue (no need to rebuild!)
            responses = self.bulk_helper.perform_bulk(raise_subreq_failure=True)
            
            # Process responses and build result dictionary
            outlet_data = {}  # Temporary storage: {outlet_num: {'current': val, 'power': val, 'voltage': val}}
            
            for idx, response in enumerate(responses):
                if idx in self.request_map:
                    outlet_num, sensor_type = self.request_map[idx]
                    
                    if outlet_num not in outlet_data:
                        outlet_data[outlet_num] = {}
                    
                    # Extract sensor value from response
                    sensor_value = response.value
                    
                    # Handle 'N/A' or invalid values
                    if sensor_value == 'N/A' or sensor_value is None:
                        print(f"WARNING: PDU Outlet {outlet_num} {sensor_type} returned 'N/A' or None, using 0.0")
                        sensor_value = 0.0
                    elif isinstance(sensor_value, str):
                        try:
                            sensor_value = float(sensor_value)
                        except ValueError:
                            print(f"WARNING: PDU Outlet {outlet_num} {sensor_type} has invalid string value '{sensor_value}', using 0.0")
                            sensor_value = 0.0
                    
                    outlet_data[outlet_num][sensor_type] = sensor_value
            
            # Build final output dictionary and calculate totals
            for outlet_num in sorted(outlet_data.keys()):
                data = outlet_data[outlet_num]
                
                current_amps = data.get('current', 0.0)
                power_watts = data.get('power', 0.0)
                voltage_volts = data.get('voltage', 0.0)
                
                # Store individual outlet data
                out_dict[f"outlet_{outlet_num}_current"] = current_amps
                out_dict[f"outlet_{outlet_num}_power"] = power_watts
                out_dict[f"outlet_{outlet_num}_voltage"] = voltage_volts
                
                # Accumulate totals
                total_current += current_amps
                total_power += power_watts
                
                print(f"Outlet {outlet_num}: {current_amps:.3f}A, {power_watts:.1f}W, {voltage_volts:.1f}V")
        
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve PDU data: {str(e)}")
        
        # Add summary data
        out_dict.update({
            'total_current': total_current,
            'total_power': total_power,
        })
        
        return out_dict
    
    def get_outlet_info(self):
        """Get information about all available outlets."""
        if self.pdu is None:
            raise RuntimeError("PDU not connected.")
        
        outlets = self.pdu.getOutlets()
        outlet_info = []
        
        for i, outlet in enumerate(outlets):
            metadata = outlet.getMetaData()
            outlet_info.append({
                'number': i + 1,
                'name': getattr(metadata, 'name', f'Outlet {i + 1}'),
                'label': getattr(metadata, 'label', f'Outlet {i + 1}')
            })
        
        return outlet_info
    
    def __del__(self):
        """Cleanup connection when object is destroyed."""
        # The RPC agent should handle cleanup automatically
        pass


# Test function
def test_raritan_pdu(hostname, outlets, username, password):
    """Test function to verify PDU connectivity and data retrieval."""
    import time
    
    try:
        pdu = RaritanPDU(hostname, outlets, username, password)
        
        print("\nTesting data retrieval for 10 seconds...")
        start_time = time.time()
        sample_count = 0
        total_get_time = 0
        
        while (time.time() - start_time) < 10.0:
            # Time the get_pdu_data call
            get_start = time.time()
            data = pdu.get_pdu_data()
            get_end = time.time()
            
            get_duration = get_end - get_start
            total_get_time += get_duration
            sample_count += 1
            
            print(f"Sample {sample_count}: get_pdu_data() took {get_duration:.3f}s")
            
            # Wait until next second
            time.sleep(max(0, 1.0 - get_duration))
        
        total_duration = time.time() - start_time
        avg_get_time = total_get_time / sample_count if sample_count > 0 else 0
        
        print(f"\nSummary:")
        print(f"  Total test duration: {total_duration:.3f}s")
        print(f"  Samples collected: {sample_count}")
        print(f"  Average get_pdu_data() time: {avg_get_time:.3f}s")
        print(f"  Total time in get_pdu_data(): {total_get_time:.3f}s")
        print(f"  Sampling rate: {sample_count/total_duration:.1f} samples/sec")
        
        print(f"\nLast data sample:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Raritan PDU connection and data retrieval")
    parser.add_argument("hostname", help="PDU hostname or IP address")
    parser.add_argument("outlets", help="Comma-separated list of outlet numbers (e.g., 30,32)")
    parser.add_argument("username", help="Username for PDU authentication")
    parser.add_argument("password", help="Password for PDU authentication")
    
    args = parser.parse_args()
    
    # Parse outlet list
    outlet_list = [int(x.strip()) for x in args.outlets.split(',')]
    
    print(f"Testing Raritan PDU connection to {args.hostname}")
    print(f"Outlets: {outlet_list}")
    print(f"Username: {args.username}")
    
    test_raritan_pdu(args.hostname, outlet_list, args.username, args.password)
