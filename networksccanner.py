#!/usr/bin/env python3
"""
Advanced Network Scanner - Detects devices on network and scans for open ports
"""
import sys
import os
import socket
import ipaddress
import argparse
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from scapy.all import ARP, Ether, srp
import nmap

class NetworkScanner:
    def __init__(self, target=None, port_range=None, threads=10, timeout=2):
        self.target = target
        self.port_range = port_range or "1-1024"
        self.threads = threads
        self.timeout = timeout
        self.devices = []
        
    def get_local_ip(self):
        """Get local IP address of the machine"""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = '127.0.0.1'
        finally:
            s.close()
        return local_ip
        
    def discover_network(self):
        """Auto-detect network if not specified"""
        if not self.target:
            local_ip = self.get_local_ip()
            # Convert to CIDR notation (e.g., 192.168.1.5 -> 192.168.1.0/24)
            ip_parts = local_ip.split('.')
            self.target = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
            print(f"Auto-detected network: {self.target}")
        return self.target
    
    def validate_ip(self, ip):
        """Validate if the provided string is a valid IP address"""
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    def scan_devices(self):
        """Use ARP to scan for alive devices on the network"""
        # If target is a single IP, don't do ARP scan
        if self.validate_ip(self.target):
            print(f"[*] Using single IP target: {self.target}")
            try:
                # Try to get hostname
                try:
                    hostname = socket.gethostbyaddr(self.target)[0]
                except socket.herror:
                    hostname = "Unknown"
                
                self.devices = [{"ip": self.target, "mac": "Unknown", "hostname": hostname}]
            except Exception as e:
                print(f"[!] Error with IP {self.target}: {e}")
            return self.devices
        
        print(f"[*] Scanning for devices on {self.target}...")
        
        # Create ARP packet
        arp = ARP(pdst=self.target)
        ether = Ether(dst="ff:ff:ff:ff:ff:ff")  # Broadcast MAC
        packet = ether/arp
        
        start_time = time.time()
        result = srp(packet, timeout=self.timeout, verbose=0)[0]
        end_time = time.time()
        
        # Process results
        self.devices = []
        for sent, received in result:
            mac = received.hwsrc
            ip = received.psrc
            
            # Try to get hostname
            try:
                hostname = socket.gethostbyaddr(ip)[0]
            except socket.herror:
                hostname = "Unknown"
            
            self.devices.append({"ip": ip, "mac": mac, "hostname": hostname})
        
        print(f"[+] Found {len(self.devices)} active devices in {end_time - start_time:.2f} seconds")
        return self.devices
    
    def scan_ports(self, ip, ports):
        """Scan ports on a specific IP address"""
        nm = nmap.PortScanner()
        
        try:
            # Use nmap to scan ports
            print(f"[*] Scanning ports on {ip}...")
            result = nm.scan(ip, ports, arguments='-sS -T4')
            
            open_ports = []
            if ip in nm.all_hosts():
                for port in nm[ip]['tcp'].keys():
                    port_info = nm[ip]['tcp'][port]
                    if port_info['state'] == 'open':
                        service = port_info['name']
                        open_ports.append({
                            'port': port,
                            'service': service,
                            'version': port_info.get('product', '') + ' ' + port_info.get('version', '')
                        })
                        
            return ip, open_ports
        except Exception as e:
            print(f"[!] Error scanning {ip}: {e}")
            return ip, []
    
    def scan_all_devices(self):
        """Scan all found devices for open ports"""
        if not self.devices:
            print("[!] No devices found to scan")
            return []
        
        results = []
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = []
            for device in self.devices:
                ip = device['ip']
                futures.append(executor.submit(self.scan_ports, ip, self.port_range))
            
            for future in futures:
                ip, open_ports = future.result()
                for device in self.devices:
                    if device['ip'] == ip:
                        device['open_ports'] = open_ports
                        results.append(device)
                        
                        # Print results for this device
                        print(f"\n[+] Device: {ip} ({device['hostname']})")
                        print(f"    MAC Address: {device['mac']}")
                        
                        if open_ports:
                            print(f"    Open ports:")
                            for port in open_ports:
                                print(f"      {port['port']}/tcp - {port['service']} {port['version']}")
                        else:
                            print("    No open ports found")
        
        return results


def get_user_input():
    """Get IP address input from user"""
    while True:
        ip = input("Enter IP address to scan (or leave blank for auto-detect): ").strip()
        if not ip:
            return None
        
        try:
            # Validate IP address
            socket.inet_aton(ip)
            return ip
        except socket.error:
            try:
                # Check if it's a valid network in CIDR notation
                ipaddress.ip_network(ip)
                return ip
            except ValueError:
                print("Invalid IP address or network. Please try again.")


def main():
    parser = argparse.ArgumentParser(description='Advanced Network Scanner')
    parser.add_argument('-n', '--network', help='Target network in CIDR notation (e.g., 192.168.1.0/24)')
    parser.add_argument('-i', '--ip', help='Target specific IP address')
    parser.add_argument('-p', '--ports', default='1-1024', help='Port range to scan (e.g., 1-1000 or 22,80,443)')
    parser.add_argument('-t', '--threads', type=int, default=10, help='Number of threads for scanning')
    parser.add_argument('-o', '--output', help='Output file for results (JSON format)')
    parser.add_argument('--timeout', type=int, default=2, help='Timeout for scans in seconds')
    parser.add_argument('--interactive', action='store_true', help='Enter interactive mode for IP input')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"ADVANCED NETWORK SCANNER")
    print(f"Scan started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Determine the target
        target = None
        if args.interactive or (not args.network and not args.ip):
            target = get_user_input()
        elif args.ip:
            target = args.ip
        else:
            target = args.network
        
        scanner = NetworkScanner(
            target=target,
            port_range=args.ports,
            threads=args.threads,
            timeout=args.timeout
        )
        
        scanner.discover_network()
        scanner.scan_devices()
        results = scanner.scan_all_devices()
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\n[+] Results saved to {args.output}")
            
    except KeyboardInterrupt:
        print("\n[!] Scan interrupted by user")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        
    print("\n" + "=" * 60)
    print(f"Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    # Check for root privileges as network scanning often requires it
    if sys.platform.startswith('linux') and not os.geteuid() == 0:
        print("[!] This script requires root privileges for proper scanning")
        print("[!] Try running with sudo")
        sys.exit(1)
        
    main()