#!/usr/bin/env python3
"""
Run SUMO simulation with live metrics displayed in a GUI window
"""
import os
import sys
import yaml
import torch
import traci
import tkinter as tk
from tkinter import ttk
from collections import deque
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.sumo_env import SumoMDPEnv, EnvConfig
from src.dqn.agent import DQNAgent, AgentConfig
from src.baseline.fixed_time_controller import FixedTimeController, FixedTimeConfig


class MetricsWindow:
    """GUI window to display live simulation metrics"""
    
    def __init__(self, mode="DQN"):
        self.mode = mode
        self.root = tk.Tk()
        self.root.title(f"SUMO Metrics - {mode}")
        self.root.geometry("500x400")
        
        # Metrics storage
        self.current_metrics = {
            'queue': 0,
            'speed': 0,
            'wait': 0,
            'vehicles': 0,
            'reward': 0
        }
        self.avg_metrics = {
            'queue': 0,
            'speed': 0,
            'wait': 0,
            'vehicles': 0,
            'reward': 0
        }
        self.total_reward = 0
        self.total_vehicles = 0
        self.max_queue = 0
        self.step = 0
        self.max_steps = 720
        
        self.setup_ui()
        
    def setup_ui(self):
        """Create the UI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2E3440', pady=15)
        title_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            title_frame,
            text=f"SUMO Simulation Metrics",
            font=('Arial', 18, 'bold'),
            bg='#2E3440',
            fg='white'
        )
        title_label.pack()
        
        mode_label = tk.Label(
            title_frame,
            text=self.mode,
            font=('Arial', 12),
            bg='#2E3440',
            fg='#88C0D0'
        )
        mode_label.pack()
        
        # Metrics table
        table_frame = tk.Frame(self.root, padx=20, pady=20)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Headers
        headers = ['Metric', 'Current', 'Average']
        for col, header in enumerate(headers):
            label = tk.Label(
                table_frame,
                text=header,
                font=('Arial', 11, 'bold'),
                bg='#4C566A',
                fg='white',
                padx=10,
                pady=8
            )
            label.grid(row=0, column=col, sticky='ew', padx=2, pady=2)
        
        # Metric rows
        self.metric_labels = {}
        metrics_info = [
            ('queue', 'Queue Length', 'vehicles'),
            ('speed', 'Avg Speed', 'm/s'),
            ('wait', 'Avg Wait Time', 's'),
            ('vehicles', 'Vehicles Passed', 'count'),
            ('reward', 'Reward', 'points')
        ]
        
        for row, (key, name, unit) in enumerate(metrics_info, start=1):
            # Metric name
            tk.Label(
                table_frame,
                text=f"{name} ({unit})",
                font=('Arial', 10),
                anchor='w',
                padx=10,
                pady=8,
                bg='#ECEFF4'
            ).grid(row=row, column=0, sticky='ew', padx=2, pady=2)
            
            # Current value
            current_label = tk.Label(
                table_frame,
                text='0',
                font=('Arial', 10, 'bold'),
                fg='#2E7D32',
                bg='white',
                padx=10,
                pady=8
            )
            current_label.grid(row=row, column=1, sticky='ew', padx=2, pady=2)
            
            # Average value
            avg_label = tk.Label(
                table_frame,
                text='0',
                font=('Arial', 10),
                fg='#1565C0',
                bg='white',
                padx=10,
                pady=8
            )
            avg_label.grid(row=row, column=2, sticky='ew', padx=2, pady=2)
            
            self.metric_labels[key] = {
                'current': current_label,
                'average': avg_label
            }
        
        # Configure grid weights
        for col in range(3):
            table_frame.columnconfigure(col, weight=1)
        
        # Summary section
        summary_frame = tk.Frame(self.root, bg='#D8DEE9', padx=20, pady=15)
        summary_frame.pack(fill=tk.X)
        
        self.total_reward_label = tk.Label(
            summary_frame,
            text='Total Reward: 0',
            font=('Arial', 11, 'bold'),
            bg='#D8DEE9',
            fg='#2E3440'
        )
        self.total_reward_label.pack()
        
        self.total_vehicles_label = tk.Label(
            summary_frame,
            text='Total Vehicles: 0',
            font=('Arial', 11),
            bg='#D8DEE9',
            fg='#2E3440'
        )
        self.total_vehicles_label.pack()
        
        self.max_queue_label = tk.Label(
            summary_frame,
            text='Max Queue: 0',
            font=('Arial', 11),
            bg='#D8DEE9',
            fg='#2E3440'
        )
        self.max_queue_label.pack()
        
        # Progress bar
        progress_frame = tk.Frame(self.root, padx=20, pady=10)
        progress_frame.pack(fill=tk.X)
        
        self.progress_label = tk.Label(
            progress_frame,
            text='Step: 0/720',
            font=('Arial', 10)
        )
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            length=400,
            mode='determinate'
        )
        self.progress_bar.pack(pady=5)
    
    def update_metrics(self, current, avg, total_reward, total_vehicles, max_queue, step, max_steps):
        """Update all metric displays"""
        self.current_metrics = current
        self.avg_metrics = avg
        self.total_reward = total_reward
        self.total_vehicles = total_vehicles
        self.max_queue = max_queue
        self.step = step
        self.max_steps = max_steps
        
        # Update metric labels
        for key in self.metric_labels:
            current_val = current.get(key, 0)
            avg_val = avg.get(key, 0)
            
            # Format values
            if key in ['speed', 'wait', 'reward']:
                current_text = f"{current_val:.1f}"
                avg_text = f"{avg_val:.1f}"
            else:
                current_text = f"{int(current_val)}"
                avg_text = f"{avg_val:.1f}"
            
            self.metric_labels[key]['current'].config(text=current_text)
            self.metric_labels[key]['average'].config(text=avg_text)
        
        # Update summary
        self.total_reward_label.config(text=f'Total Reward: {int(total_reward)}')
        self.total_vehicles_label.config(text=f'Total Vehicles: {int(total_vehicles)}')
        self.max_queue_label.config(text=f'Max Queue: {int(max_queue)}')
        
        # Update progress
        self.progress_label.config(text=f'Step: {step}/{max_steps}')
        progress_percent = (step / max_steps) * 100
        self.progress_bar['value'] = progress_percent
        
        self.root.update()
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()
    
    def close(self):
        """Close the window"""
        try:
            self.root.destroy()
        except:
            pass


def run_simulation_with_gui_metrics(mode='dqn', num_steps=720):
    """Run simulation with GUI metrics window"""
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create metrics window
    metrics_window = MetricsWindow(mode=mode.upper())
    
    # Metrics tracking
    recent_queue = deque(maxlen=50)
    recent_speed = deque(maxlen=50)
    recent_wait = deque(maxlen=50)
    recent_vehicles = deque(maxlen=50)
    recent_reward = deque(maxlen=50)
    
    total_reward = 0
    total_vehicles = 0
    max_queue = 0
    
    # Create environment config (đọc đầy đủ từ config.yaml, khớp với train.py)
    _action_dur = config['sumo'].get('action_duration', 5)
    env_config = EnvConfig(
        sumocfg_path=config['sumo']['sumocfg_path'],
        tls_id=config['sumo']['tls_id'],
        phases=config['sumo']['phases'],
        max_steps=num_steps,
        action_duration=_action_dur,
        min_phase_steps=max(1, config['sumo'].get('min_phase_duration', 5) // _action_dur),
        max_phase_steps=max(2, config['sumo'].get('max_phase_duration', 140) // _action_dur),
        warmup_steps=config['sumo'].get('warmup_steps', 0),
        gui=True,
        phase_green_min={
            int(k): max(1, v // _action_dur)
            for k, v in config['sumo'].get('phase_green_min', {}).items()
        },
        phase_green_max={
            int(k): max(1, v // _action_dur)
            for k, v in config['sumo'].get('phase_green_max', {}).items()
        },
    )
    
    # Create environment
    env = SumoMDPEnv(env_config)
    
    # Reset environment to initialize state dimensions
    initial_state = env.reset()
    state_size = len(initial_state)
    action_size = env.action_space
    
    # Setup agent
    if mode.lower() == 'dqn':
        model_path = 'outputs/dqn_vn_tls.pt'
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        agent_config = AgentConfig(
            state_dim=state_size,
            action_dim=action_size
        )
        agent = DQNAgent(agent_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.q.load_state_dict(checkpoint)
        agent.q.eval()
        print(f"✓ Loaded DQN model from {model_path}")
    else:
        # Dùng 160s schedule bất đối xứng (EW:NS = 2:1) từ config, khớp với báo cáo
        _ft_schedule = config['sumo'].get(
            'fixed_time_phase_schedule', [(2, 100), (3, 5), (0, 50), (1, 5)]
        )
        controller_config = FixedTimeConfig(
            phase_schedule=[tuple(p) for p in _ft_schedule],
            action_duration=_action_dur,
        )
        controller = FixedTimeController(controller_config)
        print("✓ Using Fixed-Time controller (160s schedule)")
    
    # Run simulation in separate thread
    def run_simulation():
        nonlocal total_reward, total_vehicles, max_queue
        
        try:
            # Use initial state from environment reset above
            state = initial_state
            done = False
            step = 0
            
            while not done and step < num_steps:
                # Select action
                if mode.lower() == 'dqn':
                    with torch.no_grad():
                        state_tensor = torch.FloatTensor(state).unsqueeze(0)
                        q_values = agent.q(state_tensor)
                        action = q_values.argmax().item()
                else:
                    action = controller.get_action()
                
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Update metrics
                queue = info.get('queue_length', 0)
                speed = info.get('avg_speed', 0)
                wait = info.get('avg_wait', 0)
                vehicles = info.get('vehicles_passed', 0)
                
                recent_queue.append(queue)
                recent_speed.append(speed)
                recent_wait.append(wait)
                recent_vehicles.append(vehicles)
                recent_reward.append(reward)
                
                total_reward += reward
                total_vehicles += vehicles
                max_queue = max(max_queue, queue)
                
                # Update GUI every 5 steps
                if step % 5 == 0:
                    current = {
                        'queue': queue,
                        'speed': speed,
                        'wait': wait,
                        'vehicles': vehicles,
                        'reward': reward
                    }
                    
                    avg = {
                        'queue': sum(recent_queue) / len(recent_queue) if recent_queue else 0,
                        'speed': sum(recent_speed) / len(recent_speed) if recent_speed else 0,
                        'wait': sum(recent_wait) / len(recent_wait) if recent_wait else 0,
                        'vehicles': sum(recent_vehicles) / len(recent_vehicles) if recent_vehicles else 0,
                        'reward': sum(recent_reward) / len(recent_reward) if recent_reward else 0
                    }
                    
                    metrics_window.update_metrics(
                        current, avg, total_reward, total_vehicles, max_queue, step, num_steps
                    )
                
                state = next_state
                step += 1
            
            # Final update
            metrics_window.update_metrics(
                current, avg, total_reward, total_vehicles, max_queue, step, num_steps
            )
            
            print(f"\n✓ Simulation completed: {step} steps")
            print(f"  Total Reward: {total_reward:.0f}")
            print(f"  Total Vehicles: {total_vehicles}")
            print(f"  Avg Queue: {sum(recent_queue) / len(recent_queue):.2f}")
            
        except Exception as e:
            print(f"❌ Error in simulation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            env.close()
    
    # Start simulation thread
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # Run GUI (blocking)
    try:
        metrics_window.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        metrics_window.close()
        env.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python gui_metrics_window.py <dqn|fixed> [num_steps]")
        sys.exit(1)
    
    mode = sys.argv[1]
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 720
    
    run_simulation_with_gui_metrics(mode, num_steps)
