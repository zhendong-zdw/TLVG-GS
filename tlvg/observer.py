# style_observer.py
from dataclasses import dataclass
from typing import Dict, Optional
from abc import ABC
from tqdm import tqdm
import torch
import os
import json
from datetime import datetime

@dataclass
class TrainingMetrics:
    iteration: int
    phase: int
    losses: Dict[str, float]
    timing: float
    
    
class TrainingObserver(ABC):
    
    def on_iteration_start(self, iteration: int): ...
    
    def on_iteration_end(self, metrics: TrainingMetrics): ...
    
    def on_phase_changed(self, previous: int, current: int): ...
    
    def on_training_end(self): ...
    

class ProgressTracker(TrainingObserver):
    
    def __init__(self, trainer):
        
        # Use wider format to accommodate more loss info
        self.bar_format = "{l_bar}{bar:40}{r_bar}"
        self.trainer = trainer
        self.phase_bars: Dict[int, tqdm] = {}
        self.current_phase: Optional[int] = None
        self.update_interval: int = 10

    def on_phase_changed(self, previous: int, current: int):
        if previous in self.phase_bars:
            self.phase_bars[previous].close()
        
        if current not in self.phase_bars:
            phase = self.trainer.phases[current]
            self.phase_bars[current] = tqdm(
                total=phase.end_iter - phase.begin_iter + 1,
                desc=f"{phase.name.title():<18}",
                bar_format=self.bar_format,
                ncols=150  # Wider terminal width to show more info
            )

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration % self.update_interval != 0:
            return 
        
        if metrics.phase in self.phase_bars:
            self.phase_bars[metrics.phase].update(self.update_interval)
            self.phase_bars[metrics.phase].set_postfix(metrics.losses)
            
    def on_training_end(self):
        
        for phase_bar in self.phase_bars.values():
            phase_bar.close()
            
            
class CheckpointSaver(TrainingObserver):
    
    def __init__(self, trainer):
        self.trainer = trainer

    def on_iteration_end(self, metrics: TrainingMetrics):
        
        if metrics.iteration in self.trainer.config.ckpt.checkpoint_iterations:                
            print("\n[ITER {}] Saving Checkpoint".format(metrics.iteration))
            torch.save(
                (self.trainer.gaussians.capture(), metrics.iteration),
                self.trainer.config.style.stylized_model_path + "/chkpnt" + str(metrics.iteration) + ".pth",
            )
        
        if metrics.iteration in self.trainer.config.ckpt.save_iterations:
            self._save_gaussians(metrics.iteration)
    
    def _save_gaussians(self, iteration):
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        self.trainer.scene.save(self.trainer.total_iterations, self.trainer.config.style.stylized_model_path)
        
    def on_training_end(self):
        self._save_gaussians(self.trainer.total_iterations)


class LossLogger(TrainingObserver):
    """Save training loss history to a log file."""
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.log_file = os.path.join(trainer.config.style.stylized_model_path, "training_log.jsonl")
        self.loss_history = []
        os.makedirs(trainer.config.style.stylized_model_path, exist_ok=True)
        
    def on_iteration_end(self, metrics: TrainingMetrics):
        # Save to JSONL format (one JSON object per line)
        log_entry = {
            "iteration": metrics.iteration,
            "phase": metrics.phase,
            "phase_name": self.trainer.phases[metrics.phase].name if metrics.phase < len(self.trainer.phases) else "unknown",
            "losses": {k: float(v) if isinstance(v, (int, float, str)) else str(v) for k, v in metrics.losses.items()},
            "timing_ms": metrics.timing,
            "timestamp": datetime.now().isoformat()
        }
        self.loss_history.append(log_entry)
        
        # Append to file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def on_training_end(self):
        # Also save a summary JSON file
        summary_file = os.path.join(self.trainer.config.style.stylized_model_path, "training_summary.json")
        with open(summary_file, "w") as f:
            json.dump({
                "total_iterations": self.trainer.total_iterations,
                "loss_history": self.loss_history
            }, f, indent=2)
