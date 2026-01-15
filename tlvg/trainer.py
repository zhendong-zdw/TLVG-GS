# style_trainer.py
from typing import Dict, List
import torch
from gs.scene import GaussianModel, Scene
from gs.gaussian_renderer import render

from .configs import ConfigManager
# from style_utils import CUDATimer
from .phases.pre_process_phase import PreProcessPhase
from .phases.stylize_phase import StylizePhase
from .phases.post_process_phase import PostProcessPhase
from .phases.pre_process_phase import *
from .phases.stylize_phase import *
from .phases.pre_process_phase import *
from .preprocess.preprocess import preprocess
from .utils.timer import CUDATimer
from .utils.network_gui import handle_network_gui
from .utils.image import render_RGBcolor_images
from .observer import TrainingMetrics
from .observer import TrainingObserver
from .observer import ProgressTracker
from .observer import CheckpointSaver
from .observer import LossLogger
import os
from .loss.fast_loss import FASTLoss
from .loss.nnfm_loss import NNFMLoss
from .loss.knnfm_loss import KNNFMLoss
from .loss.gram_loss import GRAMLoss

class StyleTrainer:
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.device = config.model.data_device
        self.timer = CUDATimer()
        
        stylize_loss_dict = {
            "fast": FASTLoss,
            "nnfm": NNFMLoss,
            "knnfm": KNNFMLoss,
            "gram": GRAMLoss
        }
        self.stylize_loss_fn = stylize_loss_dict[config.style.method](config)
        
        self._initialize_components()
        
        
    def train(self):
        self._initialize_phases()
        self._initialize_observers()
        preprocess(self)
        
        for self.iteration in range(1, self.total_iterations + 1):
            if self.config.app.enable_gui:
                handle_network_gui(self)
            self._train_iteration()
            
        for observer in self.observers:
            observer.on_training_end()
    
        
    def _train_iteration(self):
        
        # Phase transition
        new_phase = self._determine_current_phase()
        if new_phase != self.cur_phase:
            self._transition_to_phase(new_phase)
        phase = self.phases[self.cur_phase]
        
        # Update Observer
        for observer in self.observers:
            observer.on_iteration_start(self.iteration)
        
        # Original Gaussian repo's operations
        self.gaussians.update_learning_rate(self.iteration)
        self.config.set_debug(True if self.iteration - 1 == self.config.app.debug_from else False)
        
        # Calculate loss
        losses, timing = phase.process_iteration(self.iteration)
        
        metrics = TrainingMetrics(
            iteration=self.iteration,
            phase=self.cur_phase,
            losses=losses,
            timing=timing
        )
        
        # Update Observer
        for observer in self.observers:
            observer.on_iteration_end(metrics)
    
    def _determine_current_phase(self):
        new_phase = self.cur_phase
        if self.cur_phase == -1:
            return 0
        while self.iteration > self.phases[new_phase].end_iter:
            new_phase += 1
        return new_phase  
    
    def _transition_to_phase(self, new_phase):
        
        if self.cur_phase != -1:
            # Check for NaN in gaussian model before phase transition
            phase_name = self.phases[self.cur_phase].name
            g = self.gaussians
            nan_xyz = torch.isnan(g._xyz).sum().item() if g._xyz.numel() > 0 else 0
            nan_opacity = torch.isnan(g._opacity).sum().item() if g._opacity.numel() > 0 else 0
            nan_scaling = torch.isnan(g._scaling).sum().item() if g._scaling.numel() > 0 else 0
            nan_rotation = torch.isnan(g._rotation).sum().item() if g._rotation.numel() > 0 else 0
            
            total_xyz = g._xyz.numel() if g._xyz.numel() > 0 else 0
            total_opacity = g._opacity.numel() if g._opacity.numel() > 0 else 0
            total_scaling = g._scaling.numel() if g._scaling.numel() > 0 else 0
            total_rotation = g._rotation.numel() if g._rotation.numel() > 0 else 0
            
            print(f"[PHASE TRANSITION] {phase_name} phase ended -> Checking gaussian model state:")
            print(f"  Total points: {total_opacity}")
            print(f"  NaN counts: xyz={nan_xyz}/{total_xyz}, opacity={nan_opacity}/{total_opacity}, "
                  f"scaling={nan_scaling}/{total_scaling}, rotation={nan_rotation}/{total_rotation}")
            
            if nan_xyz > 0 or nan_opacity > 0 or nan_scaling > 0 or nan_rotation > 0:
                print(f"  ⚠️  WARNING: NaN detected in gaussian model at end of {phase_name} phase!")
                print(f"  -> NaN was introduced during {phase_name} phase")
                print(f"  -> This will cause problems in the next phase")
            else:
                print(f"  ✓ No NaN detected - gaussian model is clean")
            
            # Save render image before phase transition
            try:
                output_dir = os.path.join(self.config.style.stylized_model_path, "render")
                os.makedirs(output_dir, exist_ok=True)
                
                # Get a viewpoint to render (use first camera for consistency)
                cameras = self.scene.getTrainCameras()
                if len(cameras) > 0:
                    viewpoint_cam = cameras[0]
                    render_pkg = self.get_render_pkgs(viewpoint_cam)
                    render_image = render_pkg["render"]
                    
                    # Save the render image with phase name
                    output_path = os.path.join(output_dir, f"{phase_name.replace(' ', '_')}_final.png")
                    render_RGBcolor_images(output_path, render_image)
                    print(f"Saved {phase_name} final render to {output_path}")
            except Exception as e:
                print(f"Warning: Failed to save render image before phase transition: {e}")
                import traceback
                traceback.print_exc()
            
            self.phases[self.cur_phase].cleanup_phase()
            self.phases[self.cur_phase] = None
        
        self.cur_phase = new_phase
        self.phases[self.cur_phase].setup_phase()
        
        for observer in self.observers:
            observer.on_phase_changed(self.cur_phase, new_phase)    
            
    def _initialize_components(self):
        
        # Just as same as original gaussian repo's initialization
        self.gaussians = GaussianModel(self.config.model.sh_degree)
        self.scene = Scene(self.config.model, self.gaussians, -1, shuffle=False)
        
        bg_color = [1, 1, 1] if self.config.model.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=self.config.model.data_device)
            
        self.gaussians.training_setup(self.config.opt)
        
    def _initialize_phases(self):
        
        self.cur_phase = -1
        self.phases = []
        self.total_iterations = 0
        
        phase_id = 0
        
        def _add_phase(phase, phase_name, num_iter):
            if num_iter == 0:
                return
            
            nonlocal phase_id
            
            begin_iter = self.total_iterations + 1
            end_iter = self.total_iterations + num_iter
            
            self.phases.append(phase(self, phase_id, phase_name, begin_iter, end_iter))
            
            self.total_iterations += num_iter
            phase_id += 1
        
        _add_phase(PreProcessPhase,  "Pre Process", self.config.style.iterations_pre_process)
        _add_phase(StylizePhase,     "Stylize", self.config.style.iterations_stylize)
        _add_phase(PostProcessPhase, "Post Process", self.config.style.iterations_post_process)
        
        
    def _initialize_observers(self):
        
        self.observers: List[TrainingObserver] = [
            ProgressTracker(self),
            CheckpointSaver(self),
            LossLogger(self)
        ]

    def _get_background(self):
        if self.config.opt.random_background:
            return torch.rand((3), device=self.config.model.data_device)
        return self.background

    def get_render_pkgs(self, viewpoint_cam):
        result = render(viewpoint_cam, self.gaussians, self.config.pipe, self._get_background())
        # Debug: Synchronize after render to catch errors at exact location
        # torch.cuda.synchronize()
        return result

