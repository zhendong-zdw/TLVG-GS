from abc import ABC, abstractmethod
from random import randint
from ..utils.image import render_RGBcolor_images
import torch

class BasePhase(ABC):
    
    def __init__(self, trainer, id, name, begin_iter, end_iter):
        self.trainer = trainer
        self.id = id
        self.name = name
        self.begin_iter = begin_iter
        self.end_iter = end_iter
        self.viewpoint_stack = None

    def update(self, iteration, loss):
        # if self.name == "Stylize" or self.name == "Post Process":
        #     render_RGBcolor_images("./debug/image.jpg", self.render_pkg["render"])
        
        # Check parameters before backward
        self._check_nan_before_update(iteration, "before_backward")
        
        # DIAGNOSTIC: If we have loss components and any are NaN, diagnose before backward
        if hasattr(self, '_loss_components'):
            has_nan_loss = any(
                isinstance(v, torch.Tensor) and (torch.isnan(v) or torch.isinf(v))
                for v in self._loss_components.values()
            )
            if has_nan_loss:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: NaN/Inf detected in loss components BEFORE backward")
                for name, val in self._loss_components.items():
                    if isinstance(val, torch.Tensor):
                        if torch.isnan(val) or torch.isinf(val):
                            print(f"  {name} loss is NaN/Inf: {val.item()}")
                # Diagnose which component causes NaN (before backward, so we can test each)
                self._diagnose_loss_gradients_before_backward(iteration, self._loss_components)
        
        # Check if loss requires grad before backward
        if loss.requires_grad and loss.grad_fn is not None:
            # Use retain_graph if we need to diagnose after
            use_retain_graph = hasattr(self, '_loss_components')
            loss.backward(retain_graph=use_retain_graph)
            # Check gradients after backward
            self._check_nan_after_backward(iteration)
        
        self._densification(iteration)
        # Check after densification
        self._check_nan_before_update(iteration, "after_densification")
        
        # Check for NaN in parameters before optimizer step - DIAGNOSTIC ONLY
        g = self.trainer.gaussians
        has_nan_before_step = False
        if g._xyz.numel() > 0:
            has_nan_before_step = has_nan_before_step or torch.isnan(g._xyz).any().item()
        if g._opacity.numel() > 0:
            has_nan_before_step = has_nan_before_step or torch.isnan(g._opacity).any().item()
        if g._scaling.numel() > 0:
            has_nan_before_step = has_nan_before_step or torch.isnan(g._scaling).any().item()
        if g._rotation.numel() > 0:
            has_nan_before_step = has_nan_before_step or torch.isnan(g._rotation).any().item()
        
        if has_nan_before_step:
            print(f"[{self.name} iter={iteration}] DIAGNOSTIC: NaN detected in parameters before optimizer.step() - NOT SKIPPING")
        
        self.trainer.gaussians.optimizer.step()
        # Check parameters after optimizer step
        self._check_nan_after_optimizer_step(iteration)
        
        self.trainer.gaussians.optimizer.zero_grad(set_to_none=True)
    
    def _check_nan_before_update(self, iteration, stage):
        """Check for NaN in gaussian parameters before update operations"""
        g = self.trainer.gaussians
        nan_xyz = torch.isnan(g._xyz).sum().item() if g._xyz.numel() > 0 else 0
        nan_opacity = torch.isnan(g._opacity).sum().item() if g._opacity.numel() > 0 else 0
        nan_scaling = torch.isnan(g._scaling).sum().item() if g._scaling.numel() > 0 else 0
        nan_rotation = torch.isnan(g._rotation).sum().item() if g._rotation.numel() > 0 else 0
        
        if nan_xyz > 0 or nan_opacity > 0 or nan_scaling > 0 or nan_rotation > 0:
            print(f"[{self.name} iter={iteration}] NaN detected at {stage}: "
                  f"xyz={nan_xyz}, opacity={nan_opacity}, scaling={nan_scaling}, rotation={nan_rotation}")
    
    def _check_nan_after_backward(self, iteration):
        """Check for NaN in gradients after backward - DIAGNOSTIC ONLY"""
        g = self.trainer.gaussians
        nan_grad_xyz = 0
        nan_grad_opacity = 0
        nan_grad_scaling = 0
        nan_grad_rotation = 0
        
        inf_grad_xyz = 0
        inf_grad_opacity = 0
        inf_grad_scaling = 0
        inf_grad_rotation = 0
        
        # Check for NaN/Inf gradients - DO NOT FIX, just report
        if g._xyz.grad is not None:
            nan_grad_xyz = torch.isnan(g._xyz.grad).sum().item()
            inf_grad_xyz = torch.isinf(g._xyz.grad).sum().item()
            if nan_grad_xyz > 0 or inf_grad_xyz > 0:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: xyz.grad has NaN={nan_grad_xyz}, Inf={inf_grad_xyz}")
                print(f"  xyz.grad stats: min={g._xyz.grad.min().item():.6f}, max={g._xyz.grad.max().item():.6f}, mean={g._xyz.grad.mean().item():.6f}")
        
        if g._opacity.grad is not None:
            nan_grad_opacity = torch.isnan(g._opacity.grad).sum().item()
            inf_grad_opacity = torch.isinf(g._opacity.grad).sum().item()
            if nan_grad_opacity > 0 or inf_grad_opacity > 0:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: opacity.grad has NaN={nan_grad_opacity}, Inf={inf_grad_opacity}")
                print(f"  opacity.grad stats: min={g._opacity.grad.min().item():.6f}, max={g._opacity.grad.max().item():.6f}, mean={g._opacity.grad.mean().item():.6f}")
        
        if g._scaling.grad is not None:
            nan_grad_scaling = torch.isnan(g._scaling.grad).sum().item()
            inf_grad_scaling = torch.isinf(g._scaling.grad).sum().item()
            if nan_grad_scaling > 0 or inf_grad_scaling > 0:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: scaling.grad has NaN={nan_grad_scaling}, Inf={inf_grad_scaling}")
                print(f"  scaling.grad stats: min={g._scaling.grad.min().item():.6f}, max={g._scaling.grad.max().item():.6f}, mean={g._scaling.grad.mean().item():.6f}")
        
        if g._rotation.grad is not None:
            nan_grad_rotation = torch.isnan(g._rotation.grad).sum().item()
            inf_grad_rotation = torch.isinf(g._rotation.grad).sum().item()
            if nan_grad_rotation > 0 or inf_grad_rotation > 0:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: rotation.grad has NaN={nan_grad_rotation}, Inf={inf_grad_rotation}")
                print(f"  rotation.grad stats: min={g._rotation.grad.min().item():.6f}, max={g._rotation.grad.max().item():.6f}, mean={g._rotation.grad.mean().item():.6f}")
        
        if nan_grad_xyz > 0 or nan_grad_opacity > 0 or nan_grad_scaling > 0 or nan_grad_rotation > 0:
            print(f"[{self.name} iter={iteration}] NaN in GRADIENTS after backward: "
                  f"xyz_grad={nan_grad_xyz}, opacity_grad={nan_grad_opacity}, "
                  f"scaling_grad={nan_grad_scaling}, rotation_grad={nan_grad_rotation}")
            
            # If this phase has loss components stored, diagnose which one causes NaN
            # (Only if we didn't already diagnose before backward)
            if hasattr(self, '_loss_components') and not hasattr(self, '_diagnosed_before_backward'):
                self._diagnose_loss_gradients_after_backward(iteration, self._loss_components)
    
    def _diagnose_loss_gradients_before_backward(self, iteration, loss_dict):
        """Diagnose which loss component produces NaN - BEFORE main backward"""
        print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Testing each loss component BEFORE main backward...")
        g = self.trainer.gaussians
        
        # Test each loss component separately (before main backward)
        for loss_name, loss_value in loss_dict.items():
            if not isinstance(loss_value, torch.Tensor) or not loss_value.requires_grad:
                continue
            
            # Check if loss itself is NaN
            if torch.isnan(loss_value) or torch.isinf(loss_value):
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Loss '{loss_name}' itself is NaN/Inf: {loss_value.item()}")
                continue
            
            # Zero gradients before testing this loss
            g.optimizer.zero_grad(set_to_none=True)
            
            try:
                # Backward on this loss only
                loss_value.backward(retain_graph=True)
                
                # Check for NaN in gradients
                has_nan = False
                nan_counts = {}
                for param_name in ['_xyz', '_opacity', '_scaling', '_rotation']:
                    param = getattr(g, param_name)
                    if param.grad is not None:
                        nan_count = torch.isnan(param.grad).sum().item()
                        inf_count = torch.isinf(param.grad).sum().item()
                        if nan_count > 0 or inf_count > 0:
                            has_nan = True
                            nan_counts[param_name] = (nan_count, inf_count)
                
                if has_nan:
                    print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Loss '{loss_name}' produces NaN gradients!")
                    for param_name, (nan_count, inf_count) in nan_counts.items():
                        print(f"  {param_name}: NaN={nan_count}, Inf={inf_count}")
                        param = getattr(g, param_name)
                        if param.grad is not None:
                            finite_grad = param.grad[torch.isfinite(param.grad)]
                            if finite_grad.numel() > 0:
                                print(f"    finite grad stats: min={finite_grad.min().item():.6f}, max={finite_grad.max().item():.6f}, mean={finite_grad.mean().item():.6f}")
            except Exception as e:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Error during backward of '{loss_name}': {e}")
        
        # Mark that we've diagnosed before backward
        self._diagnosed_before_backward = True
        
        # Zero gradients after diagnosis
        g.optimizer.zero_grad(set_to_none=True)
    
    def _diagnose_loss_gradients_after_backward(self, iteration, loss_dict):
        """Diagnose which loss component produces NaN gradients by testing each one separately"""
        g = self.trainer.gaussians
        
        # Save current gradients
        saved_grads = {}
        for param_name in ['_xyz', '_opacity', '_scaling', '_rotation']:
            param = getattr(g, param_name)
            if param.grad is not None:
                saved_grads[param_name] = param.grad.clone()
        
        # Zero gradients
        g.optimizer.zero_grad(set_to_none=True)
        
        # Test each loss component separately
        for loss_name, loss_value in loss_dict.items():
            if not isinstance(loss_value, torch.Tensor) or not loss_value.requires_grad:
                continue
            
            # Zero gradients before testing this loss
            g.optimizer.zero_grad(set_to_none=True)
            
            try:
                # Backward on this loss only
                loss_value.backward(retain_graph=True)
                
                # Check for NaN in gradients
                has_nan = False
                nan_counts = {}
                for param_name in ['_xyz', '_opacity', '_scaling', '_rotation']:
                    param = getattr(g, param_name)
                    if param.grad is not None:
                        nan_count = torch.isnan(param.grad).sum().item()
                        inf_count = torch.isinf(param.grad).sum().item()
                        if nan_count > 0 or inf_count > 0:
                            has_nan = True
                            nan_counts[param_name] = (nan_count, inf_count)
                
                if has_nan:
                    print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Loss '{loss_name}' produces NaN gradients!")
                    for param_name, (nan_count, inf_count) in nan_counts.items():
                        print(f"  {param_name}: NaN={nan_count}, Inf={inf_count}")
                        param = getattr(g, param_name)
                        if param.grad is not None:
                            finite_grad = param.grad[torch.isfinite(param.grad)]
                            if finite_grad.numel() > 0:
                                print(f"    finite grad stats: min={finite_grad.min().item():.6f}, max={finite_grad.max().item():.6f}, mean={finite_grad.mean().item():.6f}")
            except Exception as e:
                print(f"[{self.name} iter={iteration}] DIAGNOSTIC: Error during backward of '{loss_name}': {e}")
        
        # Restore original gradients
        g.optimizer.zero_grad(set_to_none=True)
        for param_name, saved_grad in saved_grads.items():
            param = getattr(g, param_name)
            if param is not None:
                param.grad = saved_grad
    
    def _check_nan_after_optimizer_step(self, iteration):
        """Check for NaN in parameters after optimizer step - DIAGNOSTIC ONLY"""
        g = self.trainer.gaussians
        nan_xyz = torch.isnan(g._xyz).sum().item() if g._xyz.numel() > 0 else 0
        nan_opacity = torch.isnan(g._opacity).sum().item() if g._opacity.numel() > 0 else 0
        nan_scaling = torch.isnan(g._scaling).sum().item() if g._scaling.numel() > 0 else 0
        nan_rotation = torch.isnan(g._rotation).sum().item() if g._rotation.numel() > 0 else 0
        
        if nan_xyz > 0 or nan_opacity > 0 or nan_scaling > 0 or nan_rotation > 0:
            print(f"[{self.name} iter={iteration}] DIAGNOSTIC: NaN detected AFTER optimizer.step(): "
                  f"xyz={nan_xyz}, opacity={nan_opacity}, scaling={nan_scaling}, rotation={nan_rotation}")
            print(f"  -> NaN was INTRODUCED by optimizer.step()")
            
            # Report parameter statistics
            if nan_xyz > 0:
                finite_xyz = g._xyz[torch.isfinite(g._xyz).all(dim=1)]
                if finite_xyz.numel() > 0:
                    print(f"  xyz stats (finite only): min={finite_xyz.min().item():.6f}, max={finite_xyz.max().item():.6f}, mean={finite_xyz.mean().item():.6f}")
            
            if nan_opacity > 0:
                finite_opacity = g._opacity[torch.isfinite(g._opacity)]
                if finite_opacity.numel() > 0:
                    print(f"  opacity stats (finite only): min={finite_opacity.min().item():.6f}, max={finite_opacity.max().item():.6f}, mean={finite_opacity.mean().item():.6f}")
            
            if nan_scaling > 0:
                finite_scaling = g._scaling[torch.isfinite(g._scaling).all(dim=1)]
                if finite_scaling.numel() > 0:
                    print(f"  scaling stats (finite only): min={finite_scaling.min().item():.6f}, max={finite_scaling.max().item():.6f}, mean={finite_scaling.mean().item():.6f}")
            
            if nan_rotation > 0:
                finite_rotation = g._rotation[torch.isfinite(g._rotation).all(dim=1)]
                if finite_rotation.numel() > 0:
                    print(f"  rotation stats (finite only): min={finite_rotation.min().item():.6f}, max={finite_rotation.max().item():.6f}, mean={finite_rotation.mean().item():.6f}")
    
    def _get_viewpoint_cam(self):
        if not self.viewpoint_stack or len(self.viewpoint_stack) == 0:
            self.viewpoint_stack = self.trainer.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack) - 1))
    
    def setup_phase(self): ...
    def cleanup_phase(self): ...
    
    @abstractmethod
    def process_iteration(self, iteration): ...
    
    def _densification(self, iteration): ...
    