from .base_phase import BasePhase
from gs.utils.loss_utils import l1_loss, ssim
from ..utils.color_transfer import color_transfer
import torch

class PreProcessPhase(BasePhase):
    
    def setup_phase(self):
        if self.trainer.config.style.enable_color_transfer:
            color_transfer(self.trainer.ctx, self.trainer.config)
        
    def cleanup_phase(self):
        pass
    
    def process_iteration(self, iteration):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            scene_image = self.trainer.ctx.scene_images[viewpoint_cam.uid]
            
            Ll1 = l1_loss(render_image, scene_image)
            ssim_val = ssim(render_image, scene_image)
            
            loss = (
                (1.0 - self.trainer.config.opt.lambda_dssim) * Ll1 + 
                self.trainer.config.opt.lambda_dssim * (1.0 - ssim_val)
            )
            self.update(iteration, loss)
              
        return {
            "Points": f"{self.trainer.gaussians._opacity.shape[0]}",
            "Loss": f"{loss.item():.{7}f}"
        }, timer.elapsed_ms
        
    def _densification(self, iteration):
        delta_iteration = iteration - self.begin_iter
        
        gaussians = self.trainer.gaussians
        opt = self.trainer.config.opt
        scene = self.trainer.scene
        dataset = self.trainer.config.model
        
        viewspace_point_tensor = self.render_pkg["viewspace_points"]
        visibility_filter = self.render_pkg["visibility_filter"]
        radii = self.render_pkg["radii"]
        
        # Skip densification if no visible gaussians
        if visibility_filter.numel() == 0 or viewspace_point_tensor.numel() == 0:
            return
        
        # Check before densification operations
        nan_before = self._count_nan_in_gaussians()
        
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        
        if delta_iteration == 0:
            return
        
        if delta_iteration % self.trainer.config.style.style_densification_interval == 0:
            # Check before densify_and_prune
            nan_before = self._count_nan_in_gaussians()
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, 20, radii)
            # Check after densify_and_prune
            nan_after = self._count_nan_in_gaussians()
            total_before = sum(nan_before.values())
            total_after = sum(nan_after.values())
            if total_after != total_before:
                print(f"[PreProcess iter={iteration}] NaN introduced by densify_and_prune:")
                print(f"  before: xyz={nan_before['xyz']}, opacity={nan_before['opacity']}, "
                      f"scaling={nan_before['scaling']}, rotation={nan_before['rotation']}")
                print(f"  after:  xyz={nan_after['xyz']}, opacity={nan_after['opacity']}, "
                      f"scaling={nan_after['scaling']}, rotation={nan_after['rotation']}")
        
        if delta_iteration % opt.opacity_reset_interval == 0:
            nan_before = self._count_nan_in_gaussians()
            gaussians.reset_opacity()
            nan_after = self._count_nan_in_gaussians()
            total_before = sum(nan_before.values())
            total_after = sum(nan_after.values())
            if total_after != total_before:
                print(f"[PreProcess iter={iteration}] NaN introduced by reset_opacity:")
                print(f"  before: xyz={nan_before['xyz']}, opacity={nan_before['opacity']}, "
                      f"scaling={nan_before['scaling']}, rotation={nan_before['rotation']}")
                print(f"  after:  xyz={nan_after['xyz']}, opacity={nan_after['opacity']}, "
                      f"scaling={nan_after['scaling']}, rotation={nan_after['rotation']}")
    
    def _count_nan_in_gaussians(self):
        """Count NaN values in gaussian parameters"""
        g = self.trainer.gaussians
        return {
            'xyz': torch.isnan(g._xyz).sum().item() if g._xyz.numel() > 0 else 0,
            'opacity': torch.isnan(g._opacity).sum().item() if g._opacity.numel() > 0 else 0,
            'scaling': torch.isnan(g._scaling).sum().item() if g._scaling.numel() > 0 else 0,
            'rotation': torch.isnan(g._rotation).sum().item() if g._rotation.numel() > 0 else 0,
        }
