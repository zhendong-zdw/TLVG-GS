from .base_phase import BasePhase
from ..loss.other_loss import content_loss_fn
from ..loss.other_loss import image_tv_loss_fn
from ..loss.other_loss import brush_shape_loss_fn, stroke_direction_loss_fn
from ..loss.other_loss import ab_stat_loss, ab_patch_stat_loss
import torch

class StylizePhase(BasePhase):
    
    def setup_phase(self):
        
        self.initial_opacity = self.trainer.gaussians._opacity.clone().detach()
        self.initial_scaling = self.trainer.gaussians._scaling.clone().detach()
        self._orig_convert_shs_python = self.trainer.config.pipe.convert_SHs_python
        self.trainer.config.pipe.convert_SHs_python = False

    def cleanup_phase(self):
        self.trainer.config.pipe.convert_SHs_python = self._orig_convert_shs_python
    
    def _diagnose_loss_gradients(self, iteration, loss_dict):
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
                    print(f"[Stylize iter={iteration}] DIAGNOSTIC: Loss '{loss_name}' produces NaN gradients!")
                    for param_name, (nan_count, inf_count) in nan_counts.items():
                        print(f"  {param_name}: NaN={nan_count}, Inf={inf_count}")
                        param = getattr(g, param_name)
                        if param.grad is not None:
                            finite_grad = param.grad[torch.isfinite(param.grad)]
                            if finite_grad.numel() > 0:
                                print(f"    finite grad stats: min={finite_grad.min().item():.6f}, max={finite_grad.max().item():.6f}, mean={finite_grad.mean().item():.6f}")
            except Exception as e:
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: Error during backward of '{loss_name}': {e}")
        
        # Restore original gradients
        g.optimizer.zero_grad(set_to_none=True)
        for param_name, saved_grad in saved_grads.items():
            param = getattr(g, param_name)
            if param is not None:
                param.grad = saved_grad
    
    def process_iteration(self, iteration):
        
        viewpoint_cam = self._get_viewpoint_cam()
        
        with self.trainer.timer as timer:
            self.trainer.config.pipe.convert_SHs_python = False
            
            self.render_pkg = self.trainer.get_render_pkgs(viewpoint_cam)
            render_image = self.render_pkg["render"]
            
            # Check render_image for NaN before feature extraction
            if torch.isnan(render_image).any() or torch.isinf(render_image).any():
                nan_count = torch.isnan(render_image).sum().item()
                inf_count = torch.isinf(render_image).sum().item()
                print(f"[Stylize iter={iteration}] ERROR: render_image contains NaN/Inf before feature extraction: NaN={nan_count}, Inf={inf_count}")
            
            render_feature_list, _ = self.trainer.feature_extractor(
                render_image.unsqueeze(0),
                self.trainer.ctx.scene_masks[viewpoint_cam.uid].unsqueeze(0),
                self.trainer.config.style.scene_classes)
            # Batch is 1, so get the first render_feature_list
            render_feature_list = render_feature_list[0]
            
            # Check features for NaN
            for i, feat in enumerate(render_feature_list):
                if torch.isnan(feat).any() or torch.isinf(feat).any():
                    nan_count = torch.isnan(feat).sum().item()
                    inf_count = torch.isinf(feat).sum().item()
                    print(f"[Stylize iter={iteration}] ERROR: render_feature_list[{i}] contains NaN/Inf: NaN={nan_count}, Inf={inf_count}")
            
            # Check feature norms before loss computation (cosine similarity can fail with zero norm)
            for i, feat in enumerate(render_feature_list):
                feat_norm = feat.norm(dim=0)
                if (feat_norm < 1e-8).any():
                    zero_norm_count = (feat_norm < 1e-8).sum().item()
                    print(f"[Stylize iter={iteration}] WARNING: render_feature_list[{i}] has {zero_norm_count} features with near-zero norm (<1e-8)")
                if torch.isnan(feat_norm).any() or torch.isinf(feat_norm).any():
                    print(f"[Stylize iter={iteration}] ERROR: render_feature_list[{i}] norm contains NaN/Inf")
            
            stylize_loss = self.trainer.stylize_loss_fn(render_feature_list, self.trainer.ctx.style_features_list)
            content_loss = content_loss_fn(render_feature_list, self.trainer.ctx.scene_features_list[viewpoint_cam.uid])
            image_tv_loss = image_tv_loss_fn(render_image)
            
            render_depth = self.render_pkg["depth"]
            scene_depth = self.trainer.ctx.depth_images[viewpoint_cam.uid]
            depth_loss = torch.mean((render_depth - scene_depth) ** 2)
            
            loss_delta_opacity = torch.norm(self.trainer.gaussians._opacity - self.initial_opacity)
            loss_delta_scaling = torch.norm(self.trainer.gaussians._scaling - self.initial_scaling)
            
            # Check for NaN in intermediate losses
            if torch.isnan(loss_delta_opacity) or torch.isinf(loss_delta_opacity):
                print(f"[Stylize iter={iteration}] ERROR: loss_delta_opacity is NaN/Inf: {loss_delta_opacity.item()}")
            if torch.isnan(loss_delta_scaling) or torch.isinf(loss_delta_scaling):
                print(f"[Stylize iter={iteration}] ERROR: loss_delta_scaling is NaN/Inf: {loss_delta_scaling.item()}")

            sum_E = self.render_pkg["sum_E"]
            sum_E_dx = self.render_pkg["sum_E_dx"]
            sum_E_dy = self.render_pkg["sum_E_dy"]
            sum_E_xx = self.render_pkg["sum_E_xx"]
            sum_E_xy = self.render_pkg["sum_E_xy"]
            sum_E_yy = self.render_pkg["sum_E_yy"]
            sum_E_dt2 = self.render_pkg["sum_E_dt2"]
            sum_E_dn2 = self.render_pkg["sum_E_dn2"]
            
            # Check sum_E values for NaN before using them
            if torch.isnan(sum_E).any():
                print(f"[Stylize iter={iteration}] ERROR: sum_E contains NaN: {torch.isnan(sum_E).sum().item()}")
            if torch.isnan(sum_E_dt2).any():
                print(f"[Stylize iter={iteration}] ERROR: sum_E_dt2 contains NaN: {torch.isnan(sum_E_dt2).sum().item()}")
            if torch.isnan(sum_E_dn2).any():
                print(f"[Stylize iter={iteration}] ERROR: sum_E_dn2 contains NaN: {torch.isnan(sum_E_dn2).sum().item()}")
            
            vis = self.render_pkg["visibility_filter"].squeeze(-1)
            sum_E_v = sum_E[vis]
            sum_E_dx_v = sum_E_dx[vis]
            sum_E_dy_v = sum_E_dy[vis]
            sum_E_xx_v = sum_E_xx[vis]
            sum_E_xy_v = sum_E_xy[vis]
            sum_E_yy_v = sum_E_yy[vis]

            # Check if vis indices are valid
            if vis.numel() > 0 and vis.max() >= sum_E.shape[0]:
                print(f"[Stylize iter={iteration}] ERROR: visibility_filter indices out of range: max={vis.max()}, sum_E.shape={sum_E.shape}")

            # DIAGNOSTIC: Check inputs to brush and stroke losses
            if torch.isnan(sum_E_v).any() or torch.isnan(sum_E_dx_v).any() or torch.isnan(sum_E_dy_v).any():
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: brush_shape_loss inputs contain NaN")
                print(f"  sum_E_v: NaN={torch.isnan(sum_E_v).sum().item()}, shape={sum_E_v.shape}")
                print(f"  sum_E_dx_v: NaN={torch.isnan(sum_E_dx_v).sum().item()}, shape={sum_E_dx_v.shape}")
                print(f"  sum_E_dy_v: NaN={torch.isnan(sum_E_dy_v).sum().item()}, shape={sum_E_dy_v.shape}")
            
            if torch.isnan(sum_E_xx_v).any() or torch.isnan(sum_E_xy_v).any() or torch.isnan(sum_E_yy_v).any():
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: brush_shape_loss inputs contain NaN")
                print(f"  sum_E_xx_v: NaN={torch.isnan(sum_E_xx_v).sum().item()}, shape={sum_E_xx_v.shape}")
                print(f"  sum_E_xy_v: NaN={torch.isnan(sum_E_xy_v).sum().item()}, shape={sum_E_xy_v.shape}")
                print(f"  sum_E_yy_v: NaN={torch.isnan(sum_E_yy_v).sum().item()}, shape={sum_E_yy_v.shape}")
            
            sum_E_dt2_vis = sum_E_dt2[vis]
            sum_E_dn2_vis = sum_E_dn2[vis]
            
            if torch.isnan(sum_E_dt2_vis).any() or torch.isnan(sum_E_dn2_vis).any():
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: stroke_direction_loss inputs contain NaN")
                print(f"  sum_E_dt2[vis]: NaN={torch.isnan(sum_E_dt2_vis).sum().item()}, shape={sum_E_dt2_vis.shape}")
                print(f"  sum_E_dn2[vis]: NaN={torch.isnan(sum_E_dn2_vis).sum().item()}, shape={sum_E_dn2_vis.shape}")
                print(f"  vis count: {vis.sum().item()}, vis max index: {vis.max().item() if vis.numel() > 0 else 0}")
                print(f"  sum_E_dt2 shape: {sum_E_dt2.shape}, sum_E_dn2 shape: {sum_E_dn2.shape}")
            
            brush_shape_loss = brush_shape_loss_fn(sum_E_v, sum_E_dx_v, sum_E_dy_v, sum_E_xx_v, sum_E_xy_v, sum_E_yy_v)
            stroke_direction_loss = stroke_direction_loss_fn(sum_E_dt2_vis, sum_E_dn2_vis)
            
            # DIAGNOSTIC: Check outputs
            if torch.isnan(brush_shape_loss) or torch.isinf(brush_shape_loss):
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: brush_shape_loss output is NaN/Inf: {brush_shape_loss.item()}")
                if sum_E_v.numel() > 0:
                    print(f"  Input stats: sum_E_v range=[{sum_E_v.min().item():.6f}, {sum_E_v.max().item():.6f}], shape={sum_E_v.shape}")
                else:
                    print(f"  Input stats: sum_E_v is empty! shape={sum_E_v.shape}")
            
            if torch.isnan(stroke_direction_loss) or torch.isinf(stroke_direction_loss):
                print(f"[Stylize iter={iteration}] DIAGNOSTIC: stroke_direction_loss output is NaN/Inf: {stroke_direction_loss.item()}")
                if sum_E_dt2_vis.numel() > 0 and sum_E_dn2_vis.numel() > 0:
                    print(f"  Input stats: sum_E_dt2 range=[{sum_E_dt2_vis.min().item():.6f}, {sum_E_dt2_vis.max().item():.6f}], sum_E_dn2 range=[{sum_E_dn2_vis.min().item():.6f}, {sum_E_dn2_vis.max().item():.6f}]")
                else:
                    print(f"  Input stats: sum_E_dt2 shape={sum_E_dt2_vis.shape}, sum_E_dn2 shape={sum_E_dn2_vis.shape}")
            
            # Color preservation loss (Lab a/b channel statistics)
            # Use pre-process render result as reference (better 3D consistency)
            if self.trainer.config.style.enable_ab:
                pre_image = self.trainer.ctx.scene_images[viewpoint_cam.uid]  # [3, H, W] from pre-process phase
                
                # DIAGNOSTIC: Check inputs to ab_loss
                if torch.isnan(render_image).any() or torch.isinf(render_image).any():
                    print(f"[Stylize iter={iteration}] DIAGNOSTIC: ab_loss input render_image contains NaN/Inf")
                    print(f"  render_image: NaN={torch.isnan(render_image).sum().item()}, Inf={torch.isinf(render_image).sum().item()}, shape={render_image.shape}")
                if torch.isnan(pre_image).any() or torch.isinf(pre_image).any():
                    print(f"[Stylize iter={iteration}] DIAGNOSTIC: ab_loss input pre_image contains NaN/Inf")
                    print(f"  pre_image: NaN={torch.isnan(pre_image).sum().item()}, Inf={torch.isinf(pre_image).sum().item()}, shape={pre_image.shape}")
                
                # Compute color preservation loss
                if self.trainer.config.style.use_ab_patch:
                    # Patch-wise version (stronger, for local color shifts)
                    ab_loss = ab_patch_stat_loss(
                        render_rgb=render_image,
                        ref_rgb=pre_image,
                        grid=self.trainer.config.style.ab_patch_grid
                    )
                    lambda_ab = self.trainer.config.style.lambda_ab_patch
                else:
                    # Global statistics version (more stable)
                    ab_loss = ab_stat_loss(
                        render_rgb=render_image,
                        ref_rgb=pre_image
                    )
                    lambda_ab = self.trainer.config.style.lambda_ab
                
                # DIAGNOSTIC: Check ab_loss output
                if torch.isnan(ab_loss) or torch.isinf(ab_loss):
                    print(f"[Stylize iter={iteration}] DIAGNOSTIC: ab_loss output is NaN/Inf: {ab_loss.item()}")
                
                # Schedule: adjust weight based on training progress
                # Early training: stronger weight to lock colors
                # Late training: weaker weight to allow more style freedom
                progress = (iteration - self.begin_iter) / max(1, self.end_iter - self.begin_iter)
                lambda_ab_scheduled = (
                    self.trainer.config.style.lambda_ab_init * (1 - progress) +
                    self.trainer.config.style.lambda_ab_final * progress
                )
                # Use scheduled weight if lambda_ab_init is set, otherwise use fixed weight
                if self.trainer.config.style.lambda_ab_init > 0:
                    lambda_ab = lambda_ab_scheduled
            else:
                # ab loss is disabled, set to zero tensor
                ab_loss = torch.tensor(0.0, device=render_image.device, dtype=render_image.dtype, requires_grad=False)
                lambda_ab = 0.0
            
            # Check for NaN in loss components before combining
            if isinstance(stylize_loss, torch.Tensor):
                if torch.isnan(stylize_loss) or torch.isinf(stylize_loss):
                    print(f"[Stylize iter={iteration}] WARNING: stylize_loss is NaN/Inf: {stylize_loss.item()}")
            else:
                # Convert to tensor if it's not already
                stylize_loss = torch.tensor(float(stylize_loss), device=render_image.device, requires_grad=True)
            if torch.isnan(content_loss) or torch.isinf(content_loss):
                print(f"[Stylize iter={iteration}] WARNING: content_loss is NaN/Inf: {content_loss.item()}")
            if torch.isnan(image_tv_loss) or torch.isinf(image_tv_loss):
                print(f"[Stylize iter={iteration}] WARNING: image_tv_loss is NaN/Inf: {image_tv_loss.item()}")
            if torch.isnan(depth_loss) or torch.isinf(depth_loss):
                print(f"[Stylize iter={iteration}] WARNING: depth_loss is NaN/Inf: {depth_loss.item()}")
            if torch.isnan(brush_shape_loss) or torch.isinf(brush_shape_loss):
                print(f"[Stylize iter={iteration}] WARNING: brush_shape_loss is NaN/Inf: {brush_shape_loss.item()}")
            if torch.isnan(stroke_direction_loss) or torch.isinf(stroke_direction_loss):
                print(f"[Stylize iter={iteration}] WARNING: stroke_direction_loss is NaN/Inf: {stroke_direction_loss.item()}")
            if self.trainer.config.style.enable_ab:
                if torch.isnan(ab_loss) or torch.isinf(ab_loss):
                    print(f"[Stylize iter={iteration}] WARNING: ab_loss is NaN/Inf: {ab_loss.item()}")
            
            # Build loss with optional ab_loss
            loss = (
                self.trainer.config.style.lambda_stylize * stylize_loss +
                self.trainer.config.style.lambda_content * content_loss +
                self.trainer.config.style.lambda_img_tv * image_tv_loss + 
                self.trainer.config.style.lambda_depth * depth_loss +
                self.trainer.config.style.lambda_delta_opacity * loss_delta_opacity +
                self.trainer.config.style.lambda_delta_scaling * loss_delta_scaling +
                self.trainer.config.style.lambda_stroke_direction * stroke_direction_loss +
                self.trainer.config.style.lambda_brush_shape * brush_shape_loss
            )
            if self.trainer.config.style.enable_ab:
                loss = loss + lambda_ab * ab_loss
            
            # Check final loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Stylize iter={iteration}] ERROR: Final loss is NaN/Inf: {loss.item()}")
                ab_str = f"ab={ab_loss.item()}" if self.trainer.config.style.enable_ab else "ab=disabled"
                print(f"  Loss components: stylize={stylize_loss.item()}, content={content_loss.item()}, "
                      f"tv={image_tv_loss.item()}, depth={depth_loss.item()}, "
                      f"brush={brush_shape_loss.item()}, stroke={stroke_direction_loss.item()}, {ab_str}")
            
            # Store loss components for potential diagnosis
            self._loss_components = {
                'stylize': stylize_loss,
                'content': content_loss,
                'tv': image_tv_loss,
                'depth': depth_loss,
                'delta_opacity': loss_delta_opacity,
                'delta_scaling': loss_delta_scaling,
                'brush': brush_shape_loss,
                'stroke': stroke_direction_loss
            }
            if self.trainer.config.style.enable_ab:
                self._loss_components['ab'] = ab_loss
            
            self.update(iteration, loss)
            
        
        result = {
            "P": f"{self.trainer.gaussians._opacity.shape[0]}",
            "L": f"{loss.item():.{4}f}",
            "Sty": f"{stylize_loss.item():.{4}f}",
            "Con": f"{content_loss.item():.{4}f}",
            "Br": f"{brush_shape_loss.item():.{4}f}",
            "St": f"{stroke_direction_loss.item():.{4}f}",
        }
        if self.trainer.config.style.enable_ab:
            result["AB"] = f"{ab_loss.item():.{4}f}"
        return result, timer.elapsed_ms