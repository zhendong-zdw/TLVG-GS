from .base_loss import StylizeLoss
import torch
import torch.nn.functional as F


class FASTLoss(StylizeLoss):
    
    @torch.no_grad
    def cal_p(self, cf, sf):
        cf_size = cf.size()
        sf_size = sf.size()
        
        k_cross = 5

        cf_temp = cf
        sf_temp = sf

        cf_n = F.normalize(cf, 2, 0)
        sf_n = F.normalize(sf, 2, 0)
        
        dist = torch.mm(cf_n.t(), sf_n)  # inner product,the larger the value, the more similar

        hcwc, hsws = cf_size[1], sf_size[1]
        U = torch.zeros(hcwc, hsws).type_as(cf_n).to(self.config.model.data_device)  # construct affinity matrix "(h*w)*(h*w)"

        index = torch.topk(dist, k_cross, 0)[1]  # find indices k nearest neighbors along row dimension
        value = torch.ones(k_cross, hsws).type_as(cf_n).to(self.config.model.data_device) # "KCross*(h*w)"
        U.scatter_(0, index, value)  # set weight matrix

        index = torch.topk(dist, k_cross, 1)[1]  # find indices k nearest neighbors along col dimension
        value = torch.ones(hcwc, k_cross).type_as(cf_n).to(self.config.model.data_device)
        U.scatter_(1, index, value)  # set weight matrix
        
        n_cs = torch.sum(U)
        U = U / n_cs
        D1 = torch.diag(torch.sum(U, dim=1)).type_as(cf).to(self.config.model.data_device)
        
        A = torch.mm(torch.mm(cf_temp, D1), cf_temp.t())
        regularization_term = torch.eye(A.size()[0]).type_as(A).to(self.config.model.data_device) * 1e-12
        A += regularization_term
        B = torch.mm(torch.mm(cf_temp, U), sf_temp.t())
        
        try:
            p = torch.linalg.solve(A, B)
        except Exception as e:
            print(e)
            p = torch.eye(cf_size[0]).type_as(cf).to(self.config.model.data_device)
        return p


    @torch.no_grad
    def transform(self, render_feats, style_feats):
        p = self.cal_p(render_feats, style_feats)
        return torch.mm(p.t(), render_feats)
    
    
    def __call__(self, render_feats_list, style_feats_list):
        
        def cos_loss(a, b, eps=1e-8):
            # DIAGNOSTIC: Check for NaN/Inf in inputs - report but don't fix
            if torch.isnan(a).any() or torch.isinf(a).any():
                nan_count = torch.isnan(a).sum().item()
                inf_count = torch.isinf(a).sum().item()
                print(f"DIAGNOSTIC: cos_loss input 'a' contains NaN={nan_count}, Inf={inf_count}")
                print(f"  a stats: shape={a.shape}, finite_count={torch.isfinite(a).sum().item()}")
            if torch.isnan(b).any() or torch.isinf(b).any():
                nan_count = torch.isnan(b).sum().item()
                inf_count = torch.isinf(b).sum().item()
                print(f"DIAGNOSTIC: cos_loss input 'b' contains NaN={nan_count}, Inf={inf_count}")
                print(f"  b stats: shape={b.shape}, finite_count={torch.isfinite(b).sum().item()}")
            
            # Check feature norms - cosine similarity can fail with zero norm
            a_norm = a.norm(dim=1, keepdim=True)
            b_norm = b.norm(dim=1, keepdim=True)
            
            # DIAGNOSTIC: Report zero or very small norms
            zero_norm_a = (a_norm < eps).sum().item()
            zero_norm_b = (b_norm < eps).sum().item()
            if zero_norm_a > 0:
                print(f"DIAGNOSTIC: cos_loss has {zero_norm_a} features with near-zero norm in 'a'")
            if zero_norm_b > 0:
                print(f"DIAGNOSTIC: cos_loss has {zero_norm_b} features with near-zero norm in 'b'")
            
            # Clamp norms to prevent division by zero
            a_norm = torch.clamp(a_norm, min=eps)
            b_norm = torch.clamp(b_norm, min=eps)
            
            # Normalize manually for better numerical stability
            a_normalized = a / a_norm
            b_normalized = b / b_norm
            
            # Compute cosine similarity
            cossim = (a_normalized * b_normalized).sum(dim=1)
            
            # Clamp to valid range [-1, 1] to prevent numerical issues
            cossim = torch.clamp(cossim, min=-1.0, max=1.0)
            
            # DIAGNOSTIC: Check for NaN/Inf in cosine similarity
            if torch.isnan(cossim).any() or torch.isinf(cossim).any():
                nan_count = torch.isnan(cossim).sum().item()
                inf_count = torch.isinf(cossim).sum().item()
                print(f"DIAGNOSTIC: cossim contains NaN={nan_count}, Inf={inf_count}")
                print(f"  a_norm range: [{a_norm.min().item():.6f}, {a_norm.max().item():.6f}], b_norm range: [{b_norm.min().item():.6f}, {b_norm.max().item():.6f}]")
                print(f"  a_normalized stats: min={a_normalized.min().item():.6f}, max={a_normalized.max().item():.6f}, mean={a_normalized.mean().item():.6f}")
                print(f"  b_normalized stats: min={b_normalized.min().item():.6f}, max={b_normalized.max().item():.6f}, mean={b_normalized.mean().item():.6f}")
            
            loss = (1.0 - cossim).mean()
            
            # DIAGNOSTIC: Final check
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"DIAGNOSTIC: cos_loss output is NaN/Inf: loss={loss.item()}")
                print(f"  cossim stats: min={cossim.min().item():.6f}, max={cossim.max().item():.6f}, mean={cossim.mean().item():.6f}")
            
            return loss
        
        # Initialize as tensor to avoid type errors when all classes are skipped
        device = render_feats_list[0].device if len(render_feats_list) > 0 else "cuda"
        dtype = render_feats_list[0].dtype if len(render_feats_list) > 0 else torch.float32
        stylize_loss = torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
        
        valid_losses = 0
        for i in range(self.config.style.scene_classes):
            a = render_feats_list[i]
            b = self.transform(a, style_feats_list[self.config.style.override_matches[i]])
            
            # DIAGNOSTIC: Check transform output for NaN/Inf - report but don't skip
            if torch.isnan(b).any() or torch.isinf(b).any():
                nan_count = torch.isnan(b).sum().item()
                inf_count = torch.isinf(b).sum().item()
                print(f"DIAGNOSTIC: transform output for class {i} contains NaN={nan_count}, Inf={inf_count}")
                print(f"  b stats: shape={b.shape}, finite_count={torch.isfinite(b).sum().item()}")
                # Don't skip - let it propagate to see where NaN comes from
            
            loss_i = cos_loss(a, b)
            stylize_loss = stylize_loss + loss_i
            valid_losses += 1
        
        # If no valid losses, return zero tensor
        if valid_losses == 0:
            print(f"WARNING: All stylize loss components were skipped due to NaN/Inf, returning zero loss")
            return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
                
        return stylize_loss