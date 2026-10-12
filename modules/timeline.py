import math
from modules.plan_io import Plan

def compute_timeline(plan: Plan):
    """
    Computes frame-accurate timeline information for each shot.
    Returns a list of dictionaries with keys:
      nominal_start, nominal_end, head, tail, fade_frames, render_from, render_frames
    """
    fps = plan.output.fps
    
    # Calculate nominal boundaries B_k
    boundaries = [0]
    cumulative = 0.0
    for shot in plan.shots:
        cumulative += shot.duration_s
        boundaries.append(round(cumulative * fps))
        
    # Calculate D_k
    D_k = [0] * len(plan.shots)
    for i, shot in enumerate(plan.shots):
        trans = shot.transition_in
        if trans.type == "fade":
            D_k[i] = round(trans.duration_s * fps)
        else:
            D_k[i] = 0
            
    # Add a dummy D_{n+1} = 0 for the end
    D_k.append(0)
    
    timeline = []
    for i, shot in enumerate(plan.shots):
        b_k = boundaries[i]
        b_next = boundaries[i+1]
        d_k = D_k[i]
        d_next = D_k[i+1]
        
        head = math.floor(d_k / 2)
        tail = math.ceil(d_next / 2)
        
        render_from = b_k - head
        render_frames = (b_next + tail) - render_from
        
        timeline.append({
            "nominal_start": b_k,
            "nominal_end": b_next,
            "head": head,
            "tail": tail,
            "fade_frames": d_k,
            "render_from": render_from,
            "render_frames": render_frames
        })
        
    return timeline
