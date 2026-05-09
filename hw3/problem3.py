import torch
import cvxpy as cp
from problem3_helper import control_limits, f, g

from problem3_helper import NeuralVF
vf = NeuralVF()

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

def smooth_blending_safety_filter(x, u_nom, gamma, lmbda):
    """
    Compute the smooth blending safety filter.
    Refer to the definition provided in the handout.
    You might find it useful to use functions from
    previous homeworks, which we have imported for you.
    These include:
      control_limits(.)
      f(.)
      g(.)
      vf.values(.)
      vf.gradients(.)
    NOTE: some of these functions expect batched inputs,
    but x, u_nom are not batched inputs in this case.
    
    args:
        x:      torch tensor with shape [13]
        u_nom:  torch tensor with shape [4]
        
    returns:
        u_sb:   torch tensor with shape [4]
    """
    # YOUR CODE HERE
    # raise NotImplementedError # REMOVE THIS LINE

    # control bounds
    u_upper, u_lower = control_limits()
    u_upper = u_upper.numpy() # convert to numpy for CVXPY
    u_lower = u_lower.numpy()

    # batch x for vf stuff / things that expect batched inputs
    x_batch = x.unsqueeze(0)  # [1, 13]
    
    # min value func tracker
    V_min = float('inf')
    # its gradient, tracker
    dVdx_min = None
    
    # V, dVdx for all obstacles, get min V
    for obstacle in obstacles:
        # x, y, radius of each obstacle
        o_x, o_y, o_r = obstacle[0].item(), obstacle[1].item(), obstacle[2].item()

        # only p_x and p_y modified, other 11 states unchanged
        # shift state for new obstacle position (prob 3.1)
        x_new = x_batch.clone()
        x_new[0, 0] = x_batch[0, 0] - o_x
        x_new[0, 1] = x_batch[0, 1] - o_y

        # scale for new obstacle radius (prob 3.2)
        x_new[0, 0] = x_new[0, 0] * (0.5 / o_r)  # p_x
        x_new[0, 1] = x_new[0, 1] * (0.5 / o_r)  # p_y
        x_new[0, 7] = x_new[0, 7] * (0.5 / o_r)  # v_x
        x_new[0, 8] = x_new[0, 8] * (0.5 / o_r)  # v_y

        # query og value func at new state
        V_obstacle = vf.values(x_new).item()
        dVdx_obstacle = vf.gradients(x_new)[0].numpy()
        dVdx_obstacle[0] *= (0.5 / o_r)  # p_x
        dVdx_obstacle[1] *= (0.5 / o_r)  # p_y
        dVdx_obstacle[7] *= (0.5 / o_r)  # v_x
        dVdx_obstacle[8] *= (0.5 / o_r)  # v_y
        
        if V_obstacle < V_min:
            V_min = V_obstacle
            dVdx_min = dVdx_obstacle
    
    # convert to numpy
    x_np = x.numpy()
    u_nom_np = u_nom.numpy()

    # control affine dynamics from prob 1, remove batch dim
    f_np = f(x_batch)[0].detach().numpy()   # [13]
    g_np = g(x_batch)[0].detach().numpy()   # [13, 4]
    
    # QP vars
    u_sb = cp.Variable(4) # control input
    s = cp.Variable(1)  # slack variable
    
    # V_dot = dVdx @ (f(x) + g(x)u)
    # Lie derivatives
    Lf_V = dVdx_min @ f_np # scalar
    Lg_V = dVdx_min @ g_np  # [4]
    
    # (over u) min ||u - u_nom||^2 + lambda * s^2
    # stay closest to nominal controller, penalize slack
    objective = cp.Minimize(cp.sum_squares(u_sb - u_nom_np) + lmbda * cp.sum_squares(s))
    
    # constraint: dV/dx * (f + g*u) + gamma*V + s >= 0
    constraints = [ Lf_V + Lg_V @ u_sb + gamma * V_min + s >= 0,
                    u_sb <= u_upper,
                    u_sb >= u_lower,
                    s >= 0 ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if u_sb.value is None:
        return u_nom  # fallback to nominal if QP fails

    return torch.tensor(u_sb.value, dtype=torch.float32) # NOTE: ensure you return a float32 tensor

