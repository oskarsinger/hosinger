from linal.utils import get_lp_norm_gradient, quadratic, multi_dot
from linal.svd_funcs import get_schatten_p_norm as get_sp, get_svd_power

l2_breg_div, l2_breg_grad = get_lp_bregman_div_and_grad(2)

def get_lp_bregman_div_and_grad(p, ip=np.dot):

    breg_func = lambda x: np.linalg.norm(x, ord=p)
    breg_grad = lambda x: get_lp_norm_gradient(x, p) 

    return get_bregman_div_and_grad(breg_func, breg_grad, ip=ip)

def get_bregman_div_and_grad(
    get_bregman_func,
    get_bregman_func_grad, 
    get_ip=np.dot):

    def get_bregman_div(x, x_t):

        grad = get_bregman_func_grad(x_t)
        diff = x - x_t
        ip = get_ip(grad, diff)
        x_breg = get_bregman_func(x)
        x_t_breg = get_bregman_func(x_t)

        return x_breg - x_t_breg - ip

    def get_bregman_grad(x, x_t):

        x_t_grad = get_bregman_func_grad(x_t)
        x_t_ip = get_ip(x_t_grad, x_t)
        x_grad = get_bregman_func_grad(x)
        x_t_breg = get_bregman_func(x_t)

        return x_grad - x_t_grad - x_t_breg + x_t_ip

    return (get_bregman_div, get_bregman_grad)
