from tensorflow.contrib.opt import ExternalOptimizerInterface
from lbfgs import fmin_lbfgs

class PylbfgsInterface(ExternalOptimizerInterface):
    def _minimize(self, initial_val, loss_grad_func, equality_funcs,
            equality_grad_funcs, inequality_funcs, inequality_grad_funcs,
            packed_bounds, step_callback, optimizer_kwargs):
        """Wrapper for a particular optimization algorithm implementation.
        It would be appropriate for a subclass implementation of this method to
        raise `NotImplementedError` if unsupported arguments are passed: e.g. if an
        algorithm does not support constraints but `len(equality_funcs) > 0`.
        Args:
        initial_val: A NumPy vector of initial values.
        loss_grad_func: A function accepting a NumPy packed variable vector and
            returning two outputs, a loss value and the gradient of that loss with
            respect to the packed variable vector.
        equality_funcs: A list of functions each of which specifies a scalar
            quantity that an optimizer should hold exactly zero.
        equality_grad_funcs: A list of gradients of equality_funcs.
        inequality_funcs: A list of functions each of which specifies a scalar
            quantity that an optimizer should hold >= 0.
        inequality_grad_funcs: A list of gradients of inequality_funcs.
        packed_bounds: A list of bounds for each index, or `None`.
        step_callback: A callback function to execute at each optimization step,
            supplied with the current value of the packed variable vector.
        optimizer_kwargs: Other key-value arguments available to the optimizer.
        Returns:
        The optimal variable vector as a NumPy vector.
        """
        def loss_grad_func_pylbfgs(x, g):
            xval, gval = loss_grad_func(x)
            g[:] = gval
            return xval
        return fmin_lbfgs(loss_grad_func_pylbfgs, initial_val, progress=None, **optimizer_kwargs)