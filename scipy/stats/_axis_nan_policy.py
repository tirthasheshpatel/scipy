# Many scipy.stats functions support `axis` and `nan_policy` parameters.
# When the two are combined, it can be tricky to get all the behavior just
# right. This file contains utility functions useful for scipy.stats functions
# that support `axis` and `nan_policy`, including a decorator that
# automatically adds `axis` and `nan_policy` arguments to a function.

import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError
from scipy._lib._array_api import array_namespace, as_xparray, atleast_nd
import inspect


def _process_axis(axis, n_dims, xp):
    if axis is not None:
        axis = atleast_nd(axis, ndim=1, xp=xp)
        axis_int = xp.astype(axis, xp.int64)
        if not (axis_int.shape == axis.shape and xp.all(axis_int == axis)):
            raise AxisError('`axis` must be an integer, a '
                            'tuple of integers, or `None`.')
        axis = axis_int

        # TODO(tirthasheshpatel): Jax/TensorFlow don't allow inplace updates.
        # Try to implement the inplace update below but using copies.
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = xp.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            raise np.AxisError(message)

        if len(xp.unique(axis)) != len(axis):
            raise np.AxisError("`axis` must contain only distinct elements")
    return axis


def _broadcast_arrays(arrays, xp, axis=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    new_shapes = _broadcast_array_shapes(arrays, xp, axis=axis)
    if axis is None:
        new_shapes = [new_shapes]*len(arrays)
    return [xp.broadcast_to(array, new_shape)
            for array, new_shape in zip(arrays, new_shapes)]


def _broadcast_array_shapes(arrays, xp, axis=None):
    """
    Broadcast shapes of arrays, ignoring incompatibility of specified axes
    """
    shapes = [as_xparray(arr, xp=xp).shape for arr in arrays]
    return _broadcast_shapes(shapes, xp, axis)


def _broadcast_shapes(shapes, xp, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = xp.ones((len(shapes), n_dims), dtype=xp.int64)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        removed_shapes = new_shapes[:, axis]
        # Array API doesn't contain delete. replicate it using masks.
        # This is equivalent to:
        # new_shapes = np.delete(new_shapes, axis, axis=1)
        mask = xp.ones(new_shapes.shape, dtype=xp.bool)
        mask[:, axis] = 0
        new_shapes = new_shapes[mask]

    # If arrays are broadcastable, shape elements that are 1 may be replaced
    # with a corresponding non-1 shape element. Assuming arrays are
    # broadcastable, that final shape element can be found with:
    new_shape = xp.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= xp.all(new_shapes, axis=0)

    # Among all arrays, there can only be one unique non-1 shape element.
    # Therefore, if any non-1 shape element does not match what we found
    # above, the arrays must not be broadcastable after all.
    if xp.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - xp.arange(len(axis))
        new_shapes = []
        for removed_shape in removed_shapes:
            new_shape = tuple(new_shape)
            new_shape = new_shape[:new_axis] + (removed_shape,) + new_shape[new_axis+1:]
            new_shapes.append(new_shape)
        return new_shapes
    else:
        return tuple(new_shape)


def _broadcast_array_shapes_remove_axis(arrays, xp, axis=None):
    """
    Broadcast shapes of arrays, dropping specified axes

    Given a sequence of arrays `arrays` and an integer or tuple `axis`, find
    the shape of the broadcast result after consuming/dropping `axis`.
    In other words, return output shape of a typical hypothesis test on
    `arrays` vectorized along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.zeros((5, 2, 1))
    >>> b = np.zeros((9, 3))
    >>> _broadcast_array_shapes((a, b), 1)
    (5, 3)
    """
    # Note that here, `axis=None` means do not consume/drop any axes - _not_
    # ravel arrays before broadcasting.
    shapes = [arr.shape for arr in arrays]
    return _broadcast_shapes_remove_axis(shapes, xp, axis)


def _broadcast_shapes_remove_axis(shapes, xp, axis=None):
    """
    Broadcast shapes, dropping specified axes

    Same as _broadcast_array_shapes, but given a sequence
    of array shapes `shapes` instead of the arrays themselves.
    """
    shapes = _broadcast_shapes(shapes, xp, axis)
    shape = tuple(shapes[0])
    if axis is not None:
        shape = shape[:axis] + shape[axis+1:]
    return tuple(shape)


def _broadcast_concatenate(arrays, axis, xp):
    """Concatenate arrays along an axis with broadcasting."""
    arrays = _broadcast_arrays(arrays, xp, axis)
    res = xp.concatenate(arrays, axis=axis)
    return res


# TODO: add support for `axis` tuples
def _remove_nans(samples, paired, xp):
    "Remove nans from paired or unpaired 1D samples"
    # potential optimization: don't copy arrays that don't contain nans
    if not paired:
        return [sample[~xp.isnan(sample)] for sample in samples]

    # for paired samples, we need to remove the whole pair when any part
    # has a nan
    nans = xp.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | xp.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]


def _remove_sentinel(samples, paired, sentinel):
    "Remove sentinel values from paired or unpaired 1D samples"
    # could consolidate with `_remove_nans`, but it's not quite as simple as
    # passing `sentinel=np.nan` because `(np.nan == np.nan) is False`

    # potential optimization: don't copy arrays that don't contain sentinel
    if not paired:
        return [sample[sample != sentinel] for sample in samples]

    # for paired samples, we need to remove the whole pair when any part
    # has a nan
    sentinels = (samples[0] == sentinel)
    for sample in samples[1:]:
        sentinels = sentinels | (sample == sentinel)
    not_sentinels = ~sentinels
    return [sample[not_sentinels] for sample in samples]


def _masked_arrays_2_sentinel_arrays(samples, xp):
    # masked arrays in `samples` are converted to regular arrays, and values
    # corresponding with masked elements are replaced with a sentinel value

    # return without modifying arrays if none have a mask
    has_mask = False
    for sample in samples:
        mask = getattr(sample, 'mask', False)
        has_mask = has_mask or xp.any(mask)
    if not has_mask:
        return samples, None  # None means there is no sentinel value

    # Choose a sentinel value. We can't use `np.nan`, because sentinel (masked)
    # values are always omitted, but there are different nan policies.
    dtype = xp.result_type(*samples)
    # TODO(tirthasheshpatel): Verify that `isdtype(x, 'numeric')` is
    #                         equivalent to `issubdtype(x, np.number)`
    dtype = dtype if xp.isdtype(dtype, 'numeric') else xp.float64
    for i in range(len(samples)):
        # Things get more complicated if the arrays are of different types.
        # We could have different sentinel values for each array, but
        # the purpose of this code is convenience, not efficiency.
        samples[i] = xp.astype(samples[i], dtype, copy=False)

    inexact = xp.isdtype(dtype, ('real floating', 'complex floating'))
    info = xp.finfo if inexact else xp.iinfo
    max_possible, min_possible = info(dtype).max, info(dtype).min
    nextafter = xp.nextafter if inexact else (lambda x, _: x - 1)

    sentinel = max_possible
    # For simplicity, min_possible/np.infs are not candidate sentinel values
    while sentinel > min_possible:
        for sample in samples:
            if xp.any(sample == sentinel):  # choose a new sentinel value
                sentinel = nextafter(sentinel, -xp.inf)
                break
        else:  # when sentinel value is OK, break the while loop
            break
    else:
        message = ("This function replaces masked elements with sentinel "
                   "values, but the data contains all distinct values of this "
                   "data type. Consider promoting the dtype to `np.float64`.")
        raise ValueError(message)

    # replace masked elements with sentinel value
    out_samples = []
    for sample in samples:
        mask = getattr(sample, 'mask', None)
        if mask is not None:  # turn all masked arrays into sentinel arrays
            # Since only NumPy has masked arrays, we don't need to use xp here.
            mask = np.broadcast_to(mask, sample.shape)
            sample = sample.data.copy() if np.any(mask) else sample.data
            sample = np.asarray(sample)  # `sample.data` could be a memoryview?
            sample[mask] = sentinel
        out_samples.append(sample)

    return out_samples, sentinel


def _check_empty_inputs(samples, axis, xp):
    """
    Check for empty sample; return appropriate output for a vectorized hypotest
    """
    # if none of the samples are empty, we need to perform the test
    if not any(sample.size == 0 for sample in samples):
        return None
    # otherwise, the statistic and p-value will be either empty arrays or
    # arrays with NaNs. Produce the appropriate array and return it.
    output_shape = _broadcast_array_shapes_remove_axis(samples, axis)
    output = xp.ones(output_shape) * xp.nan
    return output


def _add_reduced_axes(res, reduced_axes, keepdims, xp):
    """
    Add reduced axes back to all the arrays in the result object
    if keepdims = True.
    """
    return ([xp.expand_dims(output, reduced_axes) for output in res]
            if keepdims else res)


def _xp_apply(func1d, arr, xp):
    res = None
    for idx in range(arr.shape[0]):
        out = func1d(arr[idx])
        # Kind of a hack to check if the output is an array. Array API
        # doesn't have a construct that can be used by `isinstance` to check
        # if an object is an array.
        if not hasattr(out, "shape"):
            out = tuple(out)
        if res is None:
            if isinstance(out, tuple):
                res = (xp.expand_dims(o, 0) for o in out)
            else:
                res = xp.expand_dims(out, 0)
        if isinstance(res, tuple):
            res = (xp.concatenate(r, xp.expand_dims(o, 0), axis=0)
                   for r, o in zip(res, out))
        else:
            res = xp.concatenate(res, xp.expand_dims(out, 0), axis=0)
    if res is None:
        # Input array is empty
        raise NotImplementedError("shouldn't happen.")
    return res


# Standard docstring / signature entries for `axis`, `nan_policy`, `keepdims`
_name = 'axis'
_desc = (
    """If an int, the axis of the input along which to compute the statistic.
The statistic of each axis-slice (e.g. row) of the input will appear in a
corresponding element of the output.
If ``None``, the input will be raveled before computing the statistic."""
    .split('\n'))


def _get_axis_params(default_axis=0, _name=_name, _desc=_desc):  # bind NOW
    _type = f"int or None, default: {default_axis}"
    _axis_parameter_doc = Parameter(_name, _type, _desc)
    _axis_parameter = inspect.Parameter(_name,
                                        inspect.Parameter.KEYWORD_ONLY,
                                        default=default_axis)
    return _axis_parameter_doc, _axis_parameter


_name = 'nan_policy'
_type = "{'propagate', 'omit', 'raise'}"
_desc = (
    """Defines how to handle input NaNs.

- ``propagate``: if a NaN is present in the axis slice (e.g. row) along
  which the  statistic is computed, the corresponding entry of the output
  will be NaN.
- ``omit``: NaNs will be omitted when performing the calculation.
  If insufficient data remains in the axis slice along which the
  statistic is computed, the corresponding entry of the output will be
  NaN.
- ``raise``: if a NaN is present, a ``ValueError`` will be raised."""
    .split('\n'))
_nan_policy_parameter_doc = Parameter(_name, _type, _desc)
_nan_policy_parameter = inspect.Parameter(_name,
                                          inspect.Parameter.KEYWORD_ONLY,
                                          default='propagate')

_name = 'keepdims'
_type = "bool, default: False"
_desc = (
    """If this is set to True, the axes which are reduced are left
in the result as dimensions with size one. With this option,
the result will broadcast correctly against the input array."""
    .split('\n'))
_keepdims_parameter_doc = Parameter(_name, _type, _desc)
_keepdims_parameter = inspect.Parameter(_name,
                                        inspect.Parameter.KEYWORD_ONLY,
                                        default=False)

_standard_note_addition = (
    """\nBeginning in SciPy 1.9, ``np.matrix`` inputs (not recommended for new
code) are converted to ``np.ndarray`` before the calculation is performed. In
this case, the output will be a scalar or ``np.ndarray`` of appropriate shape
rather than a 2D ``np.matrix``. Similarly, while masked elements of masked
arrays are ignored, the output will be a scalar or ``np.ndarray`` rather than a
masked array with ``mask=False``.""").split('\n')


def _axis_nan_policy_factory(tuple_to_result, default_axis=0,
                             n_samples=1, paired=False,
                             result_to_tuple=None, too_small=0,
                             n_outputs=2, kwd_samples=[], override=None):
    """Factory for a wrapper that adds axis/nan_policy params to a function.

    Parameters
    ----------
    tuple_to_result : callable
        Callable that returns an object of the type returned by the function
        being wrapped (e.g. the namedtuple or dataclass returned by a
        statistical test) provided the separate components (e.g. statistic,
        pvalue).
    default_axis : int, default: 0
        The default value of the axis argument. Standard is 0 except when
        backwards compatibility demands otherwise (e.g. `None`).
    n_samples : int or callable, default: 1
        The number of data samples accepted by the function
        (e.g. `mannwhitneyu`), a callable that accepts a dictionary of
        parameters passed into the function and returns the number of data
        samples (e.g. `wilcoxon`), or `None` to indicate an arbitrary number
        of samples (e.g. `kruskal`).
    paired : {False, True}
        Whether the function being wrapped treats the samples as paired (i.e.
        corresponding elements of each sample should be considered as different
        components of the same sample.)
    result_to_tuple : callable, optional
        Function that unpacks the results of the function being wrapped into
        a tuple. This is essentially the inverse of `tuple_to_result`. Default
        is `None`, which is appropriate for statistical tests that return a
        statistic, pvalue tuple (rather than, e.g., a non-iterable datalass).
    too_small : int, default: 0
        The largest unnacceptably small sample for the function being wrapped.
        For example, some functions require samples of size two or more or they
        raise an error. This argument prevents the error from being raised when
        input is not 1D and instead places a NaN in the corresponding element
        of the result.
    n_outputs : int or callable, default: 2
        The number of outputs produced by the function given 1d sample(s). For
        example, hypothesis tests that return a namedtuple or result object
        with attributes ``statistic`` and ``pvalue`` use the default
        ``n_outputs=2``; summary statistics with scalar output use
        ``n_outputs=1``. Alternatively, may be a callable that accepts a
        dictionary of arguments passed into the wrapped function and returns
        the number of outputs corresponding with those arguments.
    kwd_samples : sequence, default: []
        The names of keyword parameters that should be treated as samples. For
        example, `gmean` accepts as its first argument a sample `a` but
        also `weights` as a fourth, optional keyword argument. In this case, we
        use `n_samples=1` and kwd_samples=['weights'].
    override : dict, default: {'vectorization': False, 'nan_propagation': True}
        Pass a dictionary with ``'vectorization': True`` to ensure that the
        decorator overrides the function's behavior for multimensional input.
        Use ``'nan_propagation': False`` to ensure that the decorator does not
        override the function's behavior for ``nan_policy='propagate'``.
        (See `scipy.stats.mode`, for example.)
    """
    # Specify which existing behaviors the decorator must override
    temp = override or {}
    override = {'vectorization': False,
                'nan_propagation': True}
    override.update(temp)

    if result_to_tuple is None:
        def result_to_tuple(res):
            return res

    def is_too_small(samples):
        for sample in samples:
            if len(sample) <= too_small:
                return True
        return False

    def axis_nan_policy_decorator(hypotest_fun_in):
        @wraps(hypotest_fun_in)
        def axis_nan_policy_wrapper(*args, _no_deco=False, **kwds):

            if _no_deco:  # for testing, decorator does nothing
                return hypotest_fun_in(*args, **kwds)

            # We need to be flexible about whether position or keyword
            # arguments are used, but we need to make sure users don't pass
            # both for the same parameter. To complicate matters, some
            # functions accept samples with *args, and some functions already
            # accept `axis` and `nan_policy` as positional arguments.
            # The strategy is to make sure that there is no duplication
            # between `args` and `kwds`, combine the two into `kwds`, then
            # the samples, `nan_policy`, and `axis` from `kwds`, as they are
            # dealt with separately.

            # Check for intersection between positional and keyword args
            params = list(inspect.signature(hypotest_fun_in).parameters)
            if n_samples is None:
                # Give unique names to each positional sample argument
                # Note that *args can't be provided as a keyword argument
                params = [f"arg{i}" for i in range(len(args))] + params[1:]

            # raise if there are too many positional args
            maxarg = (float('inf') if inspect.getfullargspec(hypotest_fun_in).varargs
                      else len(inspect.getfullargspec(hypotest_fun_in).args))
            if len(args) > maxarg:  # let the function raise the right error
                hypotest_fun_in(*args, **kwds)

            # raise if multiple values passed for same parameter
            d_args = dict(zip(params, args))
            intersection = set(d_args) & set(kwds)
            if intersection:  # let the function raise the right error
                hypotest_fun_in(*args, **kwds)

            # Consolidate other positional and keyword args into `kwds`
            kwds.update(d_args)

            # rename avoids UnboundLocalError
            if callable(n_samples):
                # Future refactoring idea: no need for callable n_samples.
                # Just replace `n_samples` and `kwd_samples` with a single
                # list of the names of all samples, and treat all of them
                # as `kwd_samples` are treated below.
                n_samp = n_samples(kwds)
            else:
                n_samp = n_samples or len(args)

            # get the number of outputs
            n_out = n_outputs  # rename to avoid UnboundLocalError
            if callable(n_out):
                n_out = n_out(kwds)

            # If necessary, rearrange function signature: accept other samples
            # as positional args right after the first n_samp args
            kwd_samp = [name for name in kwd_samples
                        if kwds.get(name, None) is not None]
            n_kwd_samp = len(kwd_samp)
            if not kwd_samp:
                hypotest_fun_out = hypotest_fun_in
            else:
                def hypotest_fun_out(*samples, **kwds):
                    new_kwds = dict(zip(kwd_samp, samples[n_samp:]))
                    kwds.update(new_kwds)
                    return hypotest_fun_in(*samples[:n_samp], **kwds)

            # Extract the things we need here
            try:  # if something is missing
                samples = [kwds.pop(param)
                           for param in (params[:n_samp] + kwd_samp)]
                xp = array_namespace(*samples)
                samples = [atleast_nd(sample, ndim=1, xp=xp)
                           for sample in samples]
            except KeyError:  # let the function raise the right error
                # might need to revisit this if required arg is not a "sample"
                hypotest_fun_in(*args, **kwds)
            vectorized = True if 'axis' in params else False
            vectorized = vectorized and not override['vectorization']
            axis = kwds.pop('axis', default_axis)
            nan_policy = kwds.pop('nan_policy', 'propagate')
            keepdims = kwds.pop("keepdims", False)
            del args  # avoid the possibility of passing both `args` and `kwds`

            # convert masked arrays to regular arrays with sentinel values
            samples, sentinel = _masked_arrays_2_sentinel_arrays(samples, xp)
            n_dims = xp.max([sample.ndim for sample in samples],
                            initial=0)
            axis = _process_axis(axis, n_dims, xp)

            # standardize to always work along last axis
            reduced_axes = axis
            if axis is None:
                if samples:
                    # when axis=None, take the maximum of all dimensions since
                    # all the dimensions are reduced.
                    reduced_axes = tuple(range(n_dims))
                samples = [as_xparray(sample.reshape(-1)) for sample in samples]
            else:
                samples = _broadcast_arrays(samples, xp, axis=axis)
                n_axes = len(axis)
                # move all axes in `axis` to the end to be raveled
                # Array API doesn't define `moveaxis` in its standard,
                # use permute_dims instead.
                for idx, sample in enumerate(samples):
                    dims = [x for x in range(n_dims) if x not in axis] + list(axis)
                    samples[idx] = xp.permute_dims(sample, dims)
                shapes = [sample.shape for sample in samples]
                # New shape is unchanged for all axes _not_ in `axis`
                # At the end, we append the product of the shapes of the axes
                # in `axis`. Appending -1 doesn't work for zero-size arrays!
                new_shapes = [shape[:-n_axes] + (xp.prod(shape[-n_axes:]),)
                              for shape in shapes]
                samples = [sample.reshape(new_shape)
                           for sample, new_shape in zip(samples, new_shapes)]
            axis = -1  # work over the last axis

            # if axis is not needed, just handle nan_policy and return
            ndims = xp.array([sample.ndim for sample in samples])
            if xp.all(ndims <= 1):
                # Addresses nan_policy == "raise"
                if nan_policy != 'propagate' or override['nan_propagation']:
                    contains_nan = [_contains_nan(sample, nan_policy)[0]
                                    for sample in samples]
                else:
                    # Behave as though there are no NaNs (even if there are)
                    contains_nan = [False]*len(samples)

                # Addresses nan_policy == "propagate"
                if any(contains_nan) and (nan_policy == 'propagate'
                                          and override['nan_propagation']):
                    res = xp.full(n_out, xp.nan)
                    res = _add_reduced_axes(res, reduced_axes, keepdims, xp)
                    return tuple_to_result(*res)

                # Addresses nan_policy == "omit"
                if any(contains_nan) and nan_policy == 'omit':
                    # consider passing in contains_nan
                    samples = _remove_nans(samples, paired, xp)

                # ideally, this is what the behavior would be:
                # if is_too_small(samples):
                #     return tuple_to_result(np.nan, np.nan)
                # but some existing functions raise exceptions, and changing
                # behavior of those would break backward compatibility.

                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)
                res = hypotest_fun_out(*samples, **kwds)
                res = result_to_tuple(res)
                res = _add_reduced_axes(res, reduced_axes, keepdims, xp)
                return tuple_to_result(*res)

            # check for empty input
            # ideally, move this to the top, but some existing functions raise
            # exceptions for empty input, so overriding it would break
            # backward compatibility.
            empty_output = _check_empty_inputs(samples, axis, xp)
            if empty_output is not None:
                from copy import copy
                # Note: Tensors with gradients enabled can't be
                # copied unless they are the leaf node in the
                # computation graph. This might fail.
                res = [copy(empty_output) for i in range(n_out)]
                res = _add_reduced_axes(res, reduced_axes, keepdims, xp)
                return tuple_to_result(*res)

            # otherwise, concatenate all samples along axis, remembering where
            # each separate sample begins
            lengths = xp.array([sample.shape[axis] for sample in samples])
            split_indices = xp.cumsum(lengths)
            x = _broadcast_concatenate(samples, axis, xp)

            # Addresses nan_policy == "raise"
            if nan_policy != 'propagate' or override['nan_propagation']:
                contains_nan, _ = _contains_nan(x, nan_policy)
            else:
                contains_nan = False  # behave like there are no NaNs

            if vectorized and not contains_nan and not sentinel:
                res = hypotest_fun_out(*samples, axis=axis, **kwds)
                res = result_to_tuple(res)
                res = _add_reduced_axes(res, reduced_axes, keepdims, xp)
                return tuple_to_result(*res)

            # Addresses nan_policy == "omit"
            if contains_nan and nan_policy == 'omit':
                def hypotest_fun(x):
                    samples = xp.split(x, split_indices)[:n_samp+n_kwd_samp]
                    samples = _remove_nans(samples, paired, xp)
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples):
                        return xp.full(n_out, xp.nan)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))

            # Addresses nan_policy == "propagate"
            elif (contains_nan and nan_policy == 'propagate'
                  and override['nan_propagation']):
                def hypotest_fun(x):
                    if xp.isnan(x).any():
                        return xp.full(n_out, xp.nan)

                    samples = xp.split(x, split_indices)[:n_samp+n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples):
                        return xp.full(n_out, xp.nan)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))

            else:
                def hypotest_fun(x):
                    samples = xp.split(x, split_indices)[:n_samp+n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples):
                        return xp.full(n_out, xp.nan)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))    

            dims = [x.ndim - 1] + range(x.ndim - 1)
            x = xp.permute_dims(x, dims)
            # res = _xp_apply(hypotest_fun, arr=x, xp=xp)
            res = np.apply_along_axis(hypotest_fun, axis=0, arr=x)
            res = _add_reduced_axes(res, reduced_axes, keepdims, xp)
            return tuple_to_result(*res)

        _axis_parameter_doc, _axis_parameter = _get_axis_params(default_axis)
        doc = FunctionDoc(axis_nan_policy_wrapper)
        parameter_names = [param.name for param in doc['Parameters']]
        if 'axis' in parameter_names:
            doc['Parameters'][parameter_names.index('axis')] = (
                _axis_parameter_doc)
        else:
            doc['Parameters'].append(_axis_parameter_doc)
        if 'nan_policy' in parameter_names:
            doc['Parameters'][parameter_names.index('nan_policy')] = (
                _nan_policy_parameter_doc)
        else:
            doc['Parameters'].append(_nan_policy_parameter_doc)
        if 'keepdims' in parameter_names:
            doc['Parameters'][parameter_names.index('keepdims')] = (
                _keepdims_parameter_doc)
        else:
            doc['Parameters'].append(_keepdims_parameter_doc)
        doc['Notes'] += _standard_note_addition
        doc = str(doc).split("\n", 1)[1]  # remove signature
        axis_nan_policy_wrapper.__doc__ = str(doc)

        sig = inspect.signature(axis_nan_policy_wrapper)
        parameters = sig.parameters
        parameter_list = list(parameters.values())
        if 'axis' not in parameters:
            parameter_list.append(_axis_parameter)
        if 'nan_policy' not in parameters:
            parameter_list.append(_nan_policy_parameter)
        if 'keepdims' not in parameters:
            parameter_list.append(_keepdims_parameter)
        sig = sig.replace(parameters=parameter_list)
        axis_nan_policy_wrapper.__signature__ = sig

        return axis_nan_policy_wrapper
    return axis_nan_policy_decorator
