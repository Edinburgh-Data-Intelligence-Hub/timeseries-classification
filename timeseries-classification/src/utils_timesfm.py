import numpy as np
import timesfm
import torch

def linear_interpolation(arr):
    """Performs linear interpolation to fill NaN values in a 1D numpy array.
    Taken from https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_base.py
    Args:
        arr: The 1D numpy array containing NaN values.
    Returns:
        A new numpy array with NaN values filled using linear interpolation,
        or the original array if no NaNs are present.
        Returns None if the input is not a 1D array.
        Returns the original array if there are no NaN values.
    """
    nans = np.isnan(arr)
    if not np.any(nans):  # Check if there are any NaNs
        return arr
    
    def x(z):
        return z.nonzero()[0]
    
    nans_indices = x(nans)
    non_nans_indices = x(~nans)
    non_nans_values = arr[~nans]
    try:
        arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
    except ValueError:
        if non_nans_values:
            mu = np.nanmean(arr)
        else:
            mu = 0.0
        arr = np.where(np.isfinite(arr), arr, mu)
    return arr


def strip_leading_nans(arr):
    """Removes contiguous NaN values from the beginning of a NumPy array.
    Taken from https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_base.py
    Args:
    arr: The input NumPy array.
    Returns:
    A new NumPy array with leading NaN values removed.
    If the array is all NaNs or empty, returns an empty array.
    """
    isnan = np.isnan(arr)
    first_valid_index = np.argmax(~isnan)
    return arr[first_valid_index:]


def process_inputs(model, inputs):
    """
    Processes input time series data by removing leading NaNs, performing linear interpolation,
    and pad or truncating the series to match the model's context length.
    Args:
        model: The TimesFM model with a forecast configuration.
        inputs: A list of 1D numpy arrays representing time series data.
    Returns:
        A tuple containing:
            - A list of processed numpy arrays, each of length equal to model's max_context.
            - A list of boolean masks indicating padded values (True for padded, False for original data
    """
    values = []
    masks = []
    context = model.forecast_config.max_context
    
    for each_input in inputs:
        value = linear_interpolation(strip_leading_nans(np.array(each_input.cpu())))
        if (w := len(value)) >= context:
            value = value[-context:]
            mask = np.zeros_like(value, dtype=bool)
        else:
            mask = np.array([True] * (context - w) + [False] * w)
            value = np.pad(value, (context - w, 0), "constant", constant_values=0.0)
        values.append(value)
        masks.append(mask)
    
    return values, masks

def get_embeddings(horizon, model, inputs, layers_to_hook, dtype=torch.float32):
    """
    Obtains input and output embeddings from the TimesFM model given processed inputs and masks.
    Args:
        horizon: The forecast horizon.
        model: The TimesFM model with a forecast configuration.
        inputs: A list of processed numpy arrays.
        layers_to_hook: int or A list of layer indices from which to extract output embeddings.
        dtype: The data type for the input tensors (default: torch.float32).
    Returns:
        A tuple containing:
            - input_embeddings: The input embeddings from the model. 
            - output_embeddings: The output embeddings from the model.
    """
    
    values, masks = process_inputs(model, inputs)
    
    inputs = (
        torch.from_numpy(np.array(values)).to(model.model.device).to(dtype)
    )
    masks = torch.from_numpy(np.array(masks)).to(model.model.device).to(torch.bool)
    batch_size = inputs.shape[0]
    
    batch_size, context = inputs.shape[0], inputs.shape[1]
    num_decode_steps = (horizon - 1) // model.model.o
    num_input_patches = context //  model.model.p
    decode_cache_size = num_input_patches + num_decode_steps *  model.model.m
    
    # Prefill
    patched_inputs = torch.reshape(inputs, (batch_size, -1,  model.model.p))
    patched_masks = torch.reshape(masks, (batch_size, -1,  model.model.p))
    
    # running stats
    n = torch.zeros(batch_size, device=inputs.device)
    mu = torch.zeros(batch_size, device=inputs.device)
    sigma = torch.zeros(batch_size, device=inputs.device)
    patch_mu = []
    patch_sigma = []
    
    for i in range(num_input_patches):
        (n, mu, sigma), _ = timesfm.torch.util.update_running_stats(
            n, mu, sigma, patched_inputs[:, i], patched_masks[:, i]
        )
        patch_mu.append(mu)
        patch_sigma.append(sigma)
    
    context_mu = torch.stack(patch_mu, dim=1)
    context_sigma = torch.stack(patch_sigma, dim=1)
    
    decode_caches = [
        timesfm.torch.util.DecodeCache(
            next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
            key=torch.zeros(
            batch_size,
            decode_cache_size,
            model.model.h,
            model.model.hd,
            device=inputs.device,
            ),
            value=torch.zeros(
            batch_size,
            decode_cache_size,
            model.model.h,
            model.model.hd,
            device=inputs.device,
            ),
        )
        for _ in range(model.model.x)
    ]
    
    normed_inputs = timesfm.torch.util.revin(patched_inputs, context_mu, context_sigma, reverse=False)
    normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
    
    tokenizer_inputs = torch.cat([normed_inputs, patched_masks.to(normed_inputs.dtype)], dim=-1)
    input_embeddings = model.model.tokenizer(tokenizer_inputs)
    
    decode_caches = [None] * model.model.x
    
    output_embeddings = []
    output_embedding = input_embeddings
    
    # if only last layer is to be hooked
    if layers_to_hook == -1:
        layers_to_hook = list(range(model.model.x))[-1:]
    
    with torch.no_grad():
        for i, layer in enumerate(model.model.stacked_xf):
            if i in layers_to_hook:
                output_embedding, _ = layer(
                output_embedding, patched_masks[..., -1], decode_caches[i]
                )
                output_embeddings.append(input_embeddings)
    
    return input_embeddings, output_embeddings