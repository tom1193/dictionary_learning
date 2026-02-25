import random
from circuitsvis.activations import text_neuron_activations
from einops import rearrange
import torch as t
from collections import namedtuple
import umap
import pandas as pd
import plotly.express as px


def feature_effect(
    model,
    submodule,
    dictionary,
    feature,
    inputs,
    max_length=128,
    add_residual=True,  # whether to compensate for dictionary reconstruction error by adding residual
    k=10,
    largest=True,
):
    """
    Effect of ablating the feature on top k predictions for next token.
    """
    tracer_kwargs = {
        "scan": False,
        "validate": False,
        "invoker_args": dict(max_length=max_length),
    }
    # clean run
    with t.no_grad(), model.trace(inputs, **tracer_kwargs):
        if dictionary is None:
            pass
        elif not add_residual:  # run hidden state through autoencoder
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = dictionary(submodule.output[0])
            else:
                submodule.output = dictionary(submodule.output)
        clean_output = model.output.save()
    try:
        clean_logits = clean_output.value.logits[:, -1, :]
    except:
        clean_logits = clean_output.value[:, -1, :]
    clean_logprobs = t.nn.functional.log_softmax(clean_logits, dim=-1)

    # ablated run
    with t.no_grad(), model.trace(inputs, **tracer_kwargs):
        if dictionary is None:
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:, -1, feature] = 0
            else:
                submodule.output[:, -1, feature] = 0
        else:
            x = submodule.output
            if type(x.shape) == tuple:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            residual = x - x_hat

            f[:, -1, feature] = 0
            if add_residual:
                x_hat = dictionary.decode(f) + residual
            else:
                x_hat = dictionary.decode(f)

            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output = x_hat
        ablated_output = model.output.save()
    try:
        ablated_logits = ablated_output.value.logits[:, -1, :]
    except:
        ablated_logits = ablated_output.value[:, -1, :]
    ablated_logprobs = t.nn.functional.log_softmax(ablated_logits, dim=-1)

    diff = clean_logprobs - ablated_logprobs
    top_probs, top_tokens = t.topk(diff.mean(dim=0), k=k, largest=largest)
    return top_tokens, top_probs

def _list_decode(x, model):
    if isinstance(x, int):
        return model.tokenizer.decode(x)
    else:
        return [_list_decode(y, model) for y in x]

featureProfile = namedtuple("featureProfile", ["top_contexts", "top_tokens", "top_affected"])

def examine_dimension(
    model, submodule, buffer, dictionary=None, max_length=128, n_inputs=512, dim_idx=None, k=30
):

    tracer_kwargs = {
        "scan": False,
        "validate": False,
        "invoker_args": dict(max_length=max_length),
    }

    if dim_idx is None:
        dim_idx = random.randint(0, activations.shape[-1] - 1)

    inputs = buffer.tokenized_batch(batch_size=n_inputs)

    with t.no_grad(), model.trace(inputs, **tracer_kwargs):
        tokens = model.inputs[1][
            "input_ids"
        ].save()  # if you're getting errors, check here; might only work for pythia models
        activations = submodule.output
        if type(activations.shape) == tuple:
            activations = activations[0]
        if dictionary is not None:
            activations = dictionary.encode(activations)
        activations = activations[:, :, dim_idx].save()
    activations = activations.value

    # get top k tokens by mean activation
    tokens = tokens.value
    token_mean_acts = {}
    for ctx in tokens:
        for tok in ctx:
            if tok.item() in token_mean_acts:
                continue
            idxs = (tokens == tok).nonzero(as_tuple=True)
            token_mean_acts[tok.item()] = activations[idxs].mean().item()
    top_tokens = sorted(token_mean_acts.items(), key=lambda x: x[1], reverse=True)[:k]
    top_tokens = [(model.tokenizer.decode(tok), act) for tok, act in top_tokens]

    flattened_acts = rearrange(activations, "b n -> (b n)")
    topk_indices = t.argsort(flattened_acts, dim=0, descending=True)[:k]
    batch_indices = topk_indices // activations.shape[1]
    token_indices = topk_indices % activations.shape[1]
    tokens = [
        tokens[batch_idx, : token_idx + 1].tolist()
        for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    activations = [
        activations[batch_idx, : token_id + 1, None, None]
        for batch_idx, token_id in zip(batch_indices, token_indices)
    ]
    decoded_tokens = _list_decode(tokens, model)
    top_contexts = text_neuron_activations(decoded_tokens, activations)

    top_affected = feature_effect(
        model, submodule, dictionary, dim_idx, tokens, max_length=max_length, k=k
    )
    top_affected = [(model.tokenizer.decode(tok), prob.item()) for tok, prob in zip(*top_affected)]

    # return namedtuple("featureProfile", ["top_contexts", "top_tokens", "top_affected"])(
    #     top_contexts, top_tokens, top_affected
    # )
    return featureProfile(top_contexts, top_tokens, top_affected)

def examine_multiple_dimensions(
    model,
    submodule,
    buffer,
    dictionary,
    max_length=128,
    n_inputs=512,
    dim_idxs=10, # if int, number of random dimensions to examine; if list, specific dimension indices to examine
    k=30,

):
    """
    Vectorized version of `examine_dimension` over many dimensions.
    Returns dict[int, featureProfile]: mapping dim_idx -> profile
    """
    tracer_kwargs = {
        "scan": False,
        "validate": False,
        "invoker_args": dict(max_length=max_length),
    }

    # sample inputs once
    inputs = buffer.tokenized_batch(batch_size=n_inputs)

    # trace once, get tokens + activations
    with t.no_grad(), model.trace(inputs, **tracer_kwargs):
        tokens = model.inputs[1]["input_ids"].save()

        activations = submodule.output
        if type(activations.shape) == tuple:
            activations = activations[0]
        activations = activations.save()

    tokens = tokens.value
    activations = activations.value

    # Convert to feature space
    if dictionary is not None:
        activations = dictionary.encode(activations)  # (B, T, n_features)

    B, T, D = activations.shape
    
    if isinstance(dim_idxs, int):
        # randomly select dim_idxs as a list of integers
        dim_idx_list = random.sample(range(D), k=dim_idxs)
    else:
        # Check dim_idxs in range
        out_of_range = [d for d in dim_idxs if d < 0 or d >= D]
        if out_of_range:
            raise ValueError(f"Some dim_idxs out of range [0, {D-1}]: {out_of_range}")
        dim_idx_list = list(dim_idxs)

    profiles = {}

    for dim_idx in dim_idx_list:
        a = activations[:, :, dim_idx]  # (B, T)

        # top tokens by mean activation over occurrences
        unique_toks = tokens.unique()
        token_means = []
        for tok in unique_toks:
            idx = t.where(tokens == tok)
            token_means.append(a[idx].mean())
        token_means = t.stack(token_means)

        top_vals, top_inds = t.topk(token_means, k=min(k, unique_toks.numel()))
        top_tok_ids = unique_toks[top_inds]
        top_tokens = [
            (model.tokenizer.decode([tid.item()]), top_vals[i].item())
            for i, tid in enumerate(top_tok_ids)
        ]

        # top contexts by highest activation positions
        flat = a.flatten()
        k_ctx = min(k, flat.numel())
        _, top_pos = t.topk(flat, k=k_ctx)

        top_contexts = []
        for pos in top_pos:
            b = (pos // T).item()
            j = (pos % T).item()
            top_contexts.append(tokens[b, : j + 1].tolist())

        top_contexts_str = _list_decode(top_contexts, model)
        top_contexts_vis = text_neuron_activations(top_contexts_str, a)

        # top affected next tokens via feature ablation
        top_affected = feature_effect(
            model, submodule, dictionary, dim_idx, top_contexts, k=k,
        )
        top_affected = [(model.tokenizer.decode(tok), prob.item()) for tok, prob in zip(*top_affected)]
        # top_affected = (
        #     model.tokenizer.batch_decode(top_affected[0]),
        #     top_affected[1].tolist(),
        # )

        profiles[dim_idx] = featureProfile(top_contexts, top_tokens, top_affected)

    return profiles

def feature_umap(
    dictionary,
    weight="decoder",  # 'encoder' or 'decoder'
    # UMAP parameters
    n_neighbors=15,
    metric="cosine",
    min_dist=0.05,
    n_components=2,  # dimension of the UMAP embedding
    feat_idxs=None,  # if not none, indicate the feature with a red dot
):
    """
    Fit a UMAP embedding of the dictionary features and return a plotly plot of the result."""
    if weight == "encoder":
        df = pd.DataFrame(dictionary.encoder.weight.cpu().detach().numpy())
    else:
        df = pd.DataFrame(dictionary.decoder.weight.T.cpu().detach().numpy())
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        n_components=n_components,
    )
    embedding = reducer.fit_transform(df)
    if feat_idxs is None:
        colors = None
    if isinstance(feat_idxs, int):
        feat_idxs = [feat_idxs]
    else:
        colors = ["blue" if i not in feat_idxs else "red" for i in range(embedding.shape[0])]
    if n_components == 2:
        return px.scatter(x=embedding[:, 0], y=embedding[:, 1], hover_name=df.index, color=colors)
    if n_components == 3:
        return px.scatter_3d(
            x=embedding[:, 0],
            y=embedding[:, 1],
            z=embedding[:, 2],
            hover_name=df.index,
            color=colors,
        )
    raise ValueError("n_components must be 2 or 3")