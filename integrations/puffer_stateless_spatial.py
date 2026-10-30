"""Specialize the generated Puffer bridge for the sole stateless spatial actor."""

from pathlib import Path


def replace_once(text, old, new=""):
    if text.count(old) != 1:
        raise ValueError("Pinned stateless source anchor changed: " + old)
    return text.replace(old, new, 1)


def remove_block(text, anchor):
    if text.count(anchor) != 1:
        raise ValueError("Pinned stateless block changed: " + anchor)
    start = text.index(anchor)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[:start] + text[end:]


def install_stateless_spatial(source: Path):
    """No empty tensors or recurrent compatibility mode: remove external carry."""
    path = source / "src/metta_fabric.cuh"
    text = path.read_text()
    text = replace_once(text, "  int input, output, horizon, state_words;", "  int input, output, horizon;")
    text = replace_once(
        text,
        '  assert(hidden_size == state_words &&\n         "Native recurrent allocation must match Fabric state");',
        '  assert(hidden_size == 0 && state_words == 0 && "Spatial actor has no external carry");',
    )
    text = replace_once(
        text,
        "  return {input_size, output_size, horizon, state_words};",
        "  return {input_size, output_size, horizon};",
    )
    text = replace_once(text, "Prec observations, Prec state, Prec terminals,", "Prec observations, Prec terminals,")
    text = replace_once(text, "  PyObject *s = metta_cuda_tensor(state.data, {batch, arch->state_words}, device);\n")
    text = replace_once(text, '"forward_device", "OOOOiii", p, s, x,', '"forward_device", "OOOiii", p, x,')
    text = replace_once(
        text,
        "  if (!terminals.data)\n    metta_copy_device(state.data, PyTuple_GetItem(result, 1),\n                      batch * arch->state_words, stream);\n",
    )
    text = replace_once(text, "  acts.tape = PyTuple_GetItem(result, 2);", "  acts.tape = PyTuple_GetItem(result, 1);")
    text = replace_once(text, "  Py_DECREF(s);\n")
    text = replace_once(
        text, "Prec observations, Prec state, cudaStream_t stream) {", "Prec observations, cudaStream_t stream) {"
    )
    text = replace_once(
        text,
        "metta_forward(arch, weights, acts, observations, state,",
        "metta_forward(arch, weights, acts, observations,",
    )
    path.write_text(text)
    path = source / "src/algo.cu"
    text = path.read_text()
    text = replace_once(text, "    Prec mb_state;       // view into train_state (L, A, H); read with agent_off\n")
    text = replace_once(
        text, "        Prec state, Prec terminals, int agent_off,", "        Prec terminals, int agent_off,"
    )
    text = replace_once(
        text, "    Prec state_view = state;\n    state_view.data += (long)agent_off * p->state_words;\n"
    )
    text = replace_once(
        text,
        "metta_forward(p, w, activations, x, state_view, terminals,",
        "metta_forward(p, w, activations, x, terminals,",
    )
    path.write_text(text)
    path = source / "src/pufferl.cu"
    text = path.read_text()
    text = replace_once(
        text,
        "    // GPU envs have no per-env tags / frozen layout",
        '    assert(num_policies == 1 && "Stateless spatial supports one native policy; frozen opponents run in Python");\n'
        "    // GPU envs have no per-env tags / frozen layout",
    )
    for line in (
        "    Prec initial_states;\n",
        "    Prec* buffer_states;    // [num_buffers]\n",
        "    Prec train_state;  // (L, A, H) carry in env order; graph reads with dest_off\n",
    ):
        text = replace_once(text, line)
    start = text.index("// Index into (L, agents, H):")
    end = text.index("// Select time t, then agents", start)
    text = text[:start] + text[end:]
    text = replace_once(text, "        Prec* st = &pol->buffer_states[buf];\n")
    start = text.index("        // Per-policy state is compact")
    end = text.index("        Prec dec = arch_forward", start)
    text = text[:start] + text[end:]
    text = replace_once(
        text,
        "arch_forward(&pol->arch, *w, *acts, obs_b, *st, stream)",
        "arch_forward(&pol->arch, *w, *acts, obs_b, stream)",
    )
    text = remove_block(
        text,
        "    for (int b = 0; b < p->num_policies; b++) {\n        for (int i = 0; i < vec->buffers; i++) {\n            Prec* st =",
    )
    text = remove_block(text, "    if (p->hypers.reset_every_horizon) {")
    start = text.index("    if (hypers->reset_every_horizon || src.initial_states.data == NULL) {")
    end = text.index("    puf_stamp<<<", start)
    text = text[:start] + text[end:]
    text = replace_once(text, "        graph.mb_state = pufferl->train_state;\n")
    text = replace_once(
        text, "pufferl->train_activs, graph.mb_obs, graph.mb_state,", "pufferl->train_activs, graph.mb_obs,"
    )
    text = replace_once(text, "        pol->buffer_states = (Prec*)calloc(1, num_buffers * sizeof(Prec));\n")
    text = replace_once(
        text,
        "            pol->buffer_states[i] = {.shape = {L, slice, h}};\n            alloc_register(aalloc, &pol->buffer_states[i]);\n",
    )
    text = replace_once(
        text, "    // Carry path: per-slot initial RNN states. reset_every_horizon zeros train_state.\n"
    )
    text = remove_block(text, "    if (!hypers.reset_every_horizon) {")
    text = replace_once(
        text,
        "    pufferl->train_state = {.shape = {num_layers, total_agents, hidden_size}};\n    alloc_register(acts, &pufferl->train_state);\n",
    )
    if any(
        token in text
        for token in (
            "buffer_states",
            "train_state",
            "initial_states",
            "zero_term_state",
            "snapshot_state",
            "graph.mb_state",
        )
    ):
        raise ValueError("Native external carry survived specialization")
    path.write_text(text)
