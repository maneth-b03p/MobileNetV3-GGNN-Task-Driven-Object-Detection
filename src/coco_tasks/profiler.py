profiler_stats = {
    "flops":        0,      # total multiply-accumulate ops × 2
    "mem_bytes":    0,      # tensor memory touched (bytes)
    "mem_peak":     0,      # high-water mark
    "conv_log":     [],     # list of (B, C_in, C_out, H, W, K) tuples
    "linear_count": 0,      # number of linear layer calls
    "ggnn_steps":   0,      # GRU iterations across all forward passes
    "dataset_items":0,
}

def reset_profiler():
    for k in profiler_stats:
        profiler_stats[k] = [] if k == "conv_log" else 0

def log_conv(B, C_in, C_out, H_out, W_out, K):
    profiler_stats["conv_log"].append((B, C_in, C_out, H_out, W_out, K))
    # standard FLOP formula: 2 × B × C_out × H_out × W_out × C_in × K × K
    profiler_stats["flops"] += 2 * B * C_out * H_out * W_out * C_in * K * K

def log_linear(B, in_f, out_f):
    profiler_stats["flops"] += 2 * B * in_f * out_f
    profiler_stats["linear_count"] += 1

def log_mem(tensor):
    b = tensor.element_size() * tensor.nelement()
    profiler_stats["mem_bytes"] += b
    if b > profiler_stats["mem_peak"]:
        profiler_stats["mem_peak"] = b

def print_profiler_report():
    print(f"  Total FLOPs  : {profiler_stats['flops']:,}")
    print(f"  Mem touched  : {profiler_stats['mem_bytes'] / 1e6:.2f} MB")
    print(f"  Mem peak     : {profiler_stats['mem_peak'] / 1e6:.2f} MB")
    print(f"  Convolutions : {len(profiler_stats['conv_log'])}")
    print(f"  Linear calls : {profiler_stats['linear_count']}")
    print(f"  GGNN steps   : {profiler_stats['ggnn_steps']}")
    for i, (B,Ci,Co,H,W,K) in enumerate(profiler_stats["conv_log"]):
        print(f"    conv[{i:02d}] B={B} {Ci}→{Co} {H}×{W} K={K}")