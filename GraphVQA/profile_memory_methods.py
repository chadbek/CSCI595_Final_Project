import argparse
import csv
import gc
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

from gqa_dataset_entry import GQATorchDataset, GQATorchDataset_collate_fn
from pipeline_model_gat import PipelineModel


DEFAULT_CHECKPOINTS = {
    "static": "outputdir/gat_static_e100/checkpoint.pth",
    "reweight": "outputdir/gat_reweight_e100/checkpoint.pth",
    "augment": "outputdir/gat_augment_e100/checkpoint.pth",
    "prune": "outputdir/gat_prune_e100/checkpoint.pth",
}


def mb(num_bytes):
    return float(num_bytes) / (1024.0 * 1024.0)


class RssSampler:
    def __init__(self, interval_s=0.01):
        self.process = psutil.Process(os.getpid())
        self.interval_s = interval_s
        self.peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        return self.peak


@contextmanager
def rss_peak(interval_s=0.01):
    sampler = RssSampler(interval_s=interval_s).start()
    try:
        yield sampler
    finally:
        sampler.stop()


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_reset_peak(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_stats(device):
    if device.type != "cuda":
        return {}
    cuda_sync(device)
    return {
        "cuda_allocated_mb": mb(torch.cuda.memory_allocated(device)),
        "cuda_reserved_mb": mb(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_mb": mb(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_mb": mb(torch.cuda.max_memory_reserved(device)),
    }


def tensor_footprint_mb(module):
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        total += tensor.numel() * tensor.element_size()
    return mb(total)


def clone_batch(batch):
    question_id, questions, scene_graphs, programs, full_answers, labels, types = batch
    return (
        question_id,
        questions.clone(),
        scene_graphs.clone(),
        programs.clone(),
        full_answers.clone(),
        labels.clone(),
        types,
    )


def move_batch_to_device(batch, device):
    question_id, questions, scene_graphs, programs, full_answers, labels, types = clone_batch(batch)
    return (
        question_id,
        questions.to(device=device, non_blocking=True),
        scene_graphs.to(device=device, non_blocking=True),
        programs.to(device=device, non_blocking=True),
        full_answers.to(device=device, non_blocking=True),
        labels.to(device=device, non_blocking=True),
        types,
    )


def prepare_batches(split, batch_size, batches, workers, graph_method):
    dataset = GQATorchDataset(
        split,
        build_vocab_flag=False,
        load_vocab_flag=True,
        graph_method=graph_method,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=GQATorchDataset_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    cached = []
    for idx, batch in enumerate(loader):
        cached.append(clone_batch(batch))
        if idx + 1 >= batches:
            break
    return dataset, cached


def load_model(method, checkpoint_path, device):
    model = PipelineModel(graph_method=method)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint.get("epoch")
    del checkpoint
    model.to(device)
    return model, epoch


def graph_stats(model, batch, device):
    model.eval()
    _, questions, scene_graphs, programs, full_answers, labels, _ = move_batch_to_device(batch, device)
    del programs, full_answers, labels

    cuda_reset_peak(device)
    cuda_sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        questions_encoded = model.question_encoder(questions)
        q_summary = questions_encoded[0]
        _, _, _, out_graphs = model.scene_graph_encoder(scene_graphs, q=q_summary)
    cuda_sync(device)
    elapsed = time.perf_counter() - start

    return {
        "input_nodes": int(scene_graphs.x.size(0)),
        "input_edges": int(scene_graphs.edge_index.size(1)),
        "output_edges": int(out_graphs.edge_index.size(1)),
        "graph_builder_time_s": elapsed,
        **cuda_stats(device),
    }


def eval_step(model, batch, device):
    _, questions, scene_graphs, programs, full_answers, labels, _ = move_batch_to_device(batch, device)
    del labels
    with torch.no_grad():
        model(
            questions,
            scene_graphs,
            programs[:-1],
            full_answers[:-1],
        )


def text_generation_loss(loss_fn, output, target):
    vocab_size = len(GQATorchDataset.TEXT.vocab)
    return loss_fn(output.contiguous().view(-1, vocab_size), target.contiguous().view(-1))


def train_step(model, optimizer, criterion, batch, device):
    _, questions, scene_graphs, programs, full_answers, labels, _ = move_batch_to_device(batch, device)
    programs_input = programs[:-1]
    programs_target = programs[1:]
    full_answers_input = full_answers[:-1]

    programs_output, short_answer_logits = model(
        questions,
        scene_graphs,
        programs_input,
        full_answers_input,
    )
    program_loss = text_generation_loss(criterion["program"], programs_output, programs_target)
    short_answer_loss = criterion["short_answer"](short_answer_logits, labels)
    loss = program_loss + short_answer_loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def run_mode(mode, model, batches, device, warmup_batches):
    if mode == "eval":
        model.eval()
        step = lambda batch: eval_step(model, batch, device)
        optimizer = None
        criterion = None
    elif mode == "train":
        model.train()
        text_pad_idx = GQATorchDataset.TEXT.vocab.stoi[GQATorchDataset.TEXT.pad_token]
        criterion = {
            "program": torch.nn.CrossEntropyLoss(ignore_index=text_pad_idx).to(device),
            "short_answer": torch.nn.CrossEntropyLoss().to(device),
        }
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)
        step = lambda batch: train_step(model, optimizer, criterion, batch, device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for batch in batches[:warmup_batches]:
        step(batch)
    cuda_sync(device)

    cuda_reset_peak(device)
    start = time.perf_counter()
    with rss_peak() as sampler:
        for batch in batches:
            step(batch)
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    rss_peak_mb = mb(sampler.peak)

    result = {
        "mode": mode,
        "batches": len(batches),
        "time_total_s": elapsed,
        "time_per_batch_s": elapsed / max(len(batches), 1),
        "rss_peak_mb": rss_peak_mb,
        "rss_after_mb": mb(psutil.Process(os.getpid()).memory_info().rss),
        **cuda_stats(device),
    }

    del optimizer, criterion
    return result


def parse_checkpoint_overrides(values):
    checkpoints = dict(DEFAULT_CHECKPOINTS)
    for value in values:
        method, sep, path = value.partition("=")
        if not sep:
            raise ValueError("--checkpoint expects METHOD=PATH")
        checkpoints[method] = path
    return checkpoints


def write_results(results, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "memory_profile_results.json"
    csv_path = output_dir / "memory_profile_results.csv"

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rows = []
    for item in results["methods"]:
        base = {
            "method": item["method"],
            "checkpoint_epoch": item["checkpoint_epoch"],
            "checkpoint_file_mb": item["checkpoint_file_mb"],
            "model_tensors_mb": item["model_tensors_mb"],
            "rss_after_model_load_mb": item["rss_after_model_load_mb"],
            "graph_input_edges": item["graph"]["input_edges"],
            "graph_output_edges": item["graph"]["output_edges"],
            "graph_builder_time_s": item["graph"]["graph_builder_time_s"],
        }
        for mode_result in item["modes"]:
            row = dict(base)
            row.update(mode_result)
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description="Profile GraphVQA method memory usage.")
    parser.add_argument("--methods", nargs="+", default=["static", "reweight", "augment", "prune"])
    parser.add_argument("--checkpoint", action="append", default=[], help="Override checkpoint as METHOD=PATH")
    parser.add_argument("--split", default="val_unbiased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--modes", nargs="+", default=["eval", "train"], choices=["eval", "train"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="outputdir/memory_profile")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    process = psutil.Process(os.getpid())
    checkpoints = parse_checkpoint_overrides(args.checkpoint)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")
    print(f"split={args.split} batch_size={args.batch_size} batches={args.batches}")

    _, batches = prepare_batches(
        split=args.split,
        batch_size=args.batch_size,
        batches=max(args.batches, args.warmup_batches),
        workers=args.workers,
        graph_method="static",
    )
    if not batches:
        raise RuntimeError("No batches were loaded.")

    results = {
        "config": vars(args),
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dataset_rss_mb": mb(process.memory_info().rss),
        "methods": [],
    }

    for method in args.methods:
        if method not in checkpoints:
            raise ValueError(f"No checkpoint path configured for method {method}")
        checkpoint_path = Path(checkpoints[method])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        print(f"\n== {method} ==")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            cuda_reset_peak(device)

        model, epoch = load_model(method, str(checkpoint_path), device)
        cuda_sync(device)

        method_result = {
            "method": method,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": epoch,
            "checkpoint_file_mb": mb(checkpoint_path.stat().st_size),
            "model_tensors_mb": tensor_footprint_mb(model),
            "rss_after_model_load_mb": mb(process.memory_info().rss),
            "cuda_after_model_load": cuda_stats(device),
        }
        print(
            f"loaded epoch={epoch} model_tensors={method_result['model_tensors_mb']:.1f} MB "
            f"rss={method_result['rss_after_model_load_mb']:.1f} MB"
        )

        method_result["graph"] = graph_stats(model, batches[0], device)
        print(
            "graph edges "
            f"{method_result['graph']['input_edges']} -> {method_result['graph']['output_edges']}"
        )

        mode_results = []
        for mode in args.modes:
            mode_result = run_mode(
                mode=mode,
                model=model,
                batches=batches[: args.batches],
                device=device,
                warmup_batches=args.warmup_batches,
            )
            mode_results.append(mode_result)
            cuda_peak = mode_result.get("cuda_peak_allocated_mb")
            cuda_text = f" cuda_peak_alloc={cuda_peak:.1f} MB" if cuda_peak is not None else ""
            print(
                f"{mode}: rss_peak={mode_result['rss_peak_mb']:.1f} MB"
                f"{cuda_text} time/batch={mode_result['time_per_batch_s']:.3f}s"
            )

        method_result["modes"] = mode_results
        results["methods"].append(method_result)

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path, csv_path = write_results(results, Path(args.output_dir))
    print(f"\nwrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
