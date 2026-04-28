import json
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    Dinov2Model,
    Dinov2PreTrainedModel,
    Dinov2ForImageClassification,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
    YolosForObjectDetection,
    set_seed as hf_set_seed,
)
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers.trainer_utils import get_last_checkpoint

import evaluate
import huggingface_hub
import wandb

from fgvc_datasets_setup import loader as fgvc_loader
from fgvc_datasets_setup.loader import _DATASET_NUM_LABELS
from get_rank.dinov2 import (
    get_dinov2_rank_binary_search_energy_ratio,
    get_dinov2_rank_ratio,
)
from get_rank.vit import (
    get_vit_rank_binary_search_energy_ratio,
    get_vit_rank_ratio,
)
from get_rank.yolos import (
    get_yolos_rank_binary_search_energy_ratio,
    get_yolos_rank_ratio,
)
from meft import MeftConfig, MeftTrainer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("login to huggingface_hub")
try:
    huggingface_hub.login(token="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")
    print("login success")
except Exception as exc:
    print(f"login skipped: {exc}")


VOC_DET_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
VOC_DET_LABEL2ID = {name: idx for idx, name in enumerate(VOC_DET_CLASSES)}
VOC_DET_ID2LABEL = {idx: name for name, idx in VOC_DET_LABEL2ID.items()}

CLASSIFICATION_HF_DATASETS = {
    "cifar100": ("img", "fine_label"),
    "food101": ("image", "label"),
    "tiny-imagenet": ("image", "label"),
}


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "semantic_segmentation", "object_detection"],
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit-base",
        choices=[
            "vit-base",
            "vit-large",
            "vit-huge",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
            "dinov2-base-seg",
            "yolos-tiny",
        ],
    )
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--val_set_size", type=int, default=10000)
    parser.add_argument("--image_size", type=int, default=None)

    parser.add_argument("--vanilla_train", action="store_true")
    parser.add_argument("--rank_ratio", type=float, default=0.125)
    parser.add_argument("--dynamic_rank", action="store_true")
    parser.add_argument("--energy_ratio", type=float, default=0.5)
    parser.add_argument("--energy_search", action="store_true")
    parser.add_argument("--patch_locations", type=int, default=2)
    parser.add_argument(
        "--compress_method",
        type=str,
        default="dynamic_fixed_rank_dynamic_quantization",
        choices=[
            "dynamic_fixed_rank_dynamic_quantization",
            "rqd",
            "rqb",
            "energy_rqb",
            "probing_rqb",
            "tsvd",
            "rsvd",
            "nyssvd",
        ],
    )
    parser.add_argument(
        "--quant_method",
        type=str,
        default="1bit_pergroupchannel",
        choices=["1bit_pertensor", "1bit_pergroupchannel", "ternary", "two_bit_group"],
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--wandb_project_name", type=str, default="wandb")
    parser.add_argument("--wandb_run_name", type=str, default="wandb")

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_score_threshold", type=float, default=0.0)
    return parser.parse_args()


def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def append_local_result(output_dir: str, args, test_results: dict, train_total_time: float, peak_mem_gb: float | None):
    os.makedirs(output_dir, exist_ok=True)
    record = {
        "task_type": args.task_type,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "compress_method": args.compress_method,
        "quant_method": args.quant_method,
        "rank_ratio": args.rank_ratio,
        "dynamic_rank": args.dynamic_rank,
        "energy_ratio": args.energy_ratio,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "wandb_run_name": args.wandb_run_name,
        "output_dir": args.output_dir,
        "train_total_time_s": train_total_time,
        "train_peak_gpu_mem_gb": peak_mem_gb,
        "eval_accuracy": test_results.get("eval_accuracy"),
        "eval_miou": test_results.get("eval_miou"),
        "eval_pixel_accuracy": test_results.get("eval_pixel_accuracy"),
        "eval_map50": test_results.get("eval_map50"),
        "eval_ap50_per_class": test_results.get("eval_ap50_per_class"),
        "raw_test_results": test_results,
    }
    result_file = os.path.join(output_dir, "results_local.jsonl")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")


class HFDatasetWrapper(Dataset):
    def __init__(self, dataset, image_key: str, label_key: str, processor):
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example[self.image_key].convert("RGB")
        encoding = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": int(example[self.label_key]),
        }


class VOCSegmentationDataset(Dataset):
    def __init__(self, root: str, split: str, processor, image_size: int):
        image_set = "train" if split == "train" else "val"
        self.dataset = torchvision.datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=False,
        )
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        image = image.convert("RGB")
        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )
        mask = TVF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        labels = torch.as_tensor(np.array(mask), dtype=torch.long)
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": labels,
        }


class VOCDetectionDataset(Dataset):
    def __init__(self, root: str, split: str, processor, image_size: int):
        image_set = "trainval" if split == "train" else "test"
        self.dataset = torchvision.datasets.VOCDetection(
            root=root,
            year="2007",
            image_set=image_set,
            download=False,
        )
        self.processor = processor
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        image = image.convert("RGB")
        image_id = int(target["annotation"]["filename"].split(".")[0])
        width = float(target["annotation"]["size"]["width"])
        height = float(target["annotation"]["size"]["height"])
        objects = target["annotation"].get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        class_labels = []
        boxes = []
        areas = []
        for obj in objects:
            name = obj["name"]
            if name not in VOC_DET_LABEL2ID:
                continue
            bbox = obj["bndbox"]
            xmin = max(0.0, min(width, float(bbox["xmin"])))
            ymin = max(0.0, min(height, float(bbox["ymin"])))
            xmax = max(0.0, min(width, float(bbox["xmax"])))
            ymax = max(0.0, min(height, float(bbox["ymax"])))
            bw = max(0.0, xmax - xmin)
            bh = max(0.0, ymax - ymin)
            if bw <= 0 or bh <= 0:
                continue
            cx = (xmin + xmax) / 2.0 / width
            cy = (ymin + ymax) / 2.0 / height
            boxes.append([cx, cy, bw / width, bh / height])
            class_labels.append(VOC_DET_LABEL2ID[name])
            areas.append(bw * bh)

        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": self.image_size, "width": self.image_size},
        )
        labels = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(class_labels), dtype=torch.long),
            "orig_size": torch.tensor([int(height), int(width)], dtype=torch.long),
            "size": torch.tensor([int(height), int(width)], dtype=torch.long),
            "image_id": torch.tensor([image_id], dtype=torch.long),
        }
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": labels,
        }


class LiteSegDecodeHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Conv2d(hidden_size, 256, kernel_size=1)
        self.act = nn.GELU()
        self.refine = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(256, num_labels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.act(x)
        x = self.refine(x)
        x = self.act(x)
        return self.classifier(x)


class Dinov2ForSemanticSegmentationLite(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.dinov2 = Dinov2Model(config)
        self.decode_head = LiteSegDecodeHead(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        pixel_values=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.dinov2(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.last_hidden_state
        patch_tokens = hidden_states[:, 1:, :]
        batch_size, num_tokens, hidden_dim = patch_tokens.shape
        patch_size = getattr(self.config, "patch_size", 14)
        grid_h = pixel_values.shape[-2] // patch_size
        grid_w = pixel_values.shape[-1] // patch_size
        if grid_h * grid_w != num_tokens:
            side = int(math.sqrt(num_tokens))
            grid_h, grid_w = side, side
        x = patch_tokens.transpose(1, 2).reshape(batch_size, hidden_dim, grid_h, grid_w)
        logits = self.decode_head(x)
        target_size = labels.shape[-2:] if labels is not None else pixel_values.shape[-2:]
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, ignore_index=255)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class DetectionPrediction:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class VanillaVisionTrainer(Trainer):
    def __init__(self, *args, task_type: str, image_processor=None, **kwargs):
        self.task_type = task_type
        self.image_processor = image_processor
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.task_type != "object_detection":
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        metric = DetectionMAP50Metric(num_classes=len(VOC_DET_CLASSES))
        losses = []
        for inputs in dataloader:
            prepared = self._prepare_inputs(inputs)
            labels = prepared["labels"]
            with torch.no_grad():
                outputs = model(**prepared)
            if outputs.loss is not None:
                losses.append(outputs.loss.detach().float().cpu().item())

            target_sizes = torch.stack([label["orig_size"] for label in labels]).to(outputs.logits.device)
            predictions = self.image_processor.post_process_object_detection(
                DetectionPrediction(logits=outputs.logits, pred_boxes=outputs.pred_boxes),
                threshold=self.args.eval_score_threshold if hasattr(self.args, "eval_score_threshold") else 0.0,
                target_sizes=target_sizes,
            )
            metric.update(predictions, labels)

        detection_metrics = metric.compute()
        metrics = {
            f"{metric_key_prefix}_loss": float(np.mean(losses)) if losses else 0.0,
            f"{metric_key_prefix}_map50": detection_metrics["map50"],
            f"{metric_key_prefix}_ap50_per_class": detection_metrics["ap50_per_class"],
        }
        self.log(metrics)
        return metrics


class SegmentationMetric:
    def __init__(self, num_labels: int, ignore_index: int = 255):
        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_labels, num_labels), dtype=np.int64)

    def update(self, logits_or_preds: np.ndarray, labels: np.ndarray):
        if logits_or_preds.ndim == 4:
            preds = np.argmax(logits_or_preds, axis=1)
        else:
            preds = logits_or_preds
        for pred, label in zip(preds, labels):
            mask = label != self.ignore_index
            gt = label[mask].reshape(-1)
            pd = pred[mask].reshape(-1)
            if gt.size == 0:
                continue
            hist = np.bincount(
                self.num_labels * gt + pd,
                minlength=self.num_labels * self.num_labels,
            ).reshape(self.num_labels, self.num_labels)
            self.confusion += hist

    def compute(self):
        intersection = np.diag(self.confusion)
        gt_sum = self.confusion.sum(axis=1)
        pred_sum = self.confusion.sum(axis=0)
        union = gt_sum + pred_sum - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float64),
            where=union > 0,
        )
        pixel_accuracy = float(intersection.sum() / max(1, self.confusion.sum()))
        valid = union > 0
        miou = float(iou[valid].mean()) if valid.any() else 0.0
        return {
            "miou": miou,
            "pixel_accuracy": pixel_accuracy,
        }


class DetectionMAP50Metric:
    def __init__(self, num_classes: int, iou_threshold: float = 0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.predictions = {i: [] for i in range(num_classes)}
        self.ground_truths = {i: {} for i in range(num_classes)}

    @staticmethod
    def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if boxes.size == 0:
            return np.zeros((0,), dtype=np.float32)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    @staticmethod
    def _to_xyxy(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
        h, w = image_size.tolist()
        scale = torch.tensor([w, h, w, h], dtype=boxes.dtype)
        boxes = boxes * scale
        cx, cy, bw, bh = boxes.unbind(-1)
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)

    def update(self, predictions, targets):
        for image_index, (pred, target) in enumerate(zip(predictions, targets)):
            gt_boxes = self._to_xyxy(target["boxes"].cpu(), target["orig_size"].cpu()).numpy()
            gt_labels = target["class_labels"].cpu().numpy()
            pred_boxes = pred["boxes"].detach().cpu().numpy()
            pred_labels = pred["labels"].detach().cpu().numpy()
            pred_scores = pred["scores"].detach().cpu().numpy()

            for class_id in range(self.num_classes):
                class_gt_mask = gt_labels == class_id
                class_pred_mask = pred_labels == class_id
                image_id = int(target["image_id"].item()) if "image_id" in target else image_index
                self.ground_truths[class_id][image_id] = gt_boxes[class_gt_mask]
                for box, score in zip(pred_boxes[class_pred_mask], pred_scores[class_pred_mask]):
                    self.predictions[class_id].append((image_id, float(score), box))

    def compute(self):
        ap_per_class = {}
        for class_id in range(self.num_classes):
            preds = sorted(self.predictions[class_id], key=lambda x: x[1], reverse=True)
            gt_map = self.ground_truths[class_id]
            total_gt = sum(len(v) for v in gt_map.values())
            if total_gt == 0:
                ap_per_class[VOC_DET_ID2LABEL[class_id]] = 0.0
                continue

            matched = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in gt_map.items()}
            tps = np.zeros(len(preds), dtype=np.float32)
            fps = np.zeros(len(preds), dtype=np.float32)
            for idx, (image_id, _, box) in enumerate(preds):
                gt_boxes = gt_map.get(image_id, np.zeros((0, 4), dtype=np.float32))
                if gt_boxes.size == 0:
                    fps[idx] = 1.0
                    continue
                ious = self._compute_iou(box, gt_boxes)
                best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
                best_iou = ious[best_idx] if ious.size > 0 else 0.0
                if best_iou >= self.iou_threshold and not matched[image_id][best_idx]:
                    tps[idx] = 1.0
                    matched[image_id][best_idx] = True
                else:
                    fps[idx] = 1.0

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)
            recalls = tp_cum / max(total_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-8)

            ap = 0.0
            for thresh in np.arange(0.0, 1.1, 0.1):
                valid = precisions[recalls >= thresh]
                ap += np.max(valid) if valid.size > 0 else 0.0
            ap_per_class[VOC_DET_ID2LABEL[class_id]] = float(ap / 11.0)

        map50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
        return {
            "map50": map50,
            "ap50_per_class": ap_per_class,
        }


class VisionTrainer(MeftTrainer[Trainer]):
    def __init__(self, *args, task_type: str, image_processor=None, **kwargs):
        self.task_type = task_type
        self.image_processor = image_processor
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        if self.task_type != "object_detection":
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataloader = self.get_eval_dataloader(eval_dataset)
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        metric = DetectionMAP50Metric(num_classes=len(VOC_DET_CLASSES))
        losses = []
        for inputs in dataloader:
            prepared = self._prepare_inputs(inputs)
            labels = prepared["labels"]
            with torch.no_grad():
                outputs = model(**prepared)
            if outputs.loss is not None:
                losses.append(outputs.loss.detach().float().cpu().item())

            target_sizes = torch.stack([label["orig_size"] for label in labels]).to(outputs.logits.device)
            predictions = self.image_processor.post_process_object_detection(
                DetectionPrediction(logits=outputs.logits, pred_boxes=outputs.pred_boxes),
                threshold=self.args.eval_score_threshold if hasattr(self.args, "eval_score_threshold") else 0.0,
                target_sizes=target_sizes,
            )
            metric.update(predictions, labels)

        detection_metrics = metric.compute()
        metrics = {
            f"{metric_key_prefix}_loss": float(np.mean(losses)) if losses else 0.0,
            f"{metric_key_prefix}_map50": detection_metrics["map50"],
            f"{metric_key_prefix}_ap50_per_class": detection_metrics["ap50_per_class"],
        }
        self.log(metrics)
        return metrics


def build_classification_dataset(args, processor):
    if args.dataset_name in CLASSIFICATION_HF_DATASETS:
        dataset_name = args.dataset_name
        hf_name = "zh-plus/tiny-imagenet" if dataset_name == "tiny-imagenet" else dataset_name
        dataset = load_dataset(hf_name)
        image_key, label_key = CLASSIFICATION_HF_DATASETS[dataset_name]
        if dataset_name == "food101":
            test_data = dataset["validation"]
        elif dataset_name == "tiny-imagenet":
            test_data = dataset["valid"]
        else:
            test_data = dataset["test"]
        train_val = dataset["train"].train_test_split(test_size=args.val_set_size, shuffle=True, seed=args.seed)
        return (
            HFDatasetWrapper(train_val["train"], image_key, label_key, processor),
            HFDatasetWrapper(train_val["test"], image_key, label_key, processor),
            HFDatasetWrapper(test_data, image_key, label_key, processor),
        )

    if args.dataset_name in _DATASET_NUM_LABELS:
        train_dataset = fgvc_loader.construct_train_dataset(args, processor)
        val_dataset = fgvc_loader.construct_val_dataset(args, processor)
        test_dataset = fgvc_loader.construct_test_dataset(args, processor)
        return train_dataset, val_dataset, test_dataset

    raise ValueError(f"Unsupported classification dataset: {args.dataset_name}")


def build_datasets(args, processor):
    if args.task_type == "classification":
        return build_classification_dataset(args, processor)
    if args.task_type == "semantic_segmentation":
        root = os.path.join(args.data_dir, "voc")
        return (
            VOCSegmentationDataset(root, "train", processor, args.image_size),
            VOCSegmentationDataset(root, "val", processor, args.image_size),
            VOCSegmentationDataset(root, "val", processor, args.image_size),
        )
    if args.task_type == "object_detection":
        root = os.path.join(args.data_dir, "voc")
        return (
            VOCDetectionDataset(root, "train", processor, args.image_size),
            VOCDetectionDataset(root, "val", processor, args.image_size),
            VOCDetectionDataset(root, "val", processor, args.image_size),
        )
    raise ValueError(f"Unsupported task type: {args.task_type}")


def build_compute_metrics(args):
    if args.task_type == "classification":
        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred: EvalPrediction):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        return compute_metrics

    if args.task_type == "semantic_segmentation":
        metric = SegmentationMetric(num_labels=21)

        def compute_metrics(eval_pred: EvalPrediction):
            logits, labels = eval_pred
            metric.confusion.fill(0)
            metric.update(logits, labels)
            values = metric.compute()
            return {
                "miou": values["miou"],
                "pixel_accuracy": values["pixel_accuracy"],
            }

        return compute_metrics

    return None


def detection_collator(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = [example["labels"] for example in examples]
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def segmentation_collator(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }


def build_lora_config(args):
    modules_to_save = {
        "classification": ["classifier"],
        "semantic_segmentation": ["decode_head"],
        "object_detection": ["class_labels_classifier", "bbox_predictor"],
    }[args.task_type]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=modules_to_save,
    )


def normalize_compress_method(compress_method: str) -> str:
    if compress_method == "rqd":
        return "dynamic_fixed_rank_dynamic_quantization"
    return compress_method


def get_patch_locations(args):
    if args.task_type == "object_detection":
        return ("norm", "ckpt_attn", "ckpt_mlp")
    if args.patch_locations == 1:
        return ("ckpt_layer",)
    if args.patch_locations == 2:
        return ("norm", "ckpt_attn", "ckpt_mlp")
    if args.patch_locations == 3:
        return ("norm", "attn_in", "attn_out", "mlp_in", "mlp_out")
    raise ValueError("Unsupported patch_locations number.")


def build_model(args):
    if args.task_type == "classification":
        if args.dataset_name in _DATASET_NUM_LABELS:
            num_labels = _DATASET_NUM_LABELS[args.dataset_name]
        elif args.dataset_name == "cifar100":
            num_labels = 100
        elif args.dataset_name == "food101":
            num_labels = 101
        elif args.dataset_name == "tiny-imagenet":
            num_labels = 200
        else:
            raise ValueError(f"Unsupported classification dataset: {args.dataset_name}")

        if args.model_name.startswith("vit"):
            model_id = {
                "vit-base": "google/vit-base-patch16-224-in21k",
                "vit-large": "google/vit-large-patch16-224-in21k",
                "vit-huge": "google/vit-huge-patch14-224-in21k",
            }[args.model_name]
            processor = ViTImageProcessor.from_pretrained(model_id)
            model = ViTForImageClassification.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                device_map="cuda:0",
            )
        else:
            model_id = {
                "dinov2-base": "facebook/dinov2-base",
                "dinov2-large": "facebook/dinov2-large",
                "dinov2-giant": "facebook/dinov2-giant",
            }[args.model_name]
            processor = AutoImageProcessor.from_pretrained(model_id)
            model = Dinov2ForImageClassification.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                num_labels=num_labels,
                ignore_mismatched_sizes=True,
                device_map="cuda:0",
            )
        return model, processor

    if args.task_type == "semantic_segmentation":
        model_id = "facebook/dinov2-base"
        config = AutoConfig.from_pretrained(model_id)
        config.num_labels = 21
        config.id2label = {i: str(i) for i in range(21)}
        config.label2id = {str(i): i for i in range(21)}
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = Dinov2ForSemanticSegmentationLite.from_pretrained(
            model_id,
            config=config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        return model, processor

    if args.task_type == "object_detection":
        model_id = "hustvl/yolos-tiny"
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = YolosForObjectDetection.from_pretrained(
            model_id,
            num_labels=len(VOC_DET_CLASSES),
            id2label=VOC_DET_ID2LABEL,
            label2id=VOC_DET_LABEL2ID,
            ignore_mismatched_sizes=True,
            device_map="cuda:0",
        )
        return model, processor

    raise ValueError(f"Unsupported task type: {args.task_type}")


def compute_rank_dict(args, model, val_dataset):
    if not args.dynamic_rank:
        return args.rank_ratio

    if "dinov2" in args.model_name:
        if args.energy_search:
            _, rank_dict = get_dinov2_rank_binary_search_energy_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                rank_ratio=args.rank_ratio,
            )
        else:
            _, rank_dict = get_dinov2_rank_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                base_ratio=args.rank_ratio,
                energy_ratio=args.energy_ratio,
            )
        return rank_dict

    if "yolos" in args.model_name:
        if args.energy_search:
            _, rank_dict = get_yolos_rank_binary_search_energy_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                rank_ratio=args.rank_ratio,
            )
        else:
            _, rank_dict = get_yolos_rank_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                base_ratio=args.rank_ratio,
                energy_ratio=args.energy_ratio,
            )
        return rank_dict

    if "vit" in args.model_name:
        if args.energy_search:
            _, rank_dict = get_vit_rank_binary_search_energy_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                rank_ratio=args.rank_ratio,
            )
        else:
            _, rank_dict = get_vit_rank_ratio(
                model,
                val_dataset,
                batch_size=args.per_device_train_batch_size,
                patch_locations=args.patch_locations,
                base_ratio=args.rank_ratio,
                energy_ratio=args.energy_ratio,
            )
        return rank_dict

    raise ValueError(f"Dynamic rank is unsupported for model: {args.model_name}")


def build_training_arguments(args):
    use_bf16 = args.task_type != "object_detection"
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=max(1, min(32, args.per_device_train_batch_size)),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=use_bf16,
        bf16_full_eval=use_bf16,
        use_liger_kernel=True,
        logging_steps=10,
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        remove_unused_columns=False,
        label_names=["labels"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        output_dir=args.output_dir,
    )
    training_args.eval_score_threshold = args.eval_score_threshold
    return training_args


def build_trainer(args, model, processor, train_dataset, val_dataset, rank_spec):
    data_collator = None
    compute_metrics = build_compute_metrics(args)
    preprocess_logits_for_metrics = None
    if args.task_type == "semantic_segmentation":
        data_collator = segmentation_collator
        preprocess_logits_for_metrics = lambda logits, labels: torch.argmax(logits, dim=1).to(torch.uint8)
    elif args.task_type == "object_detection":
        data_collator = detection_collator

    trainer_cls = VanillaVisionTrainer if args.vanilla_train else VisionTrainer
    trainer_kwargs = dict(
        model=model,
        args=build_training_arguments(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        task_type=args.task_type,
        image_processor=processor,
    )
    if trainer_cls is VisionTrainer:
        trainer_kwargs["meft_config"] = MeftConfig(
            patch_locations=get_patch_locations(args),
            compress_method=normalize_compress_method(args.compress_method),
            compress_kwargs={"rank": rank_spec},
            quant_method=args.quant_method,
        )
    return trainer_cls(**trainer_kwargs)


def main():
    args = parse_args()
    print("Training parameters:", args)

    if args.task_type == "semantic_segmentation" and args.model_name != "dinov2-base-seg":
        raise ValueError("semantic_segmentation only supports dinov2-base-seg in the first version.")
    if args.task_type == "object_detection" and args.model_name != "yolos-tiny":
        raise ValueError("object_detection only supports yolos-tiny in the first version.")

    if args.image_size is None:
        if args.task_type == "semantic_segmentation":
            args.image_size = 448
        elif args.task_type == "object_detection":
            args.image_size = 512
        else:
            args.image_size = 224

    set_random_seed(args.seed)
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args))

    model, processor = build_model(args)
    lora_config = build_lora_config(args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset, val_dataset, test_dataset = build_datasets(args, processor)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_start = time.time()

    rank_spec = compute_rank_dict(args, model, val_dataset)
    trainer = build_trainer(args, model, processor, train_dataset, val_dataset, rank_spec)
    resume_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    if resume_checkpoint is not None:
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    train_output = trainer.train(resume_from_checkpoint=resume_checkpoint)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_total_time = time.time() - train_start
    print(f"[Train] total time: {train_total_time:.2f}s")

    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_gb = peak_mem_bytes / (1024 ** 3)
        print(f"[Train] peak GPU memory: {peak_mem_gb:.2f} GB")
        wandb.log(
            {
                "train_total_time_s": train_total_time,
                "train_peak_gpu_mem_gb": peak_mem_gb,
            },
            step=train_output.global_step if hasattr(train_output, "global_step") else None,
        )
    else:
        peak_mem_gb = None

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    if args.task_type == "classification":
        print(f"test data top-1 accuracy: {test_results['eval_accuracy']:.4f}")
        wandb.summary["test_accuracy"] = test_results["eval_accuracy"]
    elif args.task_type == "semantic_segmentation":
        print(f"test data mIoU: {test_results['eval_miou']:.4f}")
        wandb.summary["test_miou"] = test_results["eval_miou"]
        wandb.summary["test_pixel_accuracy"] = test_results["eval_pixel_accuracy"]
    else:
        print(f"test data mAP@0.5: {test_results['eval_map50']:.4f}")
        wandb.summary["test_map50"] = test_results["eval_map50"]

    append_local_result(args.output_dir, args, test_results, train_total_time, peak_mem_gb)


if __name__ == "__main__":
    main()
