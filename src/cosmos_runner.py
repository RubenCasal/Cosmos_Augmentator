from __future__ import annotations

import inspect
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import CONTROL_NAMES, ControlName, CosmosConfig

logger = logging.getLogger(__name__)


class CosmosRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class CosmosGenerationRequest:
    image_path: Path
    prompt: str
    negative_prompt: str
    seed: int
    name: str
    control_paths: dict[ControlName, Path | None]


@dataclass
class CosmosBatchResult:
    outputs: dict[str, Path] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)
    elapsed_seconds: dict[str, float] = field(default_factory=dict)


def add_cosmos_to_sys_path(repo_root: Path) -> None:
    repo_str = str(repo_root.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _safe_instantiate(cls: type, payload: dict[str, Any]) -> Any:
    signature = inspect.signature(cls)
    supported: dict[str, Any] = {}
    for key, value in payload.items():
        if key in signature.parameters:
            supported[key] = value
    return cls(**supported)


def _looks_like_control_artifact(path: Path) -> bool:
    stem = path.stem.lower()
    tokens = (
        "seg",
        "depth",
        "edge",
        "mask",
        "control",
        "condition",
        "hint",
        "g_mask",
        "vis",
    )
    return any(
        stem.endswith(f"_{token}") or stem.endswith(f"-{token}") or f"_{token}_" in stem for token in tokens
    )


def _matches_request_output_name(path: Path, request_name: str) -> bool:
    stem = path.stem
    return stem == request_name or stem.startswith(f"{request_name}_") or stem.startswith(f"{request_name}-")


class CosmosRunner:
    def __init__(self, config: CosmosConfig) -> None:
        self.config = config
        self.inference: Any | None = None
        self._import_error: Exception | None = None

        self._InferenceArguments: type | None = None
        self._SetupArguments: type | None = None
        self._ModelKey: type | None = None
        self._Control2WorldInference: type | None = None

        try:
            add_cosmos_to_sys_path(config.repo_root)
            from cosmos_transfer2.config import InferenceArguments, ModelKey, SetupArguments

            try:
                from cosmos_transfer2.infer import Control2WorldInference
            except Exception:
                from cosmos_transfer2.inference import Control2WorldInference

            self._InferenceArguments = InferenceArguments
            self._SetupArguments = SetupArguments
            self._ModelKey = ModelKey
            self._Control2WorldInference = Control2WorldInference

            logger.info("Initializing Cosmos inference pipeline once (this may take time)...")
            setup_args = self._build_setup_args()
            self.inference = self._Control2WorldInference(setup_args)
        except Exception as exc:
            self._import_error = exc
            logger.warning(
                "Failed to import cosmos_transfer2 Python API from %s. "
                "Falling back to examples/inference.py subprocess mode. Error: %s",
                config.repo_root,
                exc,
            )

    @property
    def uses_python_api(self) -> bool:
        return self.inference is not None

    def _build_setup_args(self) -> Any:
        if self._SetupArguments is None:
            raise CosmosRunnerError("Cosmos SetupArguments is not available.")

        base_payload: dict[str, Any] = {
            "enable_guardrails": not self.config.disable_guardrails,
            "context_parallel_size": 1,
            "offload_guardrail_models": False,
            "variant": self.config.model_variant,
            "distilled": self.config.model_distilled,
        }

        model_candidates: list[str | None] = [self.config.model]
        if self.config.model is None:
            model_candidates.append(None)

        model_key_candidates: list[Any] = [None]
        if self._ModelKey is not None:
            for attr_name in ["CONTROL2WORLD", "CONTROL_TO_WORLD", "VIDEO2WORLD"]:
                key_value = getattr(self._ModelKey, attr_name, None)
                if key_value is not None:
                    model_key_candidates.append(key_value)

        last_error: Exception | None = None
        for model_value in model_candidates:
            for model_key in model_key_candidates:
                payload = dict(base_payload)
                if model_value is not None:
                    payload["model"] = model_value
                if model_key is not None:
                    payload["model_key"] = model_key

                try:
                    return _safe_instantiate(self._SetupArguments, payload)
                except Exception as exc:
                    last_error = exc

        raise CosmosRunnerError(
            "Unable to instantiate SetupArguments with available constructor variants. "
            f"Last error: {last_error}. Try setting cosmos.model in config to the exact "
            "model string used by your local Cosmos setup."
        )

    def _build_control_payload(self, control_paths: dict[ControlName, Path | None]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        controls = self.config.controls.as_dict()

        for control_name in CONTROL_NAMES:
            control_cfg = controls[control_name]
            if control_cfg.is_disabled:
                continue

            control_payload: dict[str, Any] = {
                "control_weight": control_cfg.weight,
            }

            if control_cfg.is_external:
                control_path = control_paths.get(control_name)
                if control_path is None:
                    raise CosmosRunnerError(
                        f"Control '{control_name}' is external but no path was provided for this sample."
                    )
                if not control_path.exists():
                    raise CosmosRunnerError(
                        f"External control file for '{control_name}' does not exist: {control_path}"
                    )
                control_payload["control_path"] = str(control_path)

            payload[control_name] = control_payload

        return payload

    def _build_raw_payload(self, request: CosmosGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": request.name,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "video_path": str(request.image_path),
            "seed": int(request.seed),
            "guidance": self.config.guidance,
            "num_steps": self.config.num_steps,
            "resolution": self.config.resolution,
            "max_frames": self.config.max_frames,
            "num_video_frames_per_chunk": self.config.num_video_frames_per_chunk,
        }
        payload.update(self._build_control_payload(request.control_paths))
        return payload

    def _build_inference_args(self, request: CosmosGenerationRequest) -> Any:
        if self._InferenceArguments is None:
            raise CosmosRunnerError("Cosmos InferenceArguments is not available.")

        raw_payload = self._build_raw_payload(request)

        try:
            return _safe_instantiate(self._InferenceArguments, raw_payload)
        except Exception as direct_exc:
            logger.debug("Direct InferenceArguments constructor failed, trying from_files: %s", direct_exc)

        from_files = getattr(self._InferenceArguments, "from_files", None)
        if callable(from_files):
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as temp_file:
                    json.dump(raw_payload, temp_file, indent=2)
                    temp_path = Path(temp_file.name)

                loaded = from_files([str(temp_path)])
                if isinstance(loaded, list) and loaded:
                    return loaded[0]
                raise CosmosRunnerError("InferenceArguments.from_files returned an empty list.")
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        raise CosmosRunnerError("Could not build InferenceArguments via constructor or from_files.")

    def _call_generate(self, generator: Any, inference_payload: Any, output_dir: Path) -> Any:
        attempts: list[tuple[str, dict[str, Any] | None]] = [
            ("output_dir", {"output_dir": str(output_dir)}),
            ("output_path", {"output_path": str(output_dir)}),
            ("positional", None),
        ]

        last_error: Exception | None = None
        for mode, kwargs in attempts:
            try:
                if kwargs is None:
                    return generator(inference_payload, str(output_dir))
                return generator(inference_payload, **kwargs)
            except TypeError as exc:
                last_error = exc
                logger.debug("Cosmos generate call with %s failed: %s", mode, exc)
            except Exception as exc:
                # Runtime failures should bubble out immediately.
                raise CosmosRunnerError(f"Cosmos generation failed with mode '{mode}': {exc}") from exc

        raise CosmosRunnerError(f"Unable to call Cosmos generate with available signatures: {last_error}")

    def _extract_output_path(self, raw_out: Any, output_dir: Path, name: str) -> Path:
        candidates: list[Path] = []

        if isinstance(raw_out, dict):
            maybe_path = raw_out.get("output_path")
            if isinstance(maybe_path, str):
                candidates.append(Path(maybe_path))
            output_paths = raw_out.get("output_paths")
            if isinstance(output_paths, list):
                for item in output_paths:
                    if isinstance(item, str):
                        candidates.append(Path(item))
                    elif isinstance(item, dict):
                        output_path = item.get("output_path")
                        if isinstance(output_path, str):
                            candidates.append(Path(output_path))
        elif isinstance(raw_out, list):
            for item in raw_out:
                if isinstance(item, str):
                    candidates.append(Path(item))
                elif isinstance(item, dict):
                    output_path = item.get("output_path")
                    if isinstance(output_path, str):
                        candidates.append(Path(output_path))

        for candidate in candidates:
            maybe_path = candidate if candidate.is_absolute() else (output_dir / candidate)
            if maybe_path.exists() and maybe_path.is_file() and not _looks_like_control_artifact(maybe_path):
                return maybe_path.resolve()

        globbed = sorted(
            (
                p
                for p in output_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                and _matches_request_output_name(p, name)
            ),
            key=lambda p: p.stat().st_mtime,
        )

        preferred = [p for p in globbed if not _looks_like_control_artifact(p)]
        if preferred:
            return preferred[-1].resolve()
        if globbed:
            return globbed[-1].resolve()

        raise CosmosRunnerError(f"No output generated for sample '{name}'.")

    def _run_single_request(self, request: CosmosGenerationRequest, output_dir: Path) -> Path:
        if not request.image_path.exists():
            raise CosmosRunnerError(f"Image does not exist: {request.image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.inference is not None:
            try:
                inference_args = self._build_inference_args(request)
                generator = getattr(self.inference, "generate", None)
                if callable(generator):
                    raw_out = self._call_generate(generator, inference_args, output_dir)
                    return self._extract_output_path(raw_out=raw_out, output_dir=output_dir, name=request.name)
            except Exception as exc:
                logger.warning(
                    "Direct Python API generation failed for '%s'. Falling back to subprocess mode. Error: %s",
                    request.name,
                    exc,
                )

        generated_map = self._fallback_generate_many([request], output_dir=output_dir)
        generated = generated_map.get(request.name)
        if generated is None:
            raise CosmosRunnerError(f"Fallback generation did not produce output for '{request.name}'.")
        return generated

    def run_single(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: str,
        output_dir: Path,
        seed: int,
        name: str,
        control_paths: dict[ControlName, Path | None],
    ) -> Path:
        request = CosmosGenerationRequest(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            name=name,
            control_paths=control_paths,
        )
        return self._run_single_request(request=request, output_dir=output_dir)

    def run_many(self, requests: list[CosmosGenerationRequest], output_dir: Path) -> CosmosBatchResult:
        result = CosmosBatchResult()
        if not requests:
            return result

        seen_names: set[str] = set()
        for request in requests:
            if request.name in seen_names:
                raise CosmosRunnerError(f"Duplicated request name in batch: {request.name}")
            seen_names.add(request.name)

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.uses_python_api:
            for request in requests:
                start = time.perf_counter()
                try:
                    generated = self._run_single_request(request=request, output_dir=output_dir)
                    result.outputs[request.name] = generated
                except Exception as exc:
                    result.errors[request.name] = exc
                finally:
                    result.elapsed_seconds[request.name] = time.perf_counter() - start
            return result

        # Subprocess-only mode: keep one long-lived inference process for the whole augmentation.
        # This avoids reloading/downloading models per image.
        logger.info("Running Cosmos subprocess in single-process mode for %d samples.", len(requests))
        self._run_subprocess_chunk_with_retry(chunk=requests, output_dir=output_dir, result=result)

        return result

    def _run_subprocess_chunk_with_retry(
        self,
        chunk: list[CosmosGenerationRequest],
        output_dir: Path,
        result: CosmosBatchResult,
    ) -> None:
        batch_start = time.perf_counter()
        try:
            outputs = self._fallback_generate_many(chunk, output_dir)
            elapsed = time.perf_counter() - batch_start
            per_sample = elapsed / len(chunk)

            for request in chunk:
                result.elapsed_seconds[request.name] = per_sample
                output = outputs.get(request.name)
                if output is None:
                    result.errors[request.name] = CosmosRunnerError(
                        f"Subprocess batch completed but no output was found for '{request.name}'."
                    )
                else:
                    result.outputs[request.name] = output
            return
        except Exception as exc:
            elapsed = time.perf_counter() - batch_start
            if len(chunk) == 1:
                request = chunk[0]
                result.errors[request.name] = exc
                result.elapsed_seconds[request.name] = elapsed
                logger.error("Sample '%s' failed in subprocess mode: %s", request.name, exc)
                return

            mid = len(chunk) // 2
            logger.warning(
                "Subprocess batch of %d samples failed. Splitting into %d + %d to isolate bad samples.",
                len(chunk),
                mid,
                len(chunk) - mid,
            )
            self._run_subprocess_chunk_with_retry(chunk[:mid], output_dir, result)
            self._run_subprocess_chunk_with_retry(chunk[mid:], output_dir, result)

    def _clear_existing_outputs(self, output_dir: Path, request_names: list[str]) -> None:
        for name in request_names:
            for candidate in output_dir.iterdir():
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    and _matches_request_output_name(candidate, name)
                ):
                    candidate.unlink(missing_ok=True)

    def _fallback_generate_many(self, requests: list[CosmosGenerationRequest], output_dir: Path) -> dict[str, Path]:
        if not requests:
            return {}

        self._clear_existing_outputs(output_dir, [request.name for request in requests])
        payload_dir = Path(tempfile.mkdtemp(prefix="cosmos_payloads_"))
        payload_paths: list[Path] = []

        try:
            for idx, request in enumerate(requests):
                payload = self._build_raw_payload(request)
                safe_name = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in request.name)
                payload_path = payload_dir / f"{idx:05d}_{safe_name}.json"
                with payload_path.open("w", encoding="utf-8") as file_handle:
                    json.dump(payload, file_handle, indent=2)
                payload_paths.append(payload_path)

            cmd = [
                sys.executable,
                "examples/inference.py",
                "-i",
                *[str(path) for path in payload_paths],
                "-o",
                str(output_dir),
            ]
            if self.config.disable_guardrails:
                cmd.append("--disable-guardrails")

            completed = subprocess.run(
                cmd,
                cwd=self.config.repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            if completed.returncode != 0:
                raise CosmosRunnerError(
                    "Fallback subprocess generation failed.\n"
                    f"Payload JSON directory kept at: {payload_dir}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout: {completed.stdout[-4000:]}\n"
                    f"stderr: {completed.stderr[-4000:]}"
                )

            outputs: dict[str, Path] = {}
            for request in requests:
                outputs[request.name] = self._extract_output_path(raw_out=[], output_dir=output_dir, name=request.name)

            shutil.rmtree(payload_dir, ignore_errors=True)
            return outputs
        except Exception:
            # Keep payloads for debugging if this batch fails.
            raise
