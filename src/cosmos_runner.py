from __future__ import annotations

import inspect
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .types import CONTROL_NAMES, ControlName, CosmosConfig

logger = logging.getLogger(__name__)


class CosmosRunnerError(RuntimeError):
    pass


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

    def _build_inference_args(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: str,
        seed: int,
        name: str,
        control_paths: dict[ControlName, Path | None],
    ) -> Any:
        if self._InferenceArguments is None:
            raise CosmosRunnerError("Cosmos InferenceArguments is not available.")

        raw_payload: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "video_path": str(image_path),
            "seed": int(seed),
            "guidance": self.config.guidance,
            "num_steps": self.config.num_steps,
            "resolution": self.config.resolution,
            "max_frames": self.config.max_frames,
            "num_video_frames_per_chunk": self.config.num_video_frames_per_chunk,
        }
        raw_payload.update(self._build_control_payload(control_paths))

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
            if maybe_path.exists() and maybe_path.is_file():
                return maybe_path.resolve()

        globbed = sorted(
            p for p in output_dir.glob(f"{name}*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if globbed:
            return globbed[-1].resolve()

        raise CosmosRunnerError(f"No output generated for sample '{name}'.")

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
        if not image_path.exists():
            raise CosmosRunnerError(f"Image does not exist: {image_path}")

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.inference is not None:
            try:
                inference_args = self._build_inference_args(
                    image_path=image_path,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed,
                    name=name,
                    control_paths=control_paths,
                )
                generator = getattr(self.inference, "generate", None)
                if callable(generator):
                    raw_out = generator(inference_args, output_path=str(output_dir))
                    return self._extract_output_path(raw_out=raw_out, output_dir=output_dir, name=name)
            except Exception as exc:
                logger.warning(
                    "Direct Python API generation failed for '%s'. Falling back to subprocess mode. Error: %s",
                    name,
                    exc,
                )

        return self._fallback_generate(
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            output_dir=output_dir,
            seed=seed,
            name=name,
            control_paths=control_paths,
        )

    def _fallback_generate(
        self,
        image_path: Path,
        prompt: str,
        negative_prompt: str,
        output_dir: Path,
        seed: int,
        name: str,
        control_paths: dict[ControlName, Path | None],
    ) -> Path:
        payload: dict[str, Any] = {
            "name": name,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "video_path": str(image_path),
            "seed": int(seed),
            "guidance": self.config.guidance,
            "num_steps": self.config.num_steps,
            "resolution": self.config.resolution,
            "max_frames": self.config.max_frames,
            "num_video_frames_per_chunk": self.config.num_video_frames_per_chunk,
        }
        payload.update(self._build_control_payload(control_paths))

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".json",
                delete=False,
                encoding="utf-8",
            ) as temp_file:
                json.dump(payload, temp_file, indent=2)
                temp_path = Path(temp_file.name)

            cmd = [
                sys.executable,
                "examples/inference.py",
                "-i",
                str(temp_path),
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
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout: {completed.stdout[-4000:]}\n"
                    f"stderr: {completed.stderr[-4000:]}"
                )

            candidates = sorted(
                p for p in output_dir.glob(f"{name}*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not candidates:
                raise CosmosRunnerError(
                    f"Fallback generation succeeded but no output file matched '{name}*' in {output_dir}"
                )

            return candidates[-1].resolve()
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
