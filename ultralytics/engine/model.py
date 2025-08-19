# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import inspect
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    SETTINGS,
    YAML,
    callbacks,
    checks,
)


class Model(torch.nn.Module):
    """
    A base class for implementing YOLO models, unifying APIs across different model types.

    This class provides a common interface for various operations related to YOLO models, such as training,
    validation, prediction, exporting, and benchmarking. It handles different types of models, including those
    loaded from local files, Ultralytics HUB, or Triton Server.

    Attributes:
        callbacks (dict): A dictionary of callback functions for various events during model operations.
        predictor (BasePredictor): The predictor object used for making predictions.
        model (torch.nn.Module): The underlying PyTorch model.
        trainer (BaseTrainer): The trainer object used for training the model.
        ckpt (dict): The checkpoint data if the model is loaded from a *.pt file.
        cfg (str): The configuration of the model if loaded from a *.yaml file.
        ckpt_path (str): The path to the checkpoint file.
        overrides (dict): A dictionary of overrides for model configuration.
        metrics (dict): The latest training/validation metrics.
        session (HUBTrainingSession): The Ultralytics HUB session, if applicable.
        task (str): The type of task the model is intended for.
        model_name (str): The name of the model.

    Methods:
        __call__: Alias for the predict method, enabling the model instance to be callable.
        _new: Initialize a new model based on a configuration file.
        _load: Load a model from a checkpoint file.
        _check_is_pytorch_model: Ensure that the model is a PyTorch model.
        reset_weights: Reset the model's weights to their initial state.
        load: Load model weights from a specified file.
        save: Save the current state of the model to a file.
        info: Log or return information about the model.
        fuse: Fuse Conv2d and BatchNorm2d layers for optimized inference.
        predict: Perform object detection predictions.
        track: Perform object tracking.
        val: Validate the model on a dataset.
        benchmark: Benchmark the model on various export formats.
        export: Export the model to different formats.
        train: Train the model on a dataset.
        tune: Perform hyperparameter tuning.
        _apply: Apply a function to the model's tensors.
        add_callback: Add a callback function for an event.
        clear_callback: Clear all callbacks for an event.
        reset_callbacks: Reset all callbacks to their default functions.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo11n.pt")
        >>> results = model.predict("image.jpg")
        >>> model.train(data="coco8.yaml", epochs=3)
        >>> metrics = model.val()
        >>> model.export(format="onnx")
    """

    def __init__(
        self,
        model: Union[str, Path, "Model"] = "yolo11n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        初始化YOLO模型类的新实例。

        这是Model类的构造函数，负责根据提供的模型路径或名称设置模型。
        支持多种模型来源，包括本地文件、Ultralytics HUB模型、Triton Server模型，
        或者已经初始化的Model实例。

        功能特点：
        - 支持多种模型来源：本地文件、HUB模型、Triton Server
        - 自动推断任务类型（如检测、分割、分类等）
        - 支持模型配置的灵活覆盖
        - 初始化训练、预测、导出所需的核心组件

        初始化流程：
        1. 检查是否为已初始化的Model实例（如果是则复制属性）
        2. 检查是否为Ultralytics HUB模型
        3. 检查是否为Triton Server模型
        4. 根据文件扩展名决定加载方式（.yaml/.yml新建，其他格式加载权重）
        5. 设置模型相关属性
        6. 清理父类的training属性以避免冲突

        Args:
            model: 模型路径或名称，支持多种格式：
                   - 字符串：本地文件路径（如"yolo11n.pt"）
                   - 字符串：Ultralytics HUB模型ID
                   - 字符串：Triton Server URL
                   - Path：Path对象格式的文件路径
                   - Model：已初始化的Model实例（将复制属性）
            task: 模型任务类型，可选值：
                  - 'detect': 目标检测
                  - 'segment': 实例分割
                  - 'classify': 图像分类
                  - 'pose': 姿态估计
                  - 'obb': 定向边界框检测
                  如果为None，将自动从配置推断
            verbose: 是否启用详细输出模式，显示模型加载和初始化的详细信息

        Raises:
            FileNotFoundError: 指定模型文件不存在或无法访问
            ValueError: 模型文件或配置无效/不支持
            ImportError: 特定模型类型依赖的库未安装（如HUB SDK）

        使用示例：
            # 从预训练权重加载
            >>> model = Model("yolo11n.pt")
            
            # 从配置文件新建模型
            >>> model = Model("yolo11n.yaml", task="detect")
            
            # 从Ultralytics HUB加载
            >>> model = Model("https://hub.ultralytics.com/models/MODEL_ID")
            
            # 从Triton Server加载
            >>> model = Model("http://localhost:8000/v2/models/yolo11n")
            
            # 复制已有模型
            >>> original_model = Model("yolo11n.pt")
            >>> new_model = Model(original_model)
        """
        if isinstance(model, Model):
            self.__dict__ = model.__dict__  # accepts an already initialized Model
            return
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = {}  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        self.model_name = None  # model name
        model = str(model).strip()

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            from ultralytics.hub import HUBTrainingSession

            # Fetch model from HUB
            checks.check_requirements("hub-sdk>=0.0.12")
            session = HUBTrainingSession.create_session(model)
            model = session.model_file
            if session.train_args:  # training sent from HUB
                self.session = session

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            self.overrides["task"] = task or "detect"  # set `task=detect` if not explicitly set
            return

        # Load or create new YOLO model
        __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # to avoid deterministic warnings
        if str(model).endswith((".yaml", ".yml")):
            self._new(model, task=task, verbose=verbose)
        else:
            self._load(model, task=task)

        # Delete super().training for accessing self.model.training
        del self.training

    def __call__(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Alias for the predict method, enabling the model instance to be callable for predictions.

        This method simplifies the process of making predictions by allowing the model instance to be called
        directly with the required arguments.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray | torch.Tensor | List | Tuple): The source of
                the image(s) to make predictions on. Can be a file path, URL, PIL image, numpy array, PyTorch
                tensor, or a list/tuple of these.
            stream (bool): If True, treat the input source as a continuous stream for predictions.
            **kwargs (Any): Additional keyword arguments to configure the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, each encapsulated in a
                Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model("https://ultralytics.com/images/bus.jpg")
            >>> for r in results:
            ...     print(f"Detected {len(r)} objects in image")
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """
        Check if the given model string is a Triton Server URL.

        This static method determines whether the provided model string represents a valid Triton Server URL by
        parsing its components using urllib.parse.urlsplit().

        Args:
            model (str): The model string to be checked.

        Returns:
            (bool): True if the model string is a valid Triton Server URL, False otherwise.

        Examples:
            >>> Model.is_triton_model("http://localhost:8000/v2/models/yolo11n")
            True
            >>> Model.is_triton_model("yolo11n.pt")
            False
        """
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """
        Check if the provided model is an Ultralytics HUB model.

        This static method determines whether the given model string represents a valid Ultralytics HUB model
        identifier.

        Args:
            model (str): The model string to check.

        Returns:
            (bool): True if the model is a valid Ultralytics HUB model, False otherwise.

        Examples:
            >>> Model.is_hub_model("https://hub.ultralytics.com/models/MODEL")
            True
            >>> Model.is_hub_model("yolo11n.pt")
            False
        """
        from ultralytics.hub import HUB_WEB_ROOT

        return model.startswith(f"{HUB_WEB_ROOT}/models/")

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initialize a new model and infer the task type from model definitions.

        Creates a new model instance based on the provided configuration file. Loads the model configuration, infers
        the task type if not specified, and initializes the model using the appropriate class from the task map.

        Args:
            cfg (str): Path to the model configuration file in YAML format.
            task (str, optional): The specific task for the model. If None, it will be inferred from the config.
            model (torch.nn.Module, optional): A custom model instance. If provided, it will be used instead of
                creating a new one.
            verbose (bool): If True, displays model information during loading.

        Raises:
            ValueError: If the configuration file is invalid or the task cannot be inferred.
            ImportError: If the required dependencies for the specified task are not installed.

        Examples:
            >>> model = Model()
            >>> model._new("yolo11n.yaml", task="detect", verbose=True)
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        Load a model from a checkpoint file or initialize it from a weights file.

        This method handles loading models from either .pt checkpoint files or other weight file formats. It sets
        up the model, task, and related attributes based on the loaded weights.

        Args:
            weights (str): Path to the model weights file to be loaded.
            task (str, optional): The task associated with the model. If None, it will be inferred from the model.

        Raises:
            FileNotFoundError: If the specified weights file does not exist or is inaccessible.
            ValueError: If the weights file format is unsupported or invalid.

        Examples:
            >>> model = Model()
            >>> model._load("yolo11n.pt")
            >>> model._load("path/to/weights.pth", task="detect")
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights, download_dir=SETTINGS["weights_dir"])  # download and return local file
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolo11n -> yolo11n.pt

        if str(weights).rpartition(".")[-1] == "pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)
            self.task = self.model.task
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """
        Check if the model is a PyTorch model and raise TypeError if it's not.

        This method verifies that the model is either a PyTorch module or a .pt file. It's used to ensure that
        certain operations that require a PyTorch model are only performed on compatible model types.

        Raises:
            TypeError: If the model is not a PyTorch module or a .pt file. The error message provides detailed
                information about supported model formats and operations.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model._check_is_pytorch_model()  # No error raised
            >>> model = Model("yolo11n.onnx")
            >>> model._check_is_pytorch_model()  # Raises TypeError
        """
        pt_str = isinstance(self.model, (str, Path)) and str(self.model).rpartition(".")[-1] == "pt"
        pt_module = isinstance(self.model, torch.nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolo11n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        """
        Reset the model's weights to their initial state.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True,
        enabling them to be updated during training.

        Returns:
            (Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.reset_weights()
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolo11n.pt") -> "Model":
        """
        Load parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object.

        Returns:
            (Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model()
            >>> model.load("yolo11n.pt")
            >>> model.load(Path("path/to/weights.pt"))
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            self.overrides["pretrained"] = weights  # remember the weights for DDP training
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt") -> None:
        """
        Save the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename. It includes metadata such as
        the date, Ultralytics version, license information, and a link to the documentation.

        Args:
            filename (str | Path): The name of the file to save the model to.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.save("my_model.pt")
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Display model information.

        This method provides an overview or detailed information about the model, depending on the arguments
        passed. It can control the verbosity of the output and return the information as a list.

        Args:
            detailed (bool): If True, shows detailed information about the model layers and parameters.
            verbose (bool): If True, prints the information. If False, returns the information as a list.

        Returns:
            (List[str]): A list of strings containing various types of information about the model, including
                model summary, layer details, and parameter counts. Empty if verbose is True.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.info()  # Prints model summary
            >>> info_list = model.info(detailed=True, verbose=False)  # Returns detailed info as a list
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self) -> None:
        """
        Fuse Conv2d and BatchNorm2d layers in the model for optimized inference.

        This method iterates through the model's modules and fuses consecutive Conv2d and BatchNorm2d layers
        into a single layer. This fusion can significantly improve inference speed by reducing the number of
        operations and memory accesses required during forward passes.

        The fusion process typically involves folding the BatchNorm2d parameters (mean, variance, weight, and
        bias) into the preceding Conv2d layer's weights and biases. This results in a single Conv2d layer that
        performs both convolution and normalization in one step.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model.fuse()
            >>> # Model is now fused and ready for optimized inference
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        """
        Generate image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image
        source. It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor): The source of the image for
                generating embeddings. Can be a file path, URL, PIL image, numpy array, etc.
            stream (bool): If True, predictions are streamed.
            **kwargs (Any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> image = "https://ultralytics.com/images/bus.jpg"
            >>> embeddings = model.embed(image)
            >>> print(embeddings[0].shape)
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, Image.Image, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> List[Results]:
        """
        使用YOLO模型对给定图像源进行预测推理。

        这是Model类的核心方法之一，用于执行目标检测、分割等任务的推理。
        支持多种图像输入格式，包括文件路径、URL、PIL图像、numpy数组和PyTorch张量。

        功能特点：
        - 支持批量推理和流式处理
        - 可配置置信度阈值、NMS阈值等参数
        - 自动处理图像预处理和后处理
        - 支持自定义预测器

        Args:
            source: 输入图像源，支持多种格式：
                   - 字符串：文件路径或URL
                   - PIL.Image：PIL图像对象
                   - np.ndarray：numpy数组格式的图像
                   - torch.Tensor：PyTorch张量
                   - list/tuple：上述格式的批量输入
            stream: 是否以流模式处理输入（适用于视频流）
            predictor: 自定义预测器实例，None则使用默认预测器
            **kwargs: 其他配置参数，如：
                     - conf: 置信度阈值（默认0.25）
                     - iou: NMS IoU阈值
                     - imgsz: 输入图像大小
                     - device: 运行设备（cpu/cuda）

        Returns:
            List[Results]: 预测结果列表，每个元素是一个Results对象，包含：
                          - boxes: 检测框信息
                          - masks: 分割掩码（分割任务）
                          - keypoints: 关键点（姿态估计任务）
                          - names: 类别名称
                          - orig_img: 原始图像

        使用示例：
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.predict("image.jpg", conf=0.5)
            >>> for r in results:
            ...     print(f"检测到 {len(r.boxes)} 个目标")
            ...     for box in r.boxes:
            ...         print(f"类别: {r.names[int(box.cls)]}, 置信度: {box.conf:.2f}")

        内部流程：
        1. 检查输入源，设置默认值
        2. 配置预测参数
        3. 初始化预测器（如需要）
        4. 执行推理
        5. 返回后处理结果
        """
        if source is None:
            source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
            LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Conduct object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (str | Path | int | List | Tuple | np.ndarray | torch.Tensor, optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ):
        """
        Validate the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for customization through various settings. It
        supports validation with a custom validator or the default validation approach. The method combines default
        configurations, method-specific defaults, and user-provided arguments to configure the validation process.

        Args:
            validator (ultralytics.engine.validator.BaseValidator, optional): An instance of a custom validator class
                for validating the model.
            **kwargs (Any): Arbitrary keyword arguments for customizing the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.val(data="coco8.yaml", imgsz=640)
            >>> print(results.box.map)  # Print mAP50-95
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(self, data=None, format="", verbose=False, **kwargs: Any):
        """
        Benchmark the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is
        configured using a combination of default configuration values, model-specific arguments, method-specific
        defaults, and any additional user-provided keyword arguments.

        Args:
            data (str): Path to the dataset for benchmarking.
            verbose (bool): Whether to print detailed benchmark information.
            format (str): Export format name for specific benchmarking.
            **kwargs (Any): Arbitrary keyword arguments to customize the benchmarking process. Common options include:
                - imgsz (int | List[int]): Image size for benchmarking.
                - half (bool): Whether to use half-precision (FP16) mode.
                - int8 (bool): Whether to use int8 precision mode.
                - device (str): Device to run the benchmark on (e.g., 'cpu', 'cuda').

        Returns:
            (dict): A dictionary containing the results of the benchmarking process, including metrics for
                different export formats.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.benchmark(data="coco8.yaml", imgsz=640, half=True)
            >>> print(results)
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        from .exporter import export_formats

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        fmts = export_formats()
        export_args = set(dict(zip(fmts["Argument"], fmts["Arguments"])).get(format, [])) - {"batch"}
        export_kwargs = {k: v for k, v in args.items() if k in export_args}
        return benchmark(
            model=self,
            data=data,  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            device=args["device"],
            verbose=verbose,
            format=format,
            **export_kwargs,
        )

    def export(
        self,
        **kwargs: Any,
    ) -> str:
        """
        将模型导出为适合部署的不同格式。

        这是Model类的核心导出方法，支持将PyTorch模型转换为多种部署格式。
        支持导出为ONNX、TensorRT、CoreML、TFLite、OpenVINO等格式，便于在不同平台和框架上部署。

        支持的导出格式：
        - onnx: Open Neural Network Exchange格式
        - engine: TensorRT引擎（NVIDIA GPU优化）
        - coreml: Apple CoreML格式（iOS/macOS）
        - tflite: TensorFlow Lite（移动设备）
        - openvino: Intel OpenVINO格式
        - saved_model: TensorFlow SavedModel
        - torchscript: PyTorch TorchScript
        - paddle: PaddlePaddle格式

        功能特点：
        - 支持FP32、FP16、INT8量化导出
        - 支持动态输入尺寸
        - 支持模型简化和优化
        - 自动处理输入输出节点
        - 支持添加后处理模块（如NMS）

        导出流程：
        1. 检查是否为PyTorch模型
        2. 解析导出配置参数
        3. 初始化导出器
        4. 执行模型转换
        5. 保存导出文件
        6. 验证导出结果

        Args:
            **kwargs: 导出配置参数，常用参数包括：
                     - format: 导出格式（默认'onnx'）
                     - imgsz: 输入图像大小（默认模型配置）
                     - batch: 批次大小（默认1）
                     - device: 导出设备（默认'cpu'）
                     - half: 是否使用FP16半精度（默认False）
                     - int8: 是否使用INT8量化（默认False）
                     - dynamic: 是否启用动态输入尺寸（默认False）
                     - simplify: 是否简化ONNX模型（默认False）
                     - nms: 是否添加NMS后处理（默认False）
                     - workspace: TensorRT工作空间大小（默认4GB）
                     - data: 用于INT8校准的数据集路径

        Returns:
            str: 导出模型的文件路径

        Raises:
            AssertionError: 如果模型不是PyTorch模型
            ValueError: 如果指定了不支持的导出格式
            RuntimeError: 如果导出过程失败

        使用示例：
            >>> model = YOLO("yolo11n.pt")
            >>> # 导出为ONNX格式
            >>> onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
            >>> print(f"模型已导出到: {onnx_path}")

            >>> # 导出为TensorRT引擎
            >>> engine_path = model.export(format="engine", half=True, workspace=8)

            >>> # 导出为CoreML格式
            >>> coreml_path = model.export(format="coreml", nms=True)

        注意事项：
        - TensorRT导出需要NVIDIA GPU和TensorRT库
        - INT8量化需要校准数据集，可能影响模型精度
        - 动态输入可能增加推理延迟
        - 某些格式不支持所有YOLO功能（如NMS）
        - 导出后建议验证模型性能和精度
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs: Any,
    ):
        """
        使用指定数据集和配置训练模型。

        这是Model类的核心训练方法，提供完整的模型训练流程。支持从检查点恢复训练、
        与Ultralytics HUB集成、以及训练后的模型更新等功能。

        功能特点：
        - 支持自定义训练器或默认训练器
        - 自动处理数据集加载和预处理
        - 支持分布式训练（DDP）
        - 集成Ultralytics HUB训练管理
        - 自动保存最佳模型和训练日志
        - 支持训练中断后恢复

        训练流程：
        1. 检查是否为PyTorch模型
        2. 处理Ultralytics HUB集成（如使用）
        3. 检查pip包更新
        4. 合并和解析训练配置参数
        5. 初始化训练器
        6. 设置模型结构（如非恢复训练）
        7. 执行训练过程
        8. 更新模型和配置（训练完成后）

        Args:
            trainer: 自定义训练器实例，None则使用默认训练器
            **kwargs: 训练配置参数，常用参数包括：
                     - data: 数据集配置文件路径（必需）
                     - epochs: 训练轮数（默认100）
                     - batch: 批次大小（默认16）
                     - imgsz: 输入图像大小（默认640）
                     - device: 训练设备（'cpu', 'cuda', '0', '0,1,2,3'）
                     - workers: 数据加载线程数（默认8）
                     - optimizer: 优化器类型（'SGD', 'Adam', 'AdamW'）
                     - lr0: 初始学习率（默认0.01）
                     - patience: 早停耐心轮数（默认50）
                     - save: 是否保存训练结果（默认True）
                     - cache: 是否缓存数据集（默认False）
                     - resume: 是否从检查点恢复训练

        Returns:
            Dict | None: 训练指标字典，包含：
                        - fitness: 综合适应度分数
                        - metrics/mAP50-95: 各类别的mAP
                        - metrics/precision: 精确率
                        - metrics/recall: 召回率
                        - 其他任务特定指标
                        如果训练失败则返回None

        使用示例：
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.train(
            ...     data="coco8.yaml",
            ...     epochs=100,
            ...     batch=16,
            ...     imgsz=640,
            ...     device="cuda:0"
            ... )
            >>> print(f"最佳mAP50-95: {results.results_dict['metrics/mAP50-95']:.3f}")

        注意事项：
        - 使用Ultralytics HUB时，本地参数会被HUB参数覆盖
        - 训练过程中会自动创建runs/train目录保存结果
        - 支持通过resume参数从上次中断处继续训练
        - 多GPU训练时会自动使用分布式训练
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        if isinstance(kwargs.get("pretrained", None), (str, Path)):
            self.load(kwargs["pretrained"])  # load pretrained weights if provided
        overrides = YAML.load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, self.ckpt = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Conduct hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): Whether to use Ray Tune for hyperparameter tuning. If False, uses internal tuning method.
            iterations (int): Number of tuning iterations to perform.
            *args (Any): Additional positional arguments to pass to the tuner.
            **kwargs (Any): Additional keyword arguments for tuning configuration. These are combined with model
                overrides and defaults to configure the tuning process.

        Returns:
            (dict): Results of the hyperparameter search, including best parameters and performance metrics.

        Raises:
            TypeError: If the model is not a PyTorch model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.tune(data="coco8.yaml", iterations=5)
            >>> print(results)

            # Use Ray Tune for more advanced hyperparameter search
            >>> results = model.tune(use_ray=True, iterations=20, data="coco8.yaml")
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":
        """
        Apply a function to model tensors that are not parameters or registered buffers.

        This method extends the functionality of the parent class's _apply method by additionally resetting the
        predictor and updating the device in the model's overrides. It's typically used for operations like
        moving the model to a different device or changing its precision.

        Args:
            fn (Callable): A function to be applied to the model's tensors. This is typically a method like
                to(), cpu(), cuda(), half(), or float().

        Returns:
            (Model): The model instance with the function applied and updated attributes.

        Raises:
            AssertionError: If the model is not a PyTorch model.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> model = model._apply(lambda t: t.cuda())  # Move model to GPU
        """
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> Dict[int, str]:
        """
        Retrieve the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module. If the predictor is not
        initialized, it sets it up before retrieving the names.

        Returns:
            (Dict[int, str]): A dictionary of class names associated with the model, where keys are class indices and
                values are the corresponding class names.

        Raises:
            AttributeError: If the model or predictor does not have a 'names' attribute.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.names)
            {0: 'person', 1: 'bicycle', 2: 'car', ...}
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            predictor.setup_model(model=self.model, verbose=False)  # do not mess with self.predictor.model args
            return predictor.model.names
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Get the device on which the model's parameters are allocated.

        This property determines the device (CPU or GPU) where the model's parameters are currently stored. It is
        applicable only to models that are instances of torch.nn.Module.

        Returns:
            (torch.device): The device (CPU/GPU) of the model.

        Raises:
            AttributeError: If the model is not a torch.nn.Module instance.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.device)
            device(type='cuda', index=0)  # if CUDA is available
            >>> model = model.to("cpu")
            >>> print(model.device)
            device(type='cpu')
        """
        return next(self.model.parameters()).device if isinstance(self.model, torch.nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieve the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model. The transforms
        typically include preprocessing steps like resizing, normalization, and data augmentation
        that are applied to input data before it is fed into the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> transforms = model.transforms
            >>> if transforms:
            ...     print(f"Model transforms: {transforms}")
            ... else:
            ...     print("No transforms defined for this model.")
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Add a callback function for a specified event.

        This method allows registering custom callback functions that are triggered on specific events during
        model operations such as training or inference. Callbacks provide a way to extend and customize the
        behavior of the model at various stages of its lifecycle.

        Args:
            event (str): The name of the event to attach the callback to. Must be a valid event name recognized
                by the Ultralytics framework.
            func (Callable): The callback function to be registered. This function will be called when the
                specified event occurs.

        Raises:
            ValueError: If the event name is not recognized or is invalid.

        Examples:
            >>> def on_train_start(trainer):
            ...     print("Training is starting!")
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", on_train_start)
            >>> model.train(data="coco8.yaml", epochs=1)
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clear all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.
        It resets the callback list for the specified event to an empty list, effectively removing all
        registered callbacks for that event.

        Args:
            event (str): The name of the event for which to clear the callbacks. This should be a valid event name
                recognized by the Ultralytics callback system.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", lambda: print("Training started"))
            >>> model.clear_callback("on_train_start")
            >>> # All callbacks for 'on_train_start' are now removed

        Notes:
            - This method affects both custom callbacks added by the user and default callbacks
              provided by the Ultralytics framework.
            - After calling this method, no callbacks will be executed for the specified event
              until new ones are added.
            - Use with caution as it removes all callbacks, including essential ones that might
              be required for proper functioning of certain operations.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Reset all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        previously added. It iterates through all default callback events and replaces the current callbacks with the
        default ones.

        The default callbacks are defined in the 'callbacks.default_callbacks' dictionary, which contains predefined
        functions for various events in the model's lifecycle, such as on_train_start, on_epoch_end, etc.

        This method is useful when you want to revert to the original set of callbacks after making custom
        modifications, ensuring consistent behavior across different runs or experiments.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.add_callback("on_train_start", custom_function)
            >>> model.reset_callbacks()
            # All callbacks are now reset to their default functions
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset specific arguments when loading a PyTorch model checkpoint.

        This method filters the input arguments dictionary to retain only a specific set of keys that are
        considered important for model loading. It's used to ensure that only relevant arguments are preserved
        when loading a model from a checkpoint, discarding any unnecessary or potentially conflicting settings.

        Args:
            args (dict): A dictionary containing various model arguments and settings.

        Returns:
            (dict): A new dictionary containing only the specified include keys from the input arguments.

        Examples:
            >>> original_args = {"imgsz": 640, "data": "coco.yaml", "task": "detect", "batch": 16, "epochs": 100}
            >>> reset_args = Model._reset_ckpt_args(original_args)
            >>> print(reset_args)
            {'imgsz': 640, 'data': 'coco.yaml', 'task': 'detect'}
        """
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """
        Intelligently load the appropriate module based on the model task.

        This method dynamically selects and returns the correct module (model, trainer, validator, or predictor)
        based on the current task of the model and the provided key. It uses the task_map dictionary to determine
        the appropriate module to load for the specific task.

        Args:
            key (str): The type of module to load. Must be one of 'model', 'trainer', 'validator', or 'predictor'.

        Returns:
            (object): The loaded module class corresponding to the specified key and current task.

        Raises:
            NotImplementedError: If the specified key is not supported for the current task.

        Examples:
            >>> model = Model(task="detect")
            >>> predictor_class = model._smart_load("predictor")
            >>> trainer_class = model._smart_load("trainer")
        """
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(f"'{name}' model does not support '{mode}' mode for '{self.task}' task.") from e

    @property
    def task_map(self) -> dict:
        """
        Provide a mapping from model tasks to corresponding classes for different modes.

        This property method returns a dictionary that maps each supported task (e.g., detect, segment, classify)
        to a nested dictionary. The nested dictionary contains mappings for different operational modes
        (model, trainer, validator, predictor) to their respective class implementations.

        The mapping allows for dynamic loading of appropriate classes based on the model's task and the
        desired operational mode. This facilitates a flexible and extensible architecture for handling
        various tasks and modes within the Ultralytics framework.

        Returns:
            (Dict[str, Dict[str, Any]]): A dictionary mapping task names to nested dictionaries. Each nested dictionary
            contains mappings for 'model', 'trainer', 'validator', and 'predictor' keys to their respective class
            implementations for that task.

        Examples:
            >>> model = Model("yolo11n.pt")
            >>> task_map = model.task_map
            >>> detect_predictor = task_map["detect"]["predictor"]
            >>> segment_trainer = task_map["segment"]["trainer"]
        """
        raise NotImplementedError("Please provide task map for your model!")

    def eval(self):
        """
        Sets the model to evaluation mode.

        This method changes the model's mode to evaluation, which affects layers like dropout and batch normalization
        that behave differently during training and evaluation. In evaluation mode, these layers use running statistics
        rather than computing batch statistics, and dropout layers are disabled.

        Returns:
            (Model): The model instance with evaluation mode set.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> model.eval()
            >>> # Model is now in evaluation mode for inference
        """
        self.model.eval()
        return self

    def __getattr__(self, name):
        """
        Enable accessing model attributes directly through the Model class.

        This method provides a way to access attributes of the underlying model directly through the Model class
        instance. It first checks if the requested attribute is 'model', in which case it returns the model from
        the module dictionary. Otherwise, it delegates the attribute lookup to the underlying model.

        Args:
            name (str): The name of the attribute to retrieve.

        Returns:
            (Any): The requested attribute value.

        Raises:
            AttributeError: If the requested attribute does not exist in the model.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> print(model.stride)  # Access model.stride attribute
            >>> print(model.names)  # Access model.names attribute
        """
        return self._modules["model"] if name == "model" else getattr(self.model, name)
