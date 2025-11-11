from agieval.core.run.runner_type import RunnerType
from agieval.entity.eval_config import EvalConfig
from agieval.entity.flow_config import BenchmarkConfig, ContextParam, FlowStage, PluginConfig
from agieval.entity.global_param import GlobalParam
from agieval.common.param_utils import load_model_api_config

benchmark_name = "MMLU-Pro"

eval_config: EvalConfig = EvalConfig(
    runner=RunnerType.DATA_PARALLEL,
    benchmark_config_template=True,
    dataset_files=f"datasets/{benchmark_name}/",
    benchmark_config="",
    flow_config_file="",
    work_dir=f"result/{benchmark_name}",
    data_parallel=2,
    global_param=GlobalParam(),
    plugin_param=ContextParam(
        **load_model_api_config()
    )
)

benchmark_config_template = BenchmarkConfig(
    benchmark="",
    location_test="",
    use_cache=False,
    flow_config_file="",
    flow_stages=[
        FlowStage(
            plugin_implement="SimpleDataProcessor",
            plugins=[
                PluginConfig(
                    plugin_implement="GenerationScenario"
                ),
            ]
        ),
        FlowStage(
            plugin_implement="SimpleInferProcessor",
            plugins=[
                PluginConfig(
                    plugin_implement="LiteLLMModel"
                ),
                PluginConfig(
                    plugin_implement="SingleRoundTextAgent"
                )
            ]
        ),
        FlowStage(
            plugin_implement="ScoreInferProcessor",
            plugins=[
                PluginConfig(
                    plugin_implement="ScoreOpenaiHttpModel"
                ),
                PluginConfig(
                    plugin_implement="ModelScoreZeroshotV2Agent"
                )
            ]
        ),
        FlowStage(
            plugin_implement="SimpleMetricsProcessor",
            plugins=[
                PluginConfig(
                    plugin_implement="ModelEvalZeroshotV2Metrics"
                ),
            ]
        )
    ]
)