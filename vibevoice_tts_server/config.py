from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "VIBEVOICE_TTS_"}

    host: str = "0.0.0.0"
    port: int = 8100
    model_id: str = "microsoft/VibeVoice-7B-hf"
    cache_dir: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    idle_timeout: int = 300
    default_voice: str = "alloy"
    cfg_scale: float = 1.3
    n_diffusion_steps: int = 10
    max_new_tokens: int = 0  # 0 = unlimited (model stops naturally); tokens are audio frames at 7.5 Hz
    quantize_4bit: bool = False
    log_level: str = "info"
