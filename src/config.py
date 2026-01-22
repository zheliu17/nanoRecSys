from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    # Paths
    project_root: Path = Path(__file__).parent.parent.absolute()
    data_dir: Path = project_root / "data"
    raw_data_dir: Path = data_dir / "ml-20m"
    processed_data_dir: Path = data_dir / "processed"
    artifacts_dir: Path = project_root / "artifacts"

    # Dataset
    ml_20m_url: str = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    min_rating: float = 3.5  # Treat ratings >= 3.5 as "positive" interactions
    min_user_interactions: int = 5  # Filter users with too few interactions

    # Model params
    embed_dim: int = 64

    model_config = ConfigDict(env_file=".env")


settings = Settings()

# Ensure directories exist
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
