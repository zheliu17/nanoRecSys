from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "data/ml-20m"
    embed_dim: int = 64

    @property
    def EMBED_DIM(self) -> int:
        return self.embed_dim

    def DATA_PATH(self) -> str:
        return self.data_path

    @DATA_PATH.setter
    def DATA_PATH(self, value: str) -> None:
        self.data_path = value


settings = Settings()
