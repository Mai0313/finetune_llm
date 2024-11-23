from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    huggingface_token: str = Field(
        ...,
        title="Huggingface Token",
        description="Input your huggingface token here, it will be used for downloading models and datasets.",
        validation_alias=AliasChoices("HUGGINGFACE_TOKEN", "huggingface_token"),
    )


config = Config()

__all__ = ["config"]
