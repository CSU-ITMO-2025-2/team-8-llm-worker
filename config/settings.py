import enum
import functools
import os
from typing import Dict, Any, Set

import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class _LoadConfig(BaseSettings):

    KAFKA_SERVERS: str

    # AM_IN_DOCKER_COMPOSE: bool = False

    EXTRA_PARAMS: Dict[str, Any] = {}  # Словарь для хранения дополнительных параметров

    def __init__(self, **values):
        super().__init__(**values)
        self._load_extra_params()


    # @model_validator(mode='after')
    # def check_in_docker_compose(self) -> Self:
    #     if self.AM_IN_DOCKER_COMPOSE:
    #         self.DB_HOST = 'postgres_container'
    #     return self

    def _load_extra_params(self):
        """
        Загрузка EXTRA параметров из .env файла
        Если параметра нет в объекте, он добавляется в словарь EXTRA_PARAMS
        """

        _env_file = self.model_config.get('env_file')
        defined_field_names: Set[str] = set(self.model_fields.keys())

        if _env_file and os.path.exists(_env_file):
            env_vars_from_file = dotenv.dotenv_values(_env_file)

            for key, value in env_vars_from_file.items():
                if key not in defined_field_names:
                    self.EXTRA_PARAMS[key] = value

    model_config = SettingsConfigDict(env_file=f"{os.path.dirname(os.path.abspath(__file__))}\\.env.test", extra='ignore')


class Settings:
    __KAFKA_SERVERS: str


    __loaded: bool = False

    @classmethod
    def __load__(cls):
        settings = _LoadConfig()
        cls.__KAFKA_SERVERS = settings.KAFKA_SERVERS

        cls.__EXTRA_PARAMS = settings.EXTRA_PARAMS


        cls.__loaded = True

    @staticmethod
    def __check_loaded(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not Settings.__loaded:
                Settings.__load__()
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    @__check_loaded
    def KAFKA_SERVERS(cls) -> str:
        return cls.__KAFKA_SERVERS


    @classmethod
    @__check_loaded
    def EXTRA_PARAMS(cls) -> dict:
        return cls.__EXTRA_PARAMS


if __name__ == '__main__':
    print(Settings.EXTRA_PARAMS())
    pass
