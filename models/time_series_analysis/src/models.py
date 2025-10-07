from typing import Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
import json


class PathsConfig(BaseModel):
    path_to_generated: str
    path_to_provided: str
    save_path: str


class Config(BaseModel):
    paths: PathsConfig
    adg_code: Optional[str] = None
    end_date: str
    start_date: str
    time_col: str
    product_id_col: str
    target_col: str
    plot_temporal: bool = True
    plot_entity: bool = True
    TINs: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    anomaly_factor: float = 1.5
    entity_outlier_threshold: float = 10.0
    user: str
    time_range: Optional[List[str]] = None
    sub_category: Optional[str] = None
    category_score: Optional[float] = None
    brand_name: Optional[str] = None
    sub_category_ARM_1: Optional[str] = None
    aggregation_level: Optional[str] = None

    @field_validator("time_range")
    @classmethod
    def validate_time_range(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("time_range must contain exactly 2 dates")
        return v

    @model_validator(mode="after")
    def set_dates_from_time_range(self):
        if self.time_range and len(self.time_range) == 2:
            self.start_date = self.time_range[0]
            self.end_date = self.time_range[1]
        return self

    @classmethod
    def from_json_file(cls, file_path: str) -> "Config":
        """Load config from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        """Save config to JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

    def update_from_input(self, input_config: "InputConfig") -> "Config":
        """Update config with values from input config"""
        input_dict = input_config.model_dump(exclude_none=True)
        config_dict = self.model_dump()

        # Update config_dict with input_dict values
        for key, value in input_dict.items():
            if key == "time_range" and isinstance(value, list) and len(value) == 2:
                config_dict["start_date"] = value[0]
                config_dict["end_date"] = value[1]
                config_dict[key] = value
            else:
                config_dict[key] = value

        return Config(**config_dict)


class InputConfig(BaseModel):
    time_col: Optional[str] = None
    product_id_col: Optional[str] = None
    target_col: Optional[str] = None
    adg_code: Optional[str] = None
    TINs: Optional[Dict[str, Dict[str, List[str]]]] = None
    time_range: Optional[List[str]] = None
    user: Optional[str] = None
    sub_category_ARM_1: Optional[str] = None
    category_score: Optional[float] = None
    brand_name: Optional[str] = None
    plot_temporal: Optional[bool] = None
    plot_entity: Optional[bool] = None
    anomaly_factor: Optional[float] = None
    entity_outlier_threshold: Optional[float] = None

    @field_validator("time_range")
    @classmethod
    def validate_time_range(cls, v):
        if v is not None and len(v) != 2:
            raise ValueError("time_range must contain exactly 2 dates")
        return v

    @classmethod
    def from_json_file(cls, file_path: str) -> "InputConfig":
        """Load input config from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, file_path: str) -> None:
        """Save input config to JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(exclude_none=True), f, indent=4)
