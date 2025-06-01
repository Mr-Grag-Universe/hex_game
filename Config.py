import json
# import yaml
import os

class Config:
    def __init__(self, src):
        self.data = self.load_config(src)

    def load_config(self, src):
        if isinstance(src, dict):
            return src  # Если это словарь, просто возвращаем его

        if isinstance(src, str):
            # Проверяем, является ли строка путем к файлу
            if os.path.isfile(src):
                return self.load_from_file(src)
            else:
                # Если это строка, пытаемся определить формат
                return self.load_from_string(src)

        raise ValueError("Unsupported type for src. Must be dict, str, or a valid file path.")

    def load_from_file(self, filepath):
        _, ext = os.path.splitext(filepath)
        with open(filepath, 'r', encoding='utf-8') as file:
            if ext == '.json':
                return json.load(file)
            elif ext in ['.yaml', '.yml']:
                return None # yaml.safe_load(file)
            else:
                raise ValueError("Unsupported file format. Use .json or .yaml")

    def load_from_string(self, string):
        try:
            # Попробуем загрузить как JSON
            return json.loads(string)
        except json.JSONDecodeError:
            return None
            # try:
            #     # Если не удалось, попробуем загрузить как YAML
            #     return yaml.safe_load(string)
            # except yaml.YAMLError:
            #     raise ValueError("String is not valid JSON or YAML.")

    def get_data(self):
        return self.data