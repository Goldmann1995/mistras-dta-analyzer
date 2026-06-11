import importlib
import os
from typing import Optional


class PluginBase:
    """Base class for all analysis plugins (neural operators, etc.)."""

    name: str = "base_plugin"
    version: str = "0.0.0"
    description: str = ""

    def get_info(self) -> dict:
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'endpoints': self.get_endpoints(),
        }

    def get_endpoints(self) -> list[str]:
        return []

    def infer(self, file_id: str, params: dict) -> dict:
        raise NotImplementedError


class PluginManager:
    def __init__(self):
        self._plugins: dict[str, PluginBase] = {}

    def register(self, plugin: PluginBase):
        self._plugins[plugin.name] = plugin

    def unregister(self, name: str):
        self._plugins.pop(name, None)

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        return self._plugins.get(name)

    def list_plugins(self) -> list[dict]:
        return [p.get_info() for p in self._plugins.values()]

    def load_plugins_from_dir(self, plugins_dir: str):
        if not os.path.isdir(plugins_dir):
            return
        for fname in os.listdir(plugins_dir):
            if fname.startswith('_') or not fname.endswith('.py'):
                continue
            module_name = fname[:-3]
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(plugins_dir, fname))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'register'):
                    module.register(self)
            except Exception as e:
                print(f"Failed to load plugin {fname}: {e}")


plugin_manager = PluginManager()
