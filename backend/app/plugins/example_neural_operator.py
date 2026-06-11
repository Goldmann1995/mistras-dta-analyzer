"""
Example plugin: Neural Operator stub.

This demonstrates the plugin interface for integrating neural operators
(e.g., DeepONet, FNO) or other ML models for AE signal analysis.

To create a real plugin:
1. Subclass PluginBase
2. Implement infer() with your model logic
3. Define a register() function at module level
"""
from backend.app.services.plugin_manager import PluginBase


class NeuralOperatorPlugin(PluginBase):
    name = "neural_operator"
    version = "0.1.0"
    description = "Neural operator plugin for AE signal prediction and analysis. Supports DeepONet, FNO, etc."

    def get_endpoints(self) -> list[str]:
        return [
            "POST /api/plugins/neural_operator/infer",
            "GET  /api/plugins/neural_operator/models",
        ]

    def infer(self, file_id: str, params: dict) -> dict:
        return {
            'status': 'stub',
            'message': 'Neural operator inference not yet implemented. Replace this with your model.',
            'supported_models': ['DeepONet', 'FNO', 'U-Net'],
            'params_received': params,
        }


def register(manager):
    manager.register(NeuralOperatorPlugin())
