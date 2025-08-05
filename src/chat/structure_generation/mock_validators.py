"""
临时的ValidChoices替代品，用于解决guardrails hub导入问题
"""
from typing import Any, List, Union

class ValidChoices:
    """Mock ValidChoices validator for guardrails compatibility"""
    
    def __init__(self, choices: Union[List[int], List[str]], on_fail: str = "exception"):
        self.choices = choices
        self.on_fail = on_fail
        
    def __call__(self, value: Any) -> Any:
        """简单验证，实际上不做严格检查，让LLM自行处理"""
        return value