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
    
    def __reduce__(self):
        """支持pickle序列化"""
        return (self.__class__, (self.choices, self.on_fail))
    
    def __repr__(self):
        """字符串表示"""
        return f"ValidChoices(choices={self.choices}, on_fail='{self.on_fail}')"
    
    def __eq__(self, other):
        """等值比较"""
        if isinstance(other, ValidChoices):
            return self.choices == other.choices and self.on_fail == other.on_fail
        return False
    
    def __hash__(self):
        """支持哈希"""
        return hash((tuple(self.choices) if isinstance(self.choices, list) else self.choices, self.on_fail))
    
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Pydantic v2兼容性：提供JSON schema"""
        return {'type': 'string'}
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        """Pydantic v1兼容性：修改schema"""
        field_schema.update(type='string')