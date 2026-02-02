"""
RLM-Trans LLM Client
Multi-provider support: LM Studio, OpenAI, Google Gemini
"""
import os
import json
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config import LLMConfig


@dataclass
class LLMResponse:
    """Standardized LLM Response"""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track API costs"""
    root_calls: int = 0
    sub_calls: int = 0
    root_input_tokens: int = 0
    root_output_tokens: int = 0
    sub_input_tokens: int = 0
    sub_output_tokens: int = 0
    total_cost: float = 0.0
    
    def add_root_call(self, input_tokens: int, output_tokens: int, cost: float = 0.0):
        self.root_calls += 1
        self.root_input_tokens += input_tokens
        self.root_output_tokens += output_tokens
        self.total_cost += cost
        
    def add_sub_call(self, input_tokens: int, output_tokens: int, cost: float = 0.0):
        self.sub_calls += 1
        self.sub_input_tokens += input_tokens
        self.sub_output_tokens += output_tokens
        self.total_cost += cost
    
    def summary(self) -> Dict[str, Any]:
        return {
            "root_calls": self.root_calls,
            "sub_calls": self.sub_calls,
            "total_calls": self.root_calls + self.sub_calls,
            "root_tokens": self.root_input_tokens + self.root_output_tokens,
            "sub_tokens": self.sub_input_tokens + self.sub_output_tokens,
            "total_cost": self.total_cost
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def complete(self, messages: List[Dict], model: str, **kwargs) -> LLMResponse:
        """Send completion request"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test connection to provider"""
        pass


class LMStudioProvider(LLMProvider):
    """LM Studio Local Server Provider"""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url.rstrip("/")
        
    def complete(self, messages: List[Dict], model: str = "auto", **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        if model and model != "auto":
            payload["model"] = model
            
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            usage = data.get("usage", {})
            
            return LLMResponse(
                content=choice["message"]["content"],
                model=data.get("model", model),
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            )
        except requests.exceptions.Timeout:
            raise ConnectionError("LM Studio request timed out (120s). Check if LM Studio is running and responsive.")
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot connect to LM Studio at {self.base_url}. Is LM Studio running?")
        except Exception as e:
            raise ConnectionError(f"LM Studio request failed: {e}")
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except:
            return []
    
    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models in LM Studio"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            # LM Studio returns loaded models in /models endpoint
            return [m["id"] for m in data.get("data", [])]
        except:
            return []
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from LM Studio"""
        try:
            # LM Studio uses DELETE /models/{model_id} or POST /models/unload
            # Try the newer API first
            url = f"{self.base_url}/models/unload"
            response = requests.post(url, json={"model": model_id}, timeout=30)
            if response.status_code == 200:
                print(f"[LM Studio] Model '{model_id}' unloaded")
                return True
            
            # Fallback: try DELETE
            url = f"{self.base_url}/models/{model_id}"
            response = requests.delete(url, timeout=30)
            if response.status_code == 200:
                print(f"[LM Studio] Model '{model_id}' unloaded")
                return True
            
            return False
        except Exception as e:
            print(f"[LM Studio] Failed to unload model: {e}")
            return False
    
    def unload_all_models(self) -> bool:
        """Unload all currently loaded models"""
        loaded = self.get_loaded_models()
        success = True
        for model_id in loaded:
            if not self.unload_model(model_id):
                success = False
        return success
    
    def load_model(self, model_id: str) -> bool:
        """Load a specific model in LM Studio"""
        try:
            # LM Studio uses POST /models/load
            url = f"{self.base_url}/models/load"
            response = requests.post(url, json={"model": model_id}, timeout=120)
            if response.status_code == 200:
                print(f"[LM Studio] Model '{model_id}' loaded")
                return True
            else:
                print(f"[LM Studio] Failed to load model: {response.text}")
                return False
        except Exception as e:
            print(f"[LM Studio] Failed to load model: {e}")
            return False
    
    def ensure_model_loaded(self, model_id: str) -> bool:
        """
        Ensure the specified model is loaded.
        If not loaded, unload all current models and load the specified one.
        """
        if not model_id or model_id == "auto":
            return True  # Use whatever is loaded
        
        loaded = self.get_loaded_models()
        
        # Check if already loaded
        if model_id in loaded:
            print(f"[LM Studio] Model '{model_id}' already loaded")
            return True
        
        # Need to load - first unload all
        print(f"[LM Studio] Model '{model_id}' not loaded. Switching models...")
        if loaded:
            print(f"[LM Studio] Unloading current models: {loaded}")
            self.unload_all_models()
        
        # Load the new model
        return self.load_model(model_id)


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider"""
    
    # Pricing per 1K tokens (approximate)
    PRICING = {
        "gpt-4o": {"input": 0.0025, "output": 0.01},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1"
        
    def complete(self, messages: List[Dict], model: str = "gpt-4o-mini", **kwargs) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            usage = data.get("usage", {})
            
            # Calculate cost
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            pricing = self.PRICING.get(model, {"input": 0, "output": 0})
            cost = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
            
            return LLMResponse(
                content=choice["message"]["content"],
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"OpenAI API error: {e.response.text}")
        except Exception as e:
            raise ConnectionError(f"OpenAI request failed: {e}")
    
    def list_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    
    def test_connection(self) -> bool:
        try:
            url = f"{self.base_url}/models"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(url, headers=headers, timeout=10)
            return response.status_code == 200
        except:
            return False


class GeminiProvider(LLMProvider):
    """Google Gemini API Provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
    def complete(self, messages: List[Dict], model: str = "gemini-2.0-flash", **kwargs) -> LLMResponse:
        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": content}]})
        
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 4096),
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            usage = data.get("usageMetadata", {})
            
            return LLMResponse(
                content=content,
                model=model,
                input_tokens=usage.get("promptTokenCount", 0),
                output_tokens=usage.get("candidatesTokenCount", 0),
            )
        except requests.exceptions.HTTPError as e:
            raise ConnectionError(f"Gemini API error: {e.response.text}")
        except Exception as e:
            raise ConnectionError(f"Gemini request failed: {e}")
    
    def list_models(self) -> List[str]:
        return ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
    
    def test_connection(self) -> bool:
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            return response.status_code == 200
        except:
            return False


class LLMClient:
    """Unified LLM Client with multi-provider support"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.provider = self._create_provider()
        self.cost_tracker = CostTracker()
        
    def _create_provider(self) -> LLMProvider:
        """Create appropriate provider based on config"""
        if self.config.provider == "openai":
            if not self.config.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return OpenAIProvider(self.config.openai_api_key)
        
        elif self.config.provider == "gemini":
            if not self.config.gemini_api_key:
                raise ValueError("Gemini API key not configured")
            return GeminiProvider(self.config.gemini_api_key)
        
        else:  # lmstudio (default)
            return LMStudioProvider(self.config.lm_studio_url)
    
    def complete(self, messages: List[Dict], model: Optional[str] = None, 
                 is_sub_call: bool = False, **kwargs) -> LLMResponse:
        """Send completion request and track costs"""
        if model is None:
            model = self.config.sub_model if is_sub_call else self.config.root_model
            
        response = self.provider.complete(messages, model or "auto", **kwargs)
        
        # Track costs
        if is_sub_call:
            self.cost_tracker.add_sub_call(
                response.input_tokens, response.output_tokens, response.cost
            )
        else:
            self.cost_tracker.add_root_call(
                response.input_tokens, response.output_tokens, response.cost
            )
            
        return response
    
    def list_models(self) -> List[str]:
        return self.provider.list_models()
    
    def test_connection(self) -> bool:
        return self.provider.test_connection()
    
    def cost_summary(self) -> Dict[str, Any]:
        return self.cost_tracker.summary()
    
    def reset_costs(self):
        self.cost_tracker = CostTracker()
    
    def ensure_model_loaded(self, model_id: str) -> bool:
        """
        Ensure the specified model is loaded (LM Studio only).
        Returns True if successful or not applicable.
        """
        if isinstance(self.provider, LMStudioProvider):
            return self.provider.ensure_model_loaded(model_id)
        return True  # Not applicable for other providers
    
    def get_loaded_models(self) -> List[str]:
        """Get currently loaded models (LM Studio only)"""
        if isinstance(self.provider, LMStudioProvider):
            return self.provider.get_loaded_models()
        return self.provider.list_models()
